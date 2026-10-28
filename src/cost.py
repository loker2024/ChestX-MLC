"""赛题第 7 条：端到端单张图片推理耗时评测脚本（`cost.py`）。

用法（参数顺序与规则附录一致）：

    python src/cost.py <submission.csv> <pictures_100> <pictures_5>

- `submission.csv`：最终提交的提交文件，用来做预测一致性比对；
- `pictures_100`：100 张计时图目录，结构与测试集一致
  （`<subject_id>/<study_id>/<pic_id>.jpg`）；
- `pictures_5`：5 张预热图目录，结构与上面相同，预热不计入成绩。

口径（对应规则第 7 条）：

1. batch=1 串行，每张图计时包括读文件、预处理、前向、后处理和 CSV 写出，
   模型加载/首次初始化不计入；多图 study 的单图结果只是候选，无法判定最终耗时；
2. 先跑 5 张预热图（不计入成绩），再把 100 张图连续重复 3 轮；
3. 每张图先对 3 轮取均值，再对 100 张取平均，与 100ms 上限比较；
4. 前后调用与 `run_infer.py` 完全相同的推理函数（`infer.predict`），
   最后把现场按 study 聚合的概率与 `submission.csv` 比对。

精度：默认 fp32；GPU 上可用 `--precision fp16`（FP16 需在提交说明里声明）。

退出码：0 表示平均耗时未超上限且预测与 submission.csv 一致；1 表示超限或
不一致；2 表示参数/文件/配置错误或多图 study 最终耗时无法判定。

完整测试集索引（配置里的 `data.test.csv` / `data.test.images`）可用时，脚本会
核对抽样图的 Subject_id 与文件名，并补齐同 study 的其他图片后再比对；赛题附录 A
的提交包不要求携带测试集，评测环境通常读不到这些文件，此时退化为按
`pictures_100` 自身分组判断，打印 `[WARN]` 后继续，报告里
`final_timing_verifiable=false` 表示该结论未经完整索引核实。

附加参数（可选）：`--config`、`--checkpoint`、`--device`、`--precision`、
`--rounds`、`--threshold`、`--tolerance`、`--report`。默认情况下会依次尝试
仓库 `configs/default.yaml`、提交包内的 `config.yaml`、checkpoint 里存档的配置；
权重会依次尝试 `--checkpoint`、`COST_CHECKPOINT` 环境变量、submission.csv 同目录
的 `best.pt` / `model/best.pt`、仓库根目录的 `best.pt` / `model/*.pt`、以及
`outputs/*/best.pt`（按修改时间倒序）。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import tempfile
from pathlib import Path
from typing import Sequence

# 仓库中脚本位于 src/；提交包按赛题要求复制到根目录，并保留 src/ 依赖。
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "src" else SCRIPT_DIR
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch  # noqa: E402

from common.checkpoint import load_checkpoint  # noqa: E402
from common.config import Config, config_from_dict, load_config  # noqa: E402
from common.utils import resolve_device  # noqa: E402
from data.index import StudyRecord, read_split  # noqa: E402
from common.constants import CLASS_IDS  # noqa: E402
from infer.submission import SUBMISSION_COLUMNS  # noqa: E402
from infer.predict import (  # noqa: E402
    build_eval_inference_transform,
    load_checkpoint_model,
    predict_image,
    predict_study,
)
from infer.profiling import (  # noqa: E402
    COST_LIMIT_MS,
    DEFAULT_ROUNDS,
    DEFAULT_TOLERANCE,
    DEFAULT_WARMUP_IMAGES,
    ComparisonReport,
    TimingSummary,
    compare_study_predictions,
    format_conclusion,
    group_images_by_study,
    list_images,
    parse_image_identity,
    read_submission,
    select_prediction_rows,
    summarize_timings,
)

# 配置文件查找顺序：仓库默认配置 → 提交包根目录的 config.yaml。
CONFIG_CANDIDATES = ("configs/default.yaml", "config.yaml")
# checkpoint 查找顺序（相对 submission.csv 所在目录或仓库根目录）。
CHECKPOINT_CANDIDATES = ("best.pt", "model/best.pt")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析命令行参数；位置参数顺序就是规则里的调用方式。"""
    parser = argparse.ArgumentParser(
        description="按赛题第 7 条测量单张图片端到端推理耗时，并与 submission.csv 比对"
    )
    parser.add_argument("submission", help="最终提交的 submission.csv 路径")
    parser.add_argument(
        "pictures_100", help="100 张计时图目录（结构 <subject_id>/<study_id>/<pic_id>.jpg）"
    )
    parser.add_argument("pictures_5", help="5 张预热图目录（结构同上）")
    parser.add_argument("--config", help="配置文件路径；缺省按自动顺序查找")
    parser.add_argument("--checkpoint", help="模型权重路径；缺省按自动顺序查找")
    parser.add_argument(
        "--device", help="auto / cpu / cuda / cuda:0 / mps，缺省取配置里的 device"
    )
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16"),
        default="fp32",
        help="推理精度，默认 fp32；fp16 需在提交说明中声明",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=DEFAULT_ROUNDS,
        help=f"连续重复轮数，规则要求 {DEFAULT_ROUNDS}，默认 {DEFAULT_ROUNDS}",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="后处理阈值，缺省用配置里的 eval.threshold",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help=f"与 submission.csv 比对时允许的最大概率绝对差，默认 {DEFAULT_TOLERANCE:g}",
    )
    parser.add_argument("--report", help="把计时与比对明细写成 JSON 报告")
    return parser.parse_args(argv)


def resolve_checkpoint(explicit: str | None, submission_path: Path) -> Path:
    """按「显式参数 → 环境变量 → 提交包 → outputs」的顺序找模型权重。"""
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"--checkpoint 指向的文件不存在：{path}")
        return path

    candidates: list[Path] = []
    env_checkpoint = os.environ.get("COST_CHECKPOINT")
    if env_checkpoint:
        candidates.append(Path(env_checkpoint))
    for base in (submission_path.parent, ROOT):
        candidates.extend(base / relative for relative in CHECKPOINT_CANDIDATES)
        model_dir = base / "model"
        if model_dir.is_dir():
            candidates.extend(sorted(model_dir.glob("*.pt")))
    outputs_dir = ROOT / "outputs"
    if outputs_dir.is_dir():
        candidates.extend(
            sorted(
                outputs_dir.glob("*/best.pt"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        )

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "找不到模型权重：请用 --checkpoint 指定（也检查了 submission.csv 同目录的 "
        "best.pt / model/best.pt、仓库根目录 best.pt / model/*.pt、outputs/*/best.pt）"
    )


def resolve_config(
    explicit: str | None, submission_path: Path, checkpoint_path: Path
) -> Config:
    """按「显式参数 → 仓库默认配置 → 提交包配置 → checkpoint 存档配置」找配置。"""
    if explicit:
        return load_config(explicit)

    for base in (ROOT, submission_path.parent):
        for relative in CONFIG_CANDIDATES:
            candidate = base / relative
            if candidate.is_file():
                return load_config(candidate)

    # 提交包里可能只有权重没有配置：checkpoint 里存了训练时的 config，可还原。
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    raw_config = checkpoint.get("config")
    if raw_config:
        return config_from_dict(raw_config)
    raise FileNotFoundError(
        "找不到配置文件，且 checkpoint 里没有存档配置：请用 --config 指定"
    )


def load_complete_test_index(config: Config) -> dict[str, StudyRecord] | None:
    """读取配置里的完整测试集索引（study -> 记录）；读不到时返回 None。

    赛题附录 A 的提交包不要求携带测试集，评测环境直接跑 `cost.py` 时通常没有
    `data.test.csv` 与测试图片。这时只能退化为按 `pictures_100` 自身分组判断，
    无法确认抽样目录之外是否还有同 study 图片，结果只能作为候选；调用方据此把
    报告里的 `final_timing_verifiable` 标为 false。
    """
    test = config.data.test
    if not Path(test.csv).is_file() or not Path(test.images).is_dir():
        print(
            f"[WARN] 找不到完整测试集索引（{test.csv} / {test.images}），"
            "退化为按 pictures_100 自身分组比对，单图耗时只能作为候选结果"
        )
        return None
    try:
        records = read_split(test.csv, test.images, labeled=False)
    except (FileNotFoundError, OSError) as error:
        print(
            f"[WARN] 无法读取完整测试集索引（{error}），"
            "退化为按 pictures_100 自身分组比对，单图耗时只能作为候选结果"
        )
        return None
    return {record.study_id: record for record in records}


def _synchronize(device: torch.device) -> None:
    """把异步加速设备上的工作排空，保证计时区间包住全部计算。

    CUDA/MPS 的 kernel 是异步下发的，不 synchronize 就会把耗时算到下一次调用头上。
    CPU 上没有异步队列，直接返回。
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _load_model(
    config: Config, checkpoint_path: Path, device: torch.device, precision: str
) -> torch.nn.Module:
    """加载与 run_infer.py 相同的模型；fp16 时把参数转成半精度。"""
    model = load_checkpoint_model(config, checkpoint_path, device)
    if precision == "fp16":
        model = model.half()
    return model.eval()


def _time_one_image(
    model: torch.nn.Module,
    image_path: Path,
    transform,
    device: torch.device,
    threshold: float | None,
    output_path: Path,
) -> float:
    """测一张图的端到端毫秒数（读图 + 预处理 + 前向 + sigmoid + 后处理）。

    写出包含患者、检查、类别与概率的 CSV；多图检查此处只是单图候选结果。
    """
    _synchronize(device)
    started = time.perf_counter()
    probabilities = predict_image(model, image_path, transform, device)
    if any(not math.isfinite(float(value)) or not 0 <= float(value) <= 1 for value in probabilities):
        raise ValueError(f"图片 {image_path} 的概率必须为 [0, 1] 内的有限数")
    rows = ([(class_id, float(probabilities[class_id - 1])) for class_id in CLASS_IDS]
            if threshold is None else select_prediction_rows(probabilities, threshold))
    identity = parse_image_identity(image_path)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(SUBMISSION_COLUMNS)
        writer.writerows((identity.subject_id, identity.study_id, class_id, probability)
                         for class_id, probability in rows)
    _synchronize(device)
    return (time.perf_counter() - started) * 1000.0


def _format_report(
    config: Config,
    checkpoint_path: Path,
    device: torch.device,
    precision: str,
    threshold: float,
    timing: TimingSummary,
    comparison: ComparisonReport,
    images: Sequence[Path],
    warmup: Sequence[Path],
    final_timing: bool,
    final_timing_verifiable: bool,
) -> dict:
    """汇总成可写入 JSON 的报告（提交材料里可附上）。"""
    return {
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "precision": precision,
        "threshold": threshold,
        "config": config.to_dict(),
        "pictures_100": len(images),
        "warmup_images": len(warmup),
        "timing_ms": {
            "rounds": timing.rounds,
            "images": timing.images,
            "mean": timing.mean_ms,
            "median": timing.median_ms,
            "p95": timing.p95_ms,
            "min": timing.min_ms,
            "max": timing.max_ms,
            "total": timing.total_ms,
            "limit": timing.limit_ms,
            "within_limit": timing.within_limit if final_timing else None,
            "final_timing_verifiable": final_timing_verifiable,
        },
        "comparison": {
            "filtered_submission": comparison.filtered_submission,
            "checked_studies": comparison.checked_studies,
            "checked_images": comparison.checked_images,
            "inconsistent_studies": comparison.inconsistent_studies,
            "inconsistent_images": comparison.inconsistent_images,
            "max_abs_diff": comparison.max_abs_diff,
            "tolerance": comparison.tolerance,
            "issues": [
                {
                    "study_id": issue.study_id,
                    "reason": issue.reason,
                    "max_abs_diff": issue.max_abs_diff,
                }
                for issue in comparison.inconsistent
            ],
        },
    }


def _run(args: argparse.Namespace) -> int:
    """真正的主流程；异常由 `main` 统一转成退出码 2。"""
    submission_path = Path(args.submission).resolve()
    submission = read_submission(submission_path)
    images = list_images(args.pictures_100)
    warmup = list_images(args.pictures_5)

    checkpoint_path = resolve_checkpoint(args.checkpoint, submission_path)
    config = resolve_config(args.config, submission_path, checkpoint_path)
    device = resolve_device(args.device or config.device)
    threshold = config.eval.threshold if args.threshold is None else args.threshold
    if not (math.isfinite(threshold) and 0.0 <= threshold <= 1.0):
        raise ValueError(f"--threshold 必须落在 [0, 1]，当前为 {threshold!r}")
    if args.rounds <= 0:
        raise ValueError(f"--rounds 必须为正整数，当前为 {args.rounds}")

    groups = group_images_by_study(images)
    # 完整 test 索引能证明 study 的图片集合，并把抽样目录里没被抽到的同 study 图片补上；
    # 评测环境没有测试集文件时退化为按抽样目录自身分组，只作为候选结果。
    records_by_study = load_complete_test_index(config)
    if records_by_study is not None:
        for path in images:
            identity = parse_image_identity(path)
            record = records_by_study.get(identity.study_id)
            if (
                record is None
                or record.subject_id != identity.subject_id
                or path.stem not in {candidate.stem for candidate in record.image_paths}
            ):
                raise ValueError(f"计时图片 {path} 与完整测试集索引不一致，无法验证")
        final_timing = all(
            len(records_by_study[study_id].image_paths) == 1 for study_id in groups
        )
        expected_counts = {
            study_id: len(records_by_study[study_id].image_paths) for study_id in groups
        }
        full_paths_by_study = {
            study_id: [
                {candidate.stem: candidate for candidate in groups[study_id]}.get(
                    candidate.stem, candidate
                )
                for candidate in records_by_study[study_id].image_paths
            ]
            for study_id in groups
        }
        study_subjects = {
            study_id: records_by_study[study_id].subject_id for study_id in groups
        }
    else:
        # 只能相信抽样目录本身：每个 study 只出现一次就当作单图 study。
        final_timing = all(len(paths) == 1 for paths in groups.values())
        expected_counts = {study_id: len(paths) for study_id, paths in groups.items()}
        full_paths_by_study = dict(groups)
        study_subjects = {
            study_id: parse_image_identity(paths[0]).subject_id
            for study_id, paths in groups.items()
        }
    final_timing_verifiable = records_by_study is not None

    model = _load_model(config, checkpoint_path, device, args.precision)
    transform = build_eval_inference_transform(config)

    print(f"[INFO] submission：{submission_path}（{len(submission.study_ids)} 个 study）")
    print(f"[INFO] checkpoint：{checkpoint_path}")
    print(f"[INFO] 设备：{device}；推理精度：{args.precision}；阈值：{threshold:g}")
    print(
        f"[INFO] 计时目录：{args.pictures_100}（{len(images)} 张图）；"
        f"预热目录：{args.pictures_5}（{len(warmup)} 张图）"
    )
    if len(images) != 100:
        print(f"[WARN] 规则要求 100 张计时图，当前为 {len(images)} 张（调试可用，正式成绩以 100 张为准）")
    if len(warmup) != DEFAULT_WARMUP_IMAGES:
        print(f"[WARN] 规则要求 5 张预热图，当前为 {len(warmup)} 张")
    if args.precision == "fp16" and device.type == "cpu":
        print("[WARN] CPU 上 fp16 通常比 fp32 慢，且部分算子可能不支持；正式测速请在 GPU 上跑")

    # 预热：首次前向会触发 kernel 编译/缓存，规则明确这段不计入成绩。
    for path in warmup:
        probabilities = predict_image(model, path, transform, device)
        select_prediction_rows(probabilities, threshold)
    print(f"[INFO] 预热完成：{len(warmup)} 张图（不计入成绩）")

    # 提交文件里某个 study 少于 10 行，说明提交时按阈值过滤过，写出的行也要过滤；
    # 否则按全 10 类写出，保证与 submission.csv 的口径一致。
    row_thresholds = {
        path: (
            threshold
            if len(submission.probabilities.get(parse_image_identity(path).study_id, {}))
            < len(CLASS_IDS)
            else None
        )
        for path in images
    }

    # 正式计时：外层 3 轮、内层 100 张，batch=1 串行，与规则一致。
    round_times_ms: list[list[float]] = []
    with tempfile.TemporaryDirectory(prefix="chestx-cost-") as directory:
        output_path = Path(directory) / "prediction.csv"
        for round_index in range(args.rounds):
            times = [
                _time_one_image(
                    model, path, transform, device, row_thresholds[path], output_path
                )
                for path in images
            ]
            round_times_ms.append(times)
            print(
                f"[INFO] 第 {round_index + 1}/{args.rounds} 轮完成："
                f"本轮均值 {sum(times) / len(times):.3f} ms"
            )
    timing = summarize_timings(round_times_ms, limit_ms=COST_LIMIT_MS)

    print(
        f"[INFO] 单张耗时：均值 {timing.mean_ms:.3f} ms，中位数 {timing.median_ms:.3f} ms，"
        f"p95 {timing.p95_ms:.3f} ms，最小 {timing.min_ms:.3f} ms，最大 {timing.max_ms:.3f} ms；"
        f"上限 {timing.limit_ms:.0f} ms"
    )

    # 一致性校验：把 100 张图按 study 分组，用同一套推理函数聚合，再和提交文件比对。
    # 这一步不计时（规则只要求提交结果一致，不要求它属于耗时区间）。
    study_probabilities = {
        study_id: predict_study(model, paths, transform, device)
        for study_id, paths in full_paths_by_study.items()
    }
    comparison = compare_study_predictions(
        study_probabilities,
        submission,
        threshold=threshold,
        tolerance=args.tolerance,
        image_counts=expected_counts,
        study_subjects=study_subjects,
    )
    print(
        f"[INFO] 一致性：比对 {comparison.checked_images} 张图 / "
        f"{comparison.checked_studies} 个 study；最大概率差 {comparison.max_abs_diff:.6g}；"
        f"不一致 {comparison.inconsistent_studies} 个 study"
        f"（{comparison.inconsistent_images} 张图）"
    )
    for issue in comparison.inconsistent[:5]:
        print(f"[WARN] 不一致：study {issue.study_id}：{issue.reason}")

    print(
        format_conclusion(
            timing,
            comparison,
            precision=args.precision,
            warmup_images=len(warmup),
            final_timing=final_timing,
        )
    )

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        payload = _format_report(
            config,
            checkpoint_path,
            device,
            args.precision,
            threshold,
            timing,
            comparison,
            images,
            warmup,
            final_timing,
            final_timing_verifiable,
        )
        report_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[INFO] 明细报告：{report_path}")

    if not comparison.consistent:
        print(
            f"[ERROR] 预测结果与 submission.csv 有 {comparison.inconsistent_studies} 个 study 不一致",
            file=sys.stderr,
        )
        return 1
    if not final_timing:
        print("[ERROR] 含多图 study：单图候选结果耗时不含完整聚合，无法判定最终端到端耗时", file=sys.stderr)
        return 2
    if not timing.within_limit:
        print(
            f"[ERROR] 平均单张耗时 {timing.mean_ms:.3f} ms 超过上限 {timing.limit_ms:.0f} ms",
            file=sys.stderr,
        )
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口：参数/文件/配置类错误打印 `[ERROR]` 并返回 2。"""
    args = parse_args(argv)
    try:
        return _run(args)
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
