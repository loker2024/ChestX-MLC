"""赛题第 7 条：端到端单次检查（study）推理耗时评测脚本（`cost.py`）。

用法（参数顺序与规则附录一致）：

    python src/cost.py <submission.csv> <pictures_100> <pictures_5>

计时口径（2026-11-03 官方答疑群确认，覆盖旧版「单图计时」口径）：

1. 最小计时单元是单个 `study_id`，`batch=1` 以 study 为单位；一次检查的
   1~7 张图在同一次前向里一起处理，输出该检查的最终预测，而不是逐图计时；
2. 单次计时覆盖：读该 study 的全部图片 → 解码 → 预处理 → 前向 → study 聚合
   → 后处理 → 写出每张图的一行（Subject_id, Study_id, Image_id, p1..p10）
   → flush；
3. 每轮只打开一次预测文件、所有 study 复用同一句柄，轮末关闭；文件打开、
   写入、flush、关闭全部计入总时长，不做 IO 豁免；
4. 预热使用 `pictures_5` 的 5 个 study（其全部图片同批前向），不计入成绩，
   且必须与计时 study 不相交；
5. 初始化（submission.csv 与视角 CSV 的元数据读取、study 分组、路径索引）
   不计时，但只碰元数据、一次完成、不能用循环内惰性补算；脚本打印 `init_ms`
   并做自设上限校验，禁止把解码/前向/预测藏进初始化；
6. 连续 3 轮，先对每个 study 取 3 轮均值，再对全部 study 取平均，与 100 ms
   上限比较。

多模型：当前 submission 若是多模型融合，必须用同样的融合口径计时，否则既会
低估端到端耗时，也无法复现 submission.csv。`--checkpoint` 支持传入多个权重，
或在 submission.csv 同目录放 `cost_ensemble.json`：

    {"fusion": "probability", "weights": [1, 1, 1, 1],
     "checkpoints": ["weights/a.pt", "weights/b.pt", ...]}

计时循环对同一 study 只解码一次图像；相同预处理签名的模型共享一次 transform，
不同分辨率（如 320 与 384）放在同一个线程池里并行缩放，随后逐模型前向，最后
按 `infer.ensemble` 的同一公式融合；模型加载、权重读取不计入成绩。

退出码：0 表示单 study 平均耗时未超上限且预测与 submission.csv 一致；
1 表示超限或预测不一致；2 表示参数/文件/配置/权重错误。

附加参数（可选）：`--checkpoint`、`--ensemble`、`--weights`、`--fusion`、
`--config`、`--device`、`--precision`、`--rounds`、`--threshold`、
`--tolerance`、`--views-csv`、`--prediction-out`、`--preprocess-workers`、
`--init-limit-ms`、`--max-model-mb`、`--report`。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

# 仓库中脚本位于 src/；提交包按赛题要求复制到根目录，并保留 src/ 依赖。
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "src" else SCRIPT_DIR
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from common.checkpoint import load_checkpoint  # noqa: E402
from common.config import Config, config_from_dict, load_config  # noqa: E402
from common.utils import resolve_device  # noqa: E402
from infer.ensemble import fuse_probability_vectors, normalize_weights  # noqa: E402
from infer.predict import (  # noqa: E402
    StudyModelSpec,
    build_eval_inference_transform,
    load_checkpoint_model,
    predict_study_multi,
    warmup_study_shapes,
)
from infer.profiling import (  # noqa: E402
    COST_LIMIT_MS,
    COST_OUTPUT_COLUMNS,
    DEFAULT_ROUNDS,
    DEFAULT_TOLERANCE,
    DEFAULT_WARMUP_STUDIES,
    StudyJob,
    build_study_jobs,
    compare_study_predictions,
    format_conclusion,
    list_images,
    read_submission,
    read_view_map,
    summarize_timings,
    views_for_job,
)

# 配置文件查找顺序：仓库默认配置 → 提交包根目录的 config.yaml。
CONFIG_CANDIDATES = ("configs/default.yaml", "config.yaml")
# checkpoint 查找顺序（相对 submission.csv 所在目录或仓库根目录）。
CHECKPOINT_CANDIDATES = ("best.pt", "model/best.pt")
# 多模型融合清单的文件名，放在 submission.csv 同目录或提交包根目录。
ENSEMBLE_MANIFEST = "cost_ensemble.json"
# 视角元数据文件名，放在 submission.csv 同目录；内容与 test.csv 的三列一致。
VIEWS_FILENAME = "cost_views.csv"
FUSION_METHODS = ("probability", "logit")
# 初始化自设上限：只做元数据读取、分组与路径索引，2 秒足够宽裕；超出只警告，
# 因为官方没有公布该上限，脚本不替主办方判定失败。
DEFAULT_INIT_LIMIT_MS = 2000.0
# 赛题第 8 条：权重与必要配套文件解压后合计 ≤ 500 MB。
DEFAULT_MAX_MODEL_MB = 500.0


@dataclass(frozen=True)
class ModelSpec:
    """一个待加载的模型：权重路径 + 训练时存档的配置。"""

    checkpoint: Path
    config: Config


@dataclass(frozen=True)
class LoadedModel:
    """已经放到 device 上、可以前向的模型及其预处理。"""

    spec: ModelSpec
    model: torch.nn.Module
    transform: Callable
    transform_key: tuple


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析命令行参数；位置参数顺序就是规则里的调用方式。"""
    parser = argparse.ArgumentParser(
        description="按赛题第 7 条测量单次检查（study）的端到端推理耗时，并与 submission.csv 比对"
    )
    parser.add_argument("submission", help="最终提交的 submission.csv 路径")
    parser.add_argument(
        "pictures_100",
        help="100 个计时 study 的图片目录（结构 <subject_id>/<study_id>/<pic_id>.jpg）",
    )
    parser.add_argument("pictures_5", help="5 个预热 study 的图片目录（结构同上）")
    parser.add_argument(
        "--checkpoint",
        nargs="+",
        help="模型权重，可传多个；缺省读 cost_ensemble.json 或按单模型顺序查找",
    )
    parser.add_argument(
        "--ensemble",
        help=f"多模型融合清单 JSON；缺省在 submission.csv 同目录与仓库根目录找 {ENSEMBLE_MANIFEST}",
    )
    parser.add_argument(
        "--weights",
        help="逗号分隔的模型权重，缺省等权；数量必须与 --checkpoint 一致",
    )
    parser.add_argument(
        "--fusion",
        choices=FUSION_METHODS,
        default=None,
        help="融合方式，缺省取清单里的 fusion，再缺省 probability",
    )
    parser.add_argument("--config", help="配置文件路径；仅建议单模型调试时使用")
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
    parser.add_argument(
        "--views-csv",
        help=f"视角元数据 CSV（Study_id, Dicom_id, ViewPosition）；缺省找 {VIEWS_FILENAME} 或 test.csv",
    )
    parser.add_argument(
        "--prediction-out",
        help="把逐图预测写到指定 CSV；缺省写临时文件，计时结束后清理",
    )
    parser.add_argument(
        "--preprocess-workers",
        type=int,
        default=6,
        help="study 内并行解码/缩放的最大线程数；1 表示串行，默认 6",
    )
    parser.add_argument(
        "--init-limit-ms",
        type=float,
        default=DEFAULT_INIT_LIMIT_MS,
        help=f"初始化耗时自设上限，超出只警告，默认 {DEFAULT_INIT_LIMIT_MS:g} ms",
    )
    parser.add_argument(
        "--max-model-mb",
        type=float,
        default=DEFAULT_MAX_MODEL_MB,
        help=f"权重合计大小上限，超出只警告，默认 {DEFAULT_MAX_MODEL_MB:g} MB",
    )
    parser.add_argument("--report", help="把计时与比对明细写成 JSON 报告")
    return parser.parse_args(argv)


def parse_weights(text: str | None, count: int) -> list[float] | None:
    """把 `--weights 1,1,1,1` 解析成浮点列表并校验数量。"""
    if text is None:
        return None
    pieces = [piece.strip() for piece in text.split(",") if piece.strip()]
    if not pieces:
        raise ValueError("--weights 不能为空")
    try:
        values = [float(piece) for piece in pieces]
    except ValueError as error:
        raise ValueError("--weights 必须是逗号分隔的数字") from error
    if len(values) != count:
        raise ValueError(f"--weights 数量必须与模型数量一致：{len(values)} 与 {count}")
    return values


def resolve_checkpoint(explicit: str | None, submission_path: Path) -> Path:
    """按「显式参数 → 环境变量 → 提交包 → outputs」的顺序找单个模型权重。"""
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"--checkpoint 指向的文件不存在：{path}")
        return path

    candidates: list[Path] = []
    env_checkpoint = os.environ.get("COST_CHECKPOINT")
    if env_checkpoint:
        candidates.append(Path(env_checkpoint).expanduser())
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


def _find_manifest(explicit: str | None, submission_path: Path) -> Path | None:
    """找多模型融合清单：显式参数 → submission.csv 同目录 → 仓库根目录。"""
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"--ensemble 指向的文件不存在：{path}")
        return path
    for base in (submission_path.parent, ROOT):
        candidate = base / ENSEMBLE_MANIFEST
        if candidate.is_file():
            return candidate
    return None


def _read_manifest(path: Path) -> tuple[list[Path], list[float] | None, str | None]:
    """读融合清单；checkpoint 相对路径按清单所在目录解析。"""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} 不是合法 JSON：{error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{path} 顶层必须是对象")

    raw_checkpoints = payload.get("checkpoints")
    if not isinstance(raw_checkpoints, list) or not raw_checkpoints:
        raise ValueError(f"{path} 缺少非空的 checkpoints 列表")
    checkpoints = []
    for item in raw_checkpoints:
        candidate = Path(str(item)).expanduser()
        if not candidate.is_absolute():
            candidate = path.parent / candidate
        checkpoints.append(candidate)

    raw_weights = payload.get("weights")
    weights: list[float] | None = None
    if raw_weights is not None:
        if not isinstance(raw_weights, list) or not raw_weights:
            raise ValueError(f"{path} 的 weights 必须是非空数组")
        try:
            weights = [float(value) for value in raw_weights]
        except (TypeError, ValueError) as error:
            raise ValueError(f"{path} 的 weights 必须全是数字") from error
        if len(weights) != len(checkpoints):
            raise ValueError(
                f"{path} 的 weights 数量必须与 checkpoints 一致："
                f"{len(weights)} 与 {len(checkpoints)}"
            )

    fusion = payload.get("fusion")
    if fusion is not None and fusion not in FUSION_METHODS:
        raise ValueError(f"{path} 的 fusion 只能是 {FUSION_METHODS}，当前为 {fusion!r}")
    return checkpoints, weights, fusion


def resolve_ensemble(
    args: argparse.Namespace, submission_path: Path
) -> tuple[list[Path], list[float] | None, str, Path | None]:
    """确定本次计时实际要跑的模型、权重与融合方式。

    优先级：显式 `--checkpoint` > 清单里的 checkpoints > 单模型自动查找；
    权重与融合方式同理，命令行显式值优先于清单。显式 checkpoint 数量与清单
    权重数量不一致时，清单权重作废、回退等权，避免静默错配。
    """
    manifest_path = _find_manifest(args.ensemble, submission_path)
    manifest_checkpoints: list[Path] = []
    manifest_weights: list[float] | None = None
    manifest_fusion: str | None = None
    if manifest_path is not None:
        manifest_checkpoints, manifest_weights, manifest_fusion = _read_manifest(
            manifest_path
        )

    if args.checkpoint:
        checkpoints = [Path(item).expanduser() for item in args.checkpoint]
    elif manifest_checkpoints:
        checkpoints = manifest_checkpoints
    else:
        checkpoints = [resolve_checkpoint(None, submission_path)]

    for path in checkpoints:
        if not path.is_file():
            raise FileNotFoundError(f"模型权重不存在：{path}")

    weights = parse_weights(args.weights, len(checkpoints))
    if weights is None and manifest_weights and len(manifest_weights) == len(checkpoints):
        weights = manifest_weights
    elif args.checkpoint and manifest_weights and len(manifest_weights) != len(checkpoints):
        print(
            "[WARN] 融合清单的 weights 数量与 --checkpoint 不一致，"
            "本次回退等权；请用 --weights 显式指定"
        )

    fusion = args.fusion or manifest_fusion or "probability"
    if fusion not in FUSION_METHODS:
        raise ValueError(f"fusion 只能是 {FUSION_METHODS}，当前为 {fusion!r}")
    return checkpoints, weights, fusion, manifest_path


def resolve_config_for(
    checkpoint_path: Path, explicit: str | None, submission_path: Path
) -> Config:
    """取一个模型对应的配置。

    顺序：显式 `--config` → checkpoint 存档配置 → 仓库默认配置 → 提交包配置。
    存档配置优先于仓库默认配置：默认配置的骨干/输入尺寸未必与权重一致，用错
    会导致 `load_checkpoint_model` 直接报「推理配置与 checkpoint 不一致」。
    """
    if explicit:
        return load_config(explicit)

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    raw_config = checkpoint.get("config")
    if raw_config:
        return config_from_dict(raw_config)

    for base in (ROOT, submission_path.parent):
        for relative in CONFIG_CANDIDATES:
            candidate = base / relative
            if candidate.is_file():
                return load_config(candidate)
    raise FileNotFoundError(
        f"{checkpoint_path} 没有存档配置，且找不到 configs/default.yaml / config.yaml："
        "请用 --config 指定"
    )


def resolve_views(
    args: argparse.Namespace, submission_path: Path, specs: Sequence[ModelSpec]
) -> tuple[dict[str, dict[str, int]] | None, str | None]:
    """需要视角加权时读取视角元数据；不需要或读不到时返回 None。

    候选顺序：`--views-csv` → submission.csv 同目录的 `cost_views.csv` →
    各模型配置里的 `data.test.csv`。这些都只是 CSV 元数据，不碰图像像素。
    """
    if not any(spec.config.model.uses_views for spec in specs):
        return None, None

    candidates: list[Path] = []
    if args.views_csv:
        explicit = Path(args.views_csv).expanduser()
        if not explicit.is_file():
            raise FileNotFoundError(f"--views-csv 指向的文件不存在：{explicit}")
        candidates.append(explicit)
    candidates.append(submission_path.parent / VIEWS_FILENAME)
    for spec in specs:
        candidates.append(Path(spec.config.data.test.csv))

    seen: set[Path] = set()
    for path in candidates:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        try:
            return read_view_map(path), str(path)
        except (ValueError, OSError) as error:
            print(f"[WARN] 视角元数据不可用（{path}）：{error}")
    print(
        "[WARN] 当前模型使用视角加权，但找不到视角元数据；本次回退为等权均值，"
        f"预测可能与 submission.csv 不一致。请提供 {VIEWS_FILENAME} 或用 --views-csv 指定"
    )
    return None, None


def _synchronize(device: torch.device) -> None:
    """把异步加速设备上的工作排空，保证计时区间包住全部计算。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _transform_key(config: Config) -> tuple:
    """预处理的完整签名；签名相同的模型共享同一次 transform。"""
    image = config.image
    return (
        image.size,
        image.letterbox,
        image.normalize,
        image.in_channels,
        image.xrv_maxval,
    )


def _load_models(
    specs: Sequence[ModelSpec], device: torch.device, precision: str
) -> list[LoadedModel]:
    """按顺序加载全部模型；每个模型用自己 checkpoint 里的配置建预处理。"""
    loaded: list[LoadedModel] = []
    for spec in specs:
        model = load_checkpoint_model(spec.config, spec.checkpoint, device)
        if precision == "fp16":
            model = model.half()
        loaded.append(
            LoadedModel(
                spec=spec,
                model=model.eval(),
                transform=build_eval_inference_transform(spec.config),
                transform_key=_transform_key(spec.config),
            )
        )
    return loaded


def _fuse_study(
    loaded: Sequence[LoadedModel],
    image_paths: Sequence[Path],
    views: Sequence[int] | None,
    device: torch.device,
    weights: Sequence[float] | None,
    fusion: str,
    preprocess_workers: int,
    executor: ThreadPoolExecutor | None = None,
) -> np.ndarray:
    """一个 study 共享预处理、逐模型前向，再按 submission 的融合口径合成。

    `predict_study_multi` 会并行解码/缩放，并让 `transform_key` 相同的模型复用
    同一份预处理张量；数值与逐模型独立预处理完全一致。
    """
    specs = [
        StudyModelSpec(
            model=item.model,
            transform=item.transform,
            transform_key=item.transform_key,
            uses_views=item.spec.config.model.uses_views,
            frontal_weight=item.spec.config.model.frontal_weight,
            probability_space=item.spec.config.model.probability_space,
        )
        for item in loaded
    ]
    probabilities = predict_study_multi(
        specs, image_paths, device, views, max_workers=preprocess_workers, executor=executor
    )
    return fuse_probability_vectors(probabilities, weights, fusion)


def _time_one_study(
    loaded: Sequence[LoadedModel],
    job: StudyJob,
    views: Sequence[int] | None,
    device: torch.device,
    weights: Sequence[float] | None,
    fusion: str,
    preprocess_workers: int,
    writer: Any,
    handle: Any,
    executor: ThreadPoolExecutor | None = None,
) -> tuple[float, np.ndarray]:
    """测一个 study 的端到端毫秒数，并把该 study 每张图的一行写出去。

    计时区间：解码（每张图只读一次）→ 同分辨率共享预处理 → 每个模型前向 →
    study 聚合 → 融合 → 格式化 → 写行 → flush。CUDA/MPS 在首尾各同步一次，
    避免把异步 kernel 的耗时漏到区间外。
    """
    _synchronize(device)
    started = time.perf_counter()
    probabilities = _fuse_study(
        loaded, job.images, views, device, weights, fusion, preprocess_workers, executor
    )
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
    ):
        raise ValueError(f"study {job.study_id} 的概率必须为 [0, 1] 内的有限数")
    formatted = [f"{float(value):.6f}" for value in probabilities]
    for path in job.images:
        writer.writerow((job.subject_id, job.study_id, path.stem, *formatted))
    handle.flush()
    _synchronize(device)
    return (time.perf_counter() - started) * 1000.0, probabilities


def _warmup(
    loaded: Sequence[LoadedModel],
    warmup_jobs: Sequence[StudyJob],
    views_by_study: dict[str, tuple[int, ...]],
    device: torch.device,
    weights: Sequence[float] | None,
    fusion: str,
    preprocess_workers: int,
    executor: ThreadPoolExecutor | None = None,
) -> float:
    """预热：每个预热 study 走与计时完全相同的前向/融合路径，不写文件。"""
    started = time.perf_counter()
    for job in warmup_jobs:
        _fuse_study(
            loaded,
            job.images,
            views_by_study.get(job.study_id),
            device,
            weights,
            fusion,
            preprocess_workers,
            executor,
        )
    _synchronize(device)
    return (time.perf_counter() - started) * 1000.0


def _warmup_shapes(
    loaded: Sequence[LoadedModel],
    jobs: Sequence[StudyJob],
    device: torch.device,
) -> float:
    """用合成零张量预热计时集中出现的图片数形状（答疑允许，不计入成绩）。"""
    started = time.perf_counter()
    counts = [job.image_count for job in jobs]
    for item in loaded:
        config = item.spec.config
        warmup_study_shapes(
            item.model,
            counts,
            config.image.size,
            config.image.in_channels,
            device,
            config.model.frontal_weight,
            config.model.probability_space,
        )
    _synchronize(device)
    return (time.perf_counter() - started) * 1000.0


def _format_timing(timing) -> dict:
    return {
        "rounds": timing.rounds,
        "studies": timing.studies,
        "images": timing.images,
        "mean": timing.mean_ms,
        "billed_mean": timing.billed_mean_ms,
        "per_image_mean": timing.per_image_mean_ms,
        "round_overhead_per_study": timing.round_overhead_per_study_ms,
        "median": timing.median_ms,
        "p95": timing.p95_ms,
        "min": timing.min_ms,
        "max": timing.max_ms,
        "total": timing.total_ms,
        "round_total": list(timing.round_total_ms),
        "limit": timing.limit_ms,
        "within_limit": timing.within_limit,
    }


def _run(args: argparse.Namespace) -> int:
    """真正的主流程；异常由 `main` 统一转成退出码 2。"""
    submission_path = Path(args.submission).resolve()

    # ---- 初始化（不计时）：只读元数据，不碰图像像素，一次完成 ----
    init_started = time.perf_counter()
    submission = read_submission(submission_path)
    images = list_images(args.pictures_100)
    warmup_images = list_images(args.pictures_5)
    jobs = build_study_jobs(images)
    warmup_jobs = build_study_jobs(warmup_images)
    overlap = {job.study_id for job in jobs} & {job.study_id for job in warmup_jobs}
    if overlap:
        preview = "、".join(sorted(overlap)[:5])
        raise ValueError(
            f"预热 study 与计时 study 重叠（{preview}）；赛题要求两者不相交"
        )
    init_ms = (time.perf_counter() - init_started) * 1000.0
    init_within_limit = init_ms <= args.init_limit_ms
    print(
        f"[INFO] 初始化（仅元数据，不计时）：{init_ms:.3f} ms；"
        f"自设上限 {args.init_limit_ms:.0f} ms"
    )
    if not init_within_limit:
        print(
            f"[WARN] 初始化耗时 {init_ms:.3f} ms 超过自设上限 {args.init_limit_ms:.0f} ms，"
            "请确认没有把解码/前向/预测挪进初始化"
        )

    if args.rounds <= 0:
        raise ValueError(f"--rounds 必须为正整数，当前为 {args.rounds}")
    if args.preprocess_workers <= 0:
        raise ValueError(
            f"--preprocess-workers 必须为正整数，当前为 {args.preprocess_workers}"
        )
    if not (math.isfinite(args.init_limit_ms) and args.init_limit_ms > 0):
        raise ValueError(f"--init-limit-ms 必须为正数，当前为 {args.init_limit_ms!r}")

    # ---- 模型加载（不计时）：配置解析 + 建模型 + 载权重 ----
    model_started = time.perf_counter()
    checkpoints, weights, fusion, manifest_path = resolve_ensemble(args, submission_path)
    specs = [
        ModelSpec(
            checkpoint=path,
            config=resolve_config_for(path, args.config, submission_path),
        )
        for path in checkpoints
    ]
    device = resolve_device(args.device or specs[0].config.device)
    threshold = specs[0].config.eval.threshold if args.threshold is None else args.threshold
    if not (math.isfinite(threshold) and 0.0 <= threshold <= 1.0):
        raise ValueError(f"--threshold 必须落在 [0, 1]，当前为 {threshold!r}")

    views_started = time.perf_counter()
    view_map, view_source = resolve_views(args, submission_path, specs)
    views_load_ms = (time.perf_counter() - views_started) * 1000.0
    setup_ms = (views_started - model_started) * 1000.0
    views_by_study: dict[str, tuple[int, ...]] = {}
    missing_views: list[str] = []
    if view_map is not None:
        timed_studies = {job.study_id for job in jobs}
        for job in (*jobs, *warmup_jobs):
            values = views_for_job(job, view_map)
            if values is None:
                if job.study_id in timed_studies:
                    missing_views.append(job.study_id)
            else:
                views_by_study[job.study_id] = values
        if missing_views:
            print(
                f"[WARN] {len(missing_views)} 个计时 study 在视角元数据里缺记录"
                f"（例如 {missing_views[0]}），这些 study 回退为等权均值"
            )

    load_started = time.perf_counter()
    loaded = _load_models(specs, device, args.precision)
    model_load_ms = (time.perf_counter() - load_started) * 1000.0

    checkpoint_bytes = sum(
        path.stat().st_size for path in {item.resolve() for item in checkpoints}
    )
    checkpoint_mb = checkpoint_bytes / (1024.0 * 1024.0)
    normalized_weights = normalize_weights(weights, len(loaded)).tolist()

    print(f"[INFO] submission：{submission_path}（{len(submission.study_ids)} 个 study）")
    for index, item in enumerate(loaded, start=1):
        print(
            f"[INFO] 模型 {index}/{len(loaded)}：{item.spec.checkpoint}"
            f"（config 来自 {'命令行' if args.config else 'checkpoint 存档'}）"
        )
    if manifest_path is not None:
        print(f"[INFO] 融合清单：{manifest_path}")
    print(
        f"[INFO] 设备：{device}；推理精度：{args.precision}；阈值：{threshold:g}；"
        f"预处理线程：{args.preprocess_workers}；"
        f"融合：{fusion}，权重 {normalized_weights}"
    )
    print(
        f"[INFO] 权重合计：{checkpoint_mb:.1f} MB"
        f"（赛题上限 {args.max_model_mb:.0f} MB）"
    )
    if checkpoint_mb > args.max_model_mb:
        print(
            f"[WARN] 权重合计 {checkpoint_mb:.1f} MB 超过赛题第 8 条上限 "
            f"{args.max_model_mb:.0f} MB"
        )
    print(
        f"[INFO] 配置解析 {setup_ms:.3f} ms；视角元数据 {views_load_ms:.3f} ms；"
        f"模型加载 {model_load_ms:.3f} ms（均不计入成绩）"
    )
    print(
        f"[INFO] 计时目录：{args.pictures_100}（{len(jobs)} 个 study / {len(images)} 张图）；"
        f"预热目录：{args.pictures_5}（{len(warmup_jobs)} 个 study / {len(warmup_images)} 张图）"
    )
    if len(jobs) != 100:
        print(
            f"[WARN] 正赛要求 100 个计时 study，当前为 {len(jobs)} 个"
            "（官方示例数据或本地调试可用，正式成绩以 100 个为准）"
        )
    if len(warmup_jobs) != DEFAULT_WARMUP_STUDIES:
        print(
            f"[WARN] 规则要求 {DEFAULT_WARMUP_STUDIES} 个预热 study，"
            f"当前为 {len(warmup_jobs)} 个"
        )
    if args.precision == "fp16" and device.type == "cpu":
        print("[WARN] CPU 上 fp16 通常比 fp32 慢，且部分算子可能不支持；正式测速请在 GPU 上跑")
    if view_source is not None:
        print(f"[INFO] 视角元数据：{view_source}（CSV，不计入成绩）")

    # ---- 预热与正式计时共用同一个预处理线程池：一次运行只建一次 ----
    # 计划 O1：线程池生命周期从「每个 study 两次」扩大到「整次运行一次」，
    # 只改 CPU 调度，不跨 study 预读或预处理，不改变任何数值。
    preprocess_executor = (
        ThreadPoolExecutor(max_workers=args.preprocess_workers)
        if args.preprocess_workers > 1
        else None
    )
    round_times_ms: list[list[float]] = []
    round_total_ms: list[float] = []
    study_probabilities: dict[str, np.ndarray] = {}
    last_prediction_path = ""
    try:
        # ---- 预热：规则明确不计入成绩，但与正式计时走同一条路径 ----
        warmup_ms = _warmup(
            loaded,
            warmup_jobs,
            views_by_study,
            device,
            weights,
            fusion,
            args.preprocess_workers,
            preprocess_executor,
        )
        print(
            f"[INFO] 预热完成：{len(warmup_jobs)} 个 study / {len(warmup_images)} 张图"
            f"（{warmup_ms:.3f} ms，不计入成绩）"
        )
        shape_warmup_ms = _warmup_shapes(loaded, jobs, device)
        print(
            f"[INFO] 形状预热：图片数 {sorted({job.image_count for job in jobs})}"
            f"（{shape_warmup_ms:.3f} ms，不计入成绩）"
        )

        # ---- 正式计时：外层 3 轮、内层逐 study；每轮只开一次预测文件 ----
        with tempfile.TemporaryDirectory(prefix="chestx-cost-") as directory:
            if args.prediction_out:
                prediction_path = Path(args.prediction_out)
                prediction_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                prediction_path = Path(directory) / "cost_predictions.csv"
            last_prediction_path = str(prediction_path)
            print(f"[INFO] 预测写出：{prediction_path}（每轮重写，逐 study flush）")

            for round_index in range(args.rounds):
                round_started = time.perf_counter()
                handle = prediction_path.open("w", encoding="utf-8", newline="")
                try:
                    writer = csv.writer(handle, lineterminator="\n")
                    writer.writerow(COST_OUTPUT_COLUMNS)
                    times: list[float] = []
                    round_probabilities: dict[str, np.ndarray] = {}
                    for job in jobs:
                        elapsed, probabilities = _time_one_study(
                            loaded,
                            job,
                            views_by_study.get(job.study_id),
                            device,
                            weights,
                            fusion,
                            args.preprocess_workers,
                            writer,
                            handle,
                            preprocess_executor,
                        )
                        times.append(elapsed)
                        round_probabilities[job.study_id] = probabilities
                    handle.flush()
                finally:
                    handle.close()
                round_total = (time.perf_counter() - round_started) * 1000.0
                round_times_ms.append(times)
                round_total_ms.append(round_total)
                study_probabilities = round_probabilities
                print(
                    f"[INFO] 第 {round_index + 1}/{args.rounds} 轮完成："
                    f"{len(times)} 个 study，单 study 均值 {sum(times) / len(times):.3f} ms，"
                    f"轮总墙钟 {round_total:.3f} ms"
                )
    finally:
        if preprocess_executor is not None:
            preprocess_executor.shutdown()

    timing = summarize_timings(
        round_times_ms,
        images_per_study=[job.image_count for job in jobs],
        round_total_ms=round_total_ms,
        limit_ms=COST_LIMIT_MS,
    )
    print(
        f"[INFO] 单 study 耗时：均值 {timing.mean_ms:.3f} ms，"
        f"轮级 IO 分摊 {timing.round_overhead_per_study_ms:.3f} ms，"
        f"计费均值 {timing.billed_mean_ms:.3f} ms；中位数 {timing.median_ms:.3f} ms，"
        f"p95 {timing.p95_ms:.3f} ms，最小 {timing.min_ms:.3f} ms，"
        f"最大 {timing.max_ms:.3f} ms；上限 {timing.limit_ms:.0f} ms"
    )

    # ---- 一致性校验：用最后一轮计时现场算出的概率，与 submission.csv 比对 ----
    comparison = compare_study_predictions(
        study_probabilities,
        submission,
        threshold=threshold,
        tolerance=args.tolerance,
        image_counts={job.study_id: job.image_count for job in jobs},
        study_subjects={job.study_id: job.subject_id for job in jobs},
    )
    print(
        f"[INFO] 一致性：比对 {comparison.checked_studies} 个 study / "
        f"{comparison.checked_images} 张图；最大概率差 {comparison.max_abs_diff:.6g}；"
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
            warmup_studies=len(warmup_jobs),
            warmup_images=len(warmup_images),
            init_ms=init_ms,
            model_count=len(loaded),
            fusion=fusion,
        )
    )

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "submission": str(submission_path),
            "checkpoints": [
                {
                    "path": str(item.spec.checkpoint),
                    "bytes": item.spec.checkpoint.stat().st_size,
                    "config": item.spec.config.to_dict(),
                }
                for item in loaded
            ],
            "model_count": len(loaded),
            "fusion": fusion,
            "weights": normalized_weights,
            "ensemble_manifest": str(manifest_path) if manifest_path else None,
            "checkpoint_mb": checkpoint_mb,
            "device": str(device),
            "precision": args.precision,
            "preprocess_workers": args.preprocess_workers,
            "threshold": threshold,
            "views": {
                "source": view_source,
                "required": any(item.spec.config.model.uses_views for item in loaded),
                "available": view_map is not None,
                "load_ms": views_load_ms,
            },
            "units": {
                "studies": len(jobs),
                "images": len(images),
                "warmup_studies": len(warmup_jobs),
                "warmup_images": len(warmup_images),
            },
            "init_ms": init_ms,
            "init_limit_ms": args.init_limit_ms,
            "init_within_limit": init_within_limit,
            "setup_ms": setup_ms,
            "model_load_ms": model_load_ms,
            "warmup_ms": warmup_ms,
            "shape_warmup_ms": shape_warmup_ms,
            "prediction_csv": last_prediction_path,
            "prediction_csv_persisted": bool(args.prediction_out),
            "timing_ms": _format_timing(timing),
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
    if not timing.within_limit:
        print(
            f"[ERROR] 单 study 平均耗时 {timing.billed_mean_ms:.3f} ms 超过上限 "
            f"{timing.limit_ms:.0f} ms",
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
