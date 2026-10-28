"""推理耗时评测（赛题第 7 条）的可测逻辑：分组、submission 解析与比对、计时统计。

`src/cost.py` 只做命令行编排：解析参数、加载模型、跑循环、打印结论。
所有「口径」都放在这里，用合成数据就能单测，不需要真的跑 100 张图：

- `list_images` / `parse_image_identity` / `group_images_by_study`：
  把 `<subject_id>/<study_id>/<pic_id>.jpg` 目录结构还原成 study 分组；
- `read_submission`：读并校验最终提交的 submission.csv（列名、类别、概率范围、
  重复行），得到 `study_id -> {类别: 概率}`；
- `select_prediction_rows`：按阈值做后处理，口径与 `infer.submission.
  build_submission_frame` 保持一致（`>=` 阈值；一个都没过线就保留 top1）；
- `compare_study_predictions`：把脚本现场算出的 study 概率和 submission.csv
  逐 study 比对，给出差异明细；
- `summarize_timings` / `format_conclusion`：3 轮 → 每张均值 → 100 张均值，
  以及贴合规则示例的 `[CONCLUSION]` 结论行。
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from common.constants import CLASS_IDS, CSV_ENCODING, NUM_CLASSES
from data.index import index_images
from infer.submission import SUBMISSION_COLUMNS

# 规则第 7 条：单张图片平均耗时上限（毫秒）。
COST_LIMIT_MS = 100.0
# 规则第 7 条：全部样本连续重复 3 轮；预热用 5 张图。
DEFAULT_ROUNDS = 3
DEFAULT_WARMUP_IMAGES = 5
# 概率比对默认容差：fp16 与 fp32 之间通常差 1e-3 量级，留一点余量。
DEFAULT_TOLERANCE = 1e-3


def list_images(root: str | Path) -> list[Path]:
    """递归列出 root 下所有图片，按路径排序返回（顺序固定，便于复现）。

    复用 `data.index.index_images` 的索引逻辑：只认 IMAGE_SUFFIXES 里的后缀，
    且同一文件名（不含后缀）出现两次会报错——测试耗时目录同样不该有重名图。
    """
    root = Path(root)
    images = sorted(index_images(root).values())
    if not images:
        raise ValueError(f"图片目录里没有图片：{root}")
    return images


@dataclass(frozen=True)
class ImageIdentity:
    """测试目录里一张图属于哪个患者、哪次检查。"""

    subject_id: str
    study_id: str
    path: Path


def parse_image_identity(path: str | Path) -> ImageIdentity:
    """从 `<subject_id>/<study_id>/<pic_id>.jpg` 结构里取出 patient 与 study。

    规则说 pictures_100 的「图片及文件结构与测试集一致」，所以目录层级就是
    Subject_id / Study_id。层级不足时直接报错，免得拿错目录跑出一份假耗时。
    """
    path = Path(path)
    study_directory = path.parent
    subject_directory = study_directory.parent
    # 相对文件名（parent 是 "."、name 为空）或直接放在根目录下的图，
    # 都没有 <subject_id>/<study_id>/ 这两级，拿不到 study 归属。
    if not study_directory.name or not subject_directory.name:
        raise ValueError(
            f"图片路径 {path} 层级不足，期望 <subject_id>/<study_id>/<pic_id>.<后缀>"
        )
    subject_id = subject_directory.name
    study_id = study_directory.name
    return ImageIdentity(subject_id=subject_id, study_id=study_id, path=path)


def group_images_by_study(
    paths: Sequence[str | Path],
) -> dict[str, list[Path]]:
    """把同一 study 的图片归到一起，保持 study 首次出现的顺序。

    同一次检查的图必须落在同一个 Subject_id 目录下；发现同一 study 对应两个
    patient 目录就报错——测试目录摆错时，早点失败比算出一个错数字好。
    """
    groups: dict[str, list[Path]] = {}
    subjects: dict[str, str] = {}
    for path in paths:
        identity = parse_image_identity(path)
        previous = subjects.get(identity.study_id)
        if previous is not None and previous != identity.subject_id:
            raise ValueError(
                f"同一 study {identity.study_id} 落在多个 Subject_id 目录下："
                f"{previous} 与 {identity.subject_id}"
            )
        subjects[identity.study_id] = identity.subject_id
        groups.setdefault(identity.study_id, []).append(identity.path)
    return groups


@dataclass(frozen=True)
class SubmissionTable:
    """submission.csv 解析结果：study -> {类别 ID: 概率}。"""

    probabilities: Mapping[str, dict[int, float]]
    subjects: Mapping[str, str] = field(default_factory=dict)

    @property
    def study_ids(self) -> tuple[str, ...]:
        """文件里出现的 study，按首次出现顺序。"""
        return tuple(self.probabilities)

    @property
    def filtered(self) -> bool:
        """是否有 study 少于 10 个类别（即提交时按阈值过滤过）。"""
        return any(len(entry) < NUM_CLASSES for entry in self.probabilities.values())


def read_submission(path: str | Path) -> SubmissionTable:
    """读并校验 submission.csv，返回 study -> {类别: 概率}。

    校验口径与 `infer.submission.validate_submission` 一致，但只依赖文件本身
    （cost.py 在赛题服务器上不一定有 test.csv）：列名与顺序、study/类别不重复、
    Predict_class 在 1~10、Probability 是 [0, 1] 内的有限数、Subject_id/Study_id
    非空。返回的概率不做四舍五入，比对时按文件里写的值算差。
    """
    path = Path(path)
    with path.open("r", encoding=CSV_ENCODING, newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        if list(header) != list(SUBMISSION_COLUMNS):
            raise ValueError(
                f"{path} 的列名或顺序不正确：期望 {list(SUBMISSION_COLUMNS)}，"
                f"实际 {list(header)}"
            )

        probabilities: dict[str, dict[int, float]] = {}
        subjects: dict[str, str] = {}
        for line_number, row in enumerate(reader, start=2):
            subject_id = (row.get("Subject_id") or "").strip()
            study_id = (row.get("Study_id") or "").strip()
            if not subject_id or not study_id:
                raise ValueError(f"{path}:{line_number} 的 Subject_id/Study_id 为空")
            if study_id in subjects and subjects[study_id] != subject_id:
                raise ValueError(f"{path}:{line_number} 的 study {study_id} 对应多个 Subject_id")
            subjects[study_id] = subject_id

            try:
                class_id = int((row.get("Predict_class") or "").strip())
            except ValueError as error:
                raise ValueError(
                    f"{path}:{line_number} 的 Predict_class 不是整数："
                    f"{row.get('Predict_class')!r}"
                ) from error
            if class_id not in CLASS_IDS:
                raise ValueError(
                    f"{path}:{line_number} 的 Predict_class 必须在 1~{NUM_CLASSES} 内，"
                    f"实际为 {class_id}"
                )

            try:
                probability = float((row.get("Probability") or "").strip())
            except ValueError as error:
                raise ValueError(
                    f"{path}:{line_number} 的 Probability 不是数字："
                    f"{row.get('Probability')!r}"
                ) from error
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError(
                    f"{path}:{line_number} 的 Probability 必须在 [0, 1] 内且有限，"
                    f"实际为 {probability!r}"
                )

            entry = probabilities.setdefault(study_id, {})
            if class_id in entry:
                raise ValueError(
                    f"{path}:{line_number} 重复提交了 study {study_id} 的类别 {class_id}"
                )
            entry[class_id] = probability

    if not probabilities:
        raise ValueError(f"{path} 里没有任何预测行")
    return SubmissionTable(probabilities=probabilities, subjects=subjects)


def select_prediction_rows(
    probabilities: Sequence[float] | np.ndarray, threshold: float
) -> list[tuple[int, float]]:
    """按阈值挑出要写进 submission 的 (类别 ID, 概率) 行。

    与 `infer.submission.build_submission_frame` 同口径：概率 `>= threshold`
    才保留；一个都没过线时保留概率最高的那一类，保证每个 study 至少一行。
    计时循环把这次后处理也算进端到端耗时（规则要求含后处理）。
    """
    values = np.asarray(probabilities, dtype=float)
    if values.shape != (NUM_CLASSES,):
        raise ValueError(f"概率向量长度必须是 {NUM_CLASSES}，实际 shape 为 {values.shape}")
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("概率必须为 [0, 1] 内的有限数")
    if not (math.isfinite(threshold) and 0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold 必须落在 [0, 1]，当前为 {threshold!r}")

    selected = [
        (class_id, float(values[class_id - 1]))
        for class_id in CLASS_IDS
        if values[class_id - 1] >= threshold
    ]
    if not selected:
        best_class = max(CLASS_IDS, key=lambda class_id: values[class_id - 1])
        selected = [(best_class, float(values[best_class - 1]))]
    return selected


@dataclass(frozen=True)
class ComparisonIssue:
    """一个和 submission.csv 对不上的 study。"""

    study_id: str
    reason: str
    max_abs_diff: float | None = None
    image_count: int = 0


@dataclass(frozen=True)
class ComparisonReport:
    """现场推理 vs submission.csv 的比对结果。"""

    checked_studies: int
    checked_images: int
    inconsistent: tuple[ComparisonIssue, ...]
    max_abs_diff: float
    filtered_submission: bool
    threshold: float
    tolerance: float

    @property
    def consistent(self) -> bool:
        """所有 study 都在容差内一致。"""
        return not self.inconsistent

    @property
    def inconsistent_studies(self) -> int:
        """不一致的 study 数。"""
        return len(self.inconsistent)

    @property
    def inconsistent_images(self) -> int:
        """不一致 study 覆盖的图片数（由 compare_study_predictions 估算）。"""
        return sum(issue.image_count for issue in self.inconsistent)


def compare_study_predictions(
    study_probabilities: Mapping[str, np.ndarray],
    submission: SubmissionTable,
    *,
    threshold: float,
    tolerance: float = DEFAULT_TOLERANCE,
    image_counts: Mapping[str, int] | None = None,
    study_subjects: Mapping[str, str] | None = None,
) -> ComparisonReport:
    """把现场算出的 study 概率与 submission.csv 比对。

    `study_probabilities` 是 `study_id -> [10] 概率`，由 cost.py 对 pictures_100
    按 study 分组、调用与提交同一套推理函数得到。比对分两种情况：

    - 提交文件是「全 10 类」：逐类比较概率，最大绝对差 <= tolerance 即一致；
    - 提交文件是「按阈值过滤过」：用同一个 threshold 对现场概率做后处理，先比
      类别集合是否完全相同，再比交集内概率（`select_prediction_rows` 与
      `build_submission_frame` 同口径，含「都没过线时保留 top1」的回退）。

    submission.csv 里有、但 pictures_100 没覆盖到的 study 不算不一致——计时目录
    本来就是测试集的子集。反过来，现场有而提交缺的 study 一定记为不一致。
    """
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError(f"tolerance 必须是 [0, +∞) 内的有限数，当前为 {tolerance!r}")

    counts = dict(image_counts or {})
    issues: list[ComparisonIssue] = []
    max_abs_diff = 0.0

    for study_id, raw in study_probabilities.items():
        probabilities = np.asarray(raw, dtype=float)
        if probabilities.shape != (NUM_CLASSES,):
            raise ValueError(
                f"study {study_id} 的概率向量长度必须是 {NUM_CLASSES}，"
                f"实际 shape 为 {probabilities.shape}"
            )
        if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
            raise ValueError(f"study {study_id} 的概率必须为 [0, 1] 内的有限数")
        image_count = int(counts.get(study_id, 0))
        entry = submission.probabilities.get(study_id)

        if entry is None:
            issues.append(
                ComparisonIssue(
                    study_id,
                    "submission.csv 里缺少该 study",
                    image_count=image_count,
                )
            )
            continue

        if study_subjects is not None and submission.subjects.get(study_id) != study_subjects.get(study_id):
            issues.append(ComparisonIssue(study_id, "Subject_id 与测试集不一致", image_count=image_count))
            continue

        if len(entry) == NUM_CLASSES:
            differences = {
                class_id: abs(float(probabilities[class_id - 1]) - entry[class_id])
                for class_id in CLASS_IDS
            }
            worst = max(differences.values())
            max_abs_diff = max(max_abs_diff, worst)
            if worst > tolerance:
                issues.append(
                    ComparisonIssue(
                        study_id,
                        f"10 类概率最大绝对差 {worst:.6g} 超过容差 {tolerance:g}",
                        worst,
                        image_count,
                    )
                )
            continue

        # 过滤过的提交：先对齐类别集合，再比集合内概率。
        selected = select_prediction_rows(probabilities, threshold)
        selected_classes = {class_id for class_id, _ in selected}
        if selected_classes != set(entry):
            issues.append(
                ComparisonIssue(
                    study_id,
                    f"类别集合不一致：脚本算出 {sorted(selected_classes)}，"
                    f"提交文件为 {sorted(entry)}",
                    image_count=image_count,
                )
            )
            continue

        differences = {
            class_id: abs(probability - entry[class_id])
            for class_id, probability in selected
        }
        worst = max(differences.values())
        max_abs_diff = max(max_abs_diff, worst)
        if worst > tolerance:
            issues.append(
                ComparisonIssue(
                    study_id,
                    f"过滤后概率最大绝对差 {worst:.6g} 超过容差 {tolerance:g}",
                    worst,
                    image_count,
                )
            )

    return ComparisonReport(
        checked_studies=len(study_probabilities),
        checked_images=sum(counts.get(study_id, 0) for study_id in study_probabilities),
        inconsistent=tuple(issues),
        max_abs_diff=max_abs_diff,
        filtered_submission=submission.filtered,
        threshold=threshold,
        tolerance=tolerance,
    )


@dataclass(frozen=True)
class TimingSummary:
    """3 轮计时汇总：每张图先取 3 轮均值，再对全部图取平均。"""

    images: int
    rounds: int
    per_image_mean_ms: tuple[float, ...]
    mean_ms: float
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    total_ms: float
    limit_ms: float = COST_LIMIT_MS

    @property
    def within_limit(self) -> bool:
        """平均单张耗时是否不超过规则上限。"""
        return self.mean_ms <= self.limit_ms


def summarize_timings(
    round_times_ms: Sequence[Sequence[float]],
    limit_ms: float = COST_LIMIT_MS,
) -> TimingSummary:
    """按规则汇总计时：先「每张图 3 轮取均值」，再「对全部图取平均」。

    `round_times_ms[round][image]` 是某轮某张图的端到端毫秒数。允许轮数或图数
    与默认值不同（测试用 1 轮 2 张图），但要求每一轮的图数一致、至少一张图。
    """
    if not round_times_ms:
        raise ValueError("计时结果为空，至少要有 1 轮")
    per_round = [list(times) for times in round_times_ms]
    if any(len(times) == 0 for times in per_round):
        raise ValueError("每一轮至少要计一张图")
    image_counts = {len(times) for times in per_round}
    if len(image_counts) != 1:
        raise ValueError(f"各轮计时的图片数不一致：{sorted(image_counts)}")

    values = np.asarray(per_round, dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("计时结果必须是有限的非负数")

    per_image = values.mean(axis=0)
    return TimingSummary(
        images=int(per_image.size),
        rounds=len(per_round),
        per_image_mean_ms=tuple(float(value) for value in per_image),
        mean_ms=float(per_image.mean()),
        median_ms=float(np.median(per_image)),
        p95_ms=float(np.percentile(per_image, 95)),
        min_ms=float(per_image.min()),
        max_ms=float(per_image.max()),
        total_ms=float(values.sum()),
        limit_ms=float(limit_ms),
    )


def format_conclusion(
    timing: TimingSummary,
    comparison: ComparisonReport,
    *,
    precision: str,
    warmup_images: int,
    final_timing: bool = True,
) -> str:
    """生成规则示例要求的单行 `[CONCLUSION]` 结论。

    形如：
    `[CONCLUSION] 结论：100 张样本、batch=1 串行、连续 3 轮；单张端到端墙钟时间
    先按每张 3 轮取均值，再对 100 张取平均，最终平均耗时 95.2 ms，未超过上限；
    预测结果与 submission.csv 比对 100 张图片（58 个 study），不一致 0 张；
    推理精度 fp16；预热 5 张图（不计入成绩）。`
    """
    if warmup_images <= 0:
        raise ValueError(f"warmup_images 必须为正数，当前为 {warmup_images!r}")
    status = ("未超过上限" if timing.within_limit else "超过上限") if final_timing else "多图 study 最终耗时无法判定"
    timing_kind = "单张端到端墙钟时间" if final_timing else "单张候选结果墙钟时间"
    return (
        f"[CONCLUSION] 结论：{timing.images} 张样本、batch=1 串行、"
        f"连续 {timing.rounds} 轮；{timing_kind}先按每张 {timing.rounds} 轮取均值，"
        f"再对 {timing.images} 张取平均，平均耗时 {timing.mean_ms:.1f} ms，{status}；"
        f"预测结果与 submission.csv 比对 {comparison.checked_images} 张图片"
        f"（{comparison.checked_studies} 个 study），不一致 "
        f"{comparison.inconsistent_images} 张；推理精度 {precision}；"
        f"预热 {warmup_images} 张图（不计入成绩）。"
    )
