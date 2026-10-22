"""数据体检：训练之前，一次性确认 CSV、图片和数据划分没有明显问题。

用法（在仓库根目录执行，配置里的相对路径才落在预期位置）：

    python src/check_data.py --config configs/default.yaml          # 快速体检
    python src/check_data.py --config configs/default.yaml --full   # 全量体检

默认只读每份 CSV 的前 1000 行，方便在本地快速跑一遍；`--full` 读全量数据，
正式训练前应该至少跑一次 `--full`。

检查内容：

- CSV 是否含必需列；
- ID 是否为空、是否保持 `subject_00000` 这样的前导零格式；
- 每个 Dicom_id 是否都能在对应的图片目录里找到；
- 同一 study 的多张图是否属于同一患者、标签是否完全一致；
- train / valid / test 之间的 Subject_id、Study_id、Dicom_id 是否重叠；
- 标签分布：每个类别有多少张图、多少个 study，以及类别 10 的使用情况。

任何一项不通过，脚本会打印原因并以退出码 1 结束，方便直接串进训练前的流程。
"""

import argparse
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from common.config import Config, load_config
from common.constants import CSV_ENCODING, NUM_CLASSES
from data.index import index_images
from data.labels import parse_label_string

# 三份数据划分。test 没有 Predict_class 列，其余两份有。
SPLITS = ("train", "valid", "test")
LABELED_SPLITS = ("train", "valid")

# 每份 CSV 都必须有的列；标签列只对 train / valid 做要求。
REQUIRED_COLUMNS = ("Dicom_id", "Subject_id", "Study_id")

# 快速模式读多少行。1000 行足以暴露列名、ID 格式这类系统性问题，
# 又不用等全量 CSV 和几万张图片扫描完。
FAST_ROWS = 1000

# ID 的固定格式：前缀 + 补零到固定位数的编号。
# 固定位数正是「前导零」的验收点：`subject_0`、`pic_12345` 这类被截断的
# 编号在这里会直接判为不合格。
ID_PATTERNS = {
    "Dicom_id": re.compile(r"^pic_\d{6}$"),
    "Subject_id": re.compile(r"^subject_\d{5}$"),
    "Study_id": re.compile(r"^study_\d{5}$"),
}


class Checker:
    """收集检查结果，最后统一汇总。

    一项检查不通过时只记录、不抛异常，这样一次运行能看到所有问题，
    而不是修好一个再跑一遍才发现下一个。
    """

    def __init__(self) -> None:
        self.failures: list[str] = []

    def info(self, message: str) -> None:
        """打印一条统计信息，不参与通过/失败判定。"""
        print(f"  · {message}")

    def check(self, ok: bool, message: str, detail: str = "") -> None:
        """打印一项检查；不通过时连同细节记进 failures。"""
        mark = "✓" if ok else "✗"
        print(f"  {mark} {message}" + (f"：{detail}" if detail else ""))
        if not ok:
            self.failures.append(f"{message}（{detail}）" if detail else message)


def read_split(csv_path: Path, labeled: bool, full: bool) -> pd.DataFrame:
    """读取一份标注 CSV，并确认必需列都在。

    两个关键点：

    - `dtype=str` + `keep_default_na=False`：ID 全程按字符串读，
      否则 `subject_00000` 会被 pandas 当成数字，前导零在读取时就丢了；
    - `nrows`：快速模式只读前 1000 行，`--full` 时读全部。
    """
    frame = pd.read_csv(
        csv_path,
        dtype=str,
        keep_default_na=False,
        encoding=CSV_ENCODING,
        nrows=None if full else FAST_ROWS,
    )

    required = REQUIRED_COLUMNS + (("Predict_class",) if labeled else ())
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"缺少必需的列 {missing}，实际列为 {list(frame.columns)}")

    return frame


def check_ids(frame: pd.DataFrame, checker: Checker) -> None:
    """检查三个 ID 列没有空值，并且都保持补零格式。"""
    for column, pattern in ID_PATTERNS.items():
        values = frame[column]

        empty = int((values == "").sum())
        checker.check(empty == 0, f"{column} 无空值", f"{empty} 个空值" if empty else "")

        # fullmatch 要求整个字符串都符合格式，避免 `subject_00000_x` 这种漏网。
        bad = values[~values.str.fullmatch(pattern.pattern)]
        detail = f"例如 {bad.iloc[0]!r}，共 {len(bad)} 个" if len(bad) else ""
        checker.check(bad.empty, f"{column} 保持前导零格式", detail)


def check_duplicates(frame: pd.DataFrame, checker: Checker) -> None:
    """同一 split 内 Dicom_id 只能出现一次（图片名即主键）。"""
    duplicated = frame["Dicom_id"][frame["Dicom_id"].duplicated()]
    detail = ""
    if len(duplicated):
        detail = f"例如 {duplicated.iloc[0]!r}，共 {duplicated.nunique()} 个重复"
    checker.check(duplicated.empty, "Dicom_id 在本 split 内唯一", detail)


def check_images(frame: pd.DataFrame, images_root: Path, checker: Checker) -> None:
    """逐个确认 CSV 里的 Dicom_id 都能在图片目录中找到。"""
    try:
        images = index_images(images_root)
    except (OSError, ValueError) as error:
        # 目录不存在或图片名重复都在这里暴露，后面的逐张核对也就没有意义了。
        checker.check(False, "图片目录可索引", str(error))
        return

    missing = [dicom_id for dicom_id in frame["Dicom_id"] if dicom_id not in images]
    found = len(frame) - len(missing)
    detail = ""
    if missing:
        detail = f"缺少 {len(missing):,} 张，例如 {', '.join(missing[:5])}"
    checker.check(
        not missing, f"图片齐全（{found:,} / {len(frame):,}）", detail
    )


def check_labels(frame: pd.DataFrame, checker: Checker) -> None:
    """解析全部标签，并检查同一 study 内的标签与患者是否一致。

    解析只做一遍：得到的 multi-hot 矩阵同时用于一致性检查和标签分布统计，
    避免把同一列字符串反复解析几次。
    """
    vectors = np.zeros((len(frame), NUM_CLASSES), dtype=np.float32)
    parse_errors: list[str] = []
    for row, text in enumerate(frame["Predict_class"]):
        try:
            vectors[row] = parse_label_string(text)
        except ValueError as error:
            parse_errors.append(f"{frame['Dicom_id'].iloc[row]}：{error}")

    detail = f"例如 {parse_errors[0]}，共 {len(parse_errors)} 行" if parse_errors else ""
    checker.check(not parse_errors, "标签串均可解析", detail)

    # groupby.indices 给的是「行号」，正好用来同时取标签向量和 ID 列。
    groups = frame.groupby("Study_id", sort=False).indices

    label_conflicts: list[str] = []
    subject_conflicts: list[str] = []
    for study_id, rows in groups.items():
        first = rows[0]
        if frame["Subject_id"].iloc[rows].nunique() > 1:
            subject_conflicts.append(f"{study_id} 对应多个 Subject_id")
        for row in rows[1:]:
            # 比较解析后的向量而不是原始字符串：`7;6` 和 `6;7` 语义相同，
            # 只有类别集合真的不同才算标注不一致。
            if not np.array_equal(vectors[first], vectors[row]):
                label_conflicts.append(
                    f"{study_id} 下 {frame['Dicom_id'].iloc[first]} 与 "
                    f"{frame['Dicom_id'].iloc[row]} 标签不同"
                )
                break

    detail = f"例如 {label_conflicts[0]}，共 {len(label_conflicts)} 个 study" if label_conflicts else ""
    checker.check(not label_conflicts, "同一 study 的标签一致", detail)

    detail = f"例如 {subject_conflicts[0]}，共 {len(subject_conflicts)} 个 study" if subject_conflicts else ""
    checker.check(not subject_conflicts, "同一 study 属于同一患者", detail)

    _print_label_distribution(frame, vectors, groups, checker)


def _print_label_distribution(
    frame: pd.DataFrame,
    vectors: np.ndarray,
    groups: dict[str, np.ndarray],
    checker: Checker,
) -> None:
    """打印每类命中多少张图、多少个 study，以及类别 10 的单独出现次数。"""
    # 一个 study 取第一张图代表它：标签一致时这样统计最省事，
    # 不一致时上面已经报错了，这里仍能给出大致分布。
    study_rows = [rows[0] for rows in groups.values()]
    image_counts = vectors.sum(axis=0).astype(int)
    study_counts = (
        vectors[study_rows].sum(axis=0).astype(int)
        if study_rows
        else np.zeros(NUM_CLASSES, dtype=int)
    )
    empty = int((vectors.sum(axis=1) == 0).sum())

    checker.info(f"标签串 {frame['Predict_class'].nunique():,} 种；无标签图像 {empty:,} 张")
    print("    类别   图像数   检查数")
    for index in range(NUM_CLASSES):
        print(
            f"    {index + 1:>4}   {image_counts[index]:>7,}   "
            f"{study_counts[index]:>7,}"
        )

    # 类别 10 在数据里比较特殊：它从不和其他类别同时出现。
    # 单独统计一下，方便和 plan0 里记录的基准数字对照。
    only_class_10 = (vectors.sum(axis=1) == 1) & (vectors[:, NUM_CLASSES - 1] == 1)
    study_only_class_10 = (
        int(only_class_10[study_rows].sum()) if study_rows else 0
    )
    checker.info(
        f"类别 10 单独成标签：{int(only_class_10.sum()):,} 张图 / "
        f"{study_only_class_10:,} 个 study"
    )


def check_overlaps(
    frames: dict[str, pd.DataFrame], checker: Checker, full: bool
) -> None:
    """两两比较三份数据划分，确认患者、检查、图片都不重复。

    train/valid 重叠意味着验证集信息泄漏；test 与训练集重叠则会让线上成绩
    虚高，所以这里一律要求为 0。
    """
    print("\n== 跨 split 重叠 ==")
    mode = "全量" if full else "抽样"
    for first, second in combinations(SPLITS, 2):
        for column in ("Subject_id", "Study_id", "Dicom_id"):
            overlap = set(frames[first][column]) & set(frames[second][column])
            detail = ""
            if overlap:
                preview = "、".join(sorted(overlap)[:5])
                detail = f"重叠 {len(overlap)} 个，例如 {preview}"
            checker.check(
                not overlap,
                f"{first}/{second} 的 {column} 不重叠（{mode}）",
                detail,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练前的数据体检")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="YAML 配置文件路径，默认 configs/default.yaml",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help=f"读取全量 CSV 并逐行核对图片（默认每个 split 只查前 {FAST_ROWS} 行）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config: Config = load_config(args.config)

    mode = "全量" if args.full else f"抽样（每份 CSV 前 {FAST_ROWS} 行）"
    print(f"数据体检：{args.config}，模式：{mode}")

    checker = Checker()
    frames: dict[str, pd.DataFrame] = {}

    for split in SPLITS:
        split_config = getattr(config.data, split)
        print(f"\n== {split}：{split_config.csv} ==")

        try:
            frame = read_split(
                split_config.csv, labeled=split in LABELED_SPLITS, full=args.full
            )
        except (OSError, ValueError) as error:
            checker.check(False, "CSV 可读取", str(error))
            continue

        frames[split] = frame
        checker.info(
            f"{len(frame):,} 张图 / {frame['Study_id'].nunique():,} 个 study / "
            f"{frame['Subject_id'].nunique():,} 位患者"
        )
        checker.info(f"列：{', '.join(frame.columns)}")

        check_ids(frame, checker)
        check_duplicates(frame, checker)
        check_images(frame, split_config.images, checker)
        if split in LABELED_SPLITS:
            check_labels(frame, checker)

    if len(frames) == len(SPLITS):
        check_overlaps(frames, checker, args.full)
    else:
        checker.info("有 CSV 没读成功，跳过跨 split 重叠检查")

    print()
    if checker.failures:
        print(f"体检未通过：共 {len(checker.failures)} 项问题")
        for failure in checker.failures:
            print(f"  - {failure}")
        raise SystemExit(1)

    print("全部检查通过")


if __name__ == "__main__":
    main()
