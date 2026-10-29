"""把标注 CSV 与图片目录合成「按 study 聚合」的记录列表。

赛题的样本单位是「检查（study）」而不是「图片」：一次检查含 1~7 张胸片，
这些图片共享同一组标签。所以 read_split 做三件事：

1. 读 CSV——全程按字符串读，保住 `subject_00000` 这类 ID 的前导零
   （pandas 默认会把看起来像数字的列转成 int，前导零就丢了）；
2. 递归索引图片目录，建立「Dicom_id -> 图片路径」映射；
3. 把 Study_id 相同的若干行归并成一条 StudyRecord，并校验同 study 的标签一致
   ——一张图一个标签是数据契约，不一致就说明标注导出错了，必须当场报错。
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from common.constants import CSV_ENCODING, IMAGE_SUFFIXES
from data.labels import parse_label_string

# 每份标注 CSV 都必须有的列。ViewPosition / Rows 等列目前不参与训练，所以不校验。
REQUIRED_COLUMNS = ("Dicom_id", "Subject_id", "Study_id")


@dataclass(frozen=True)
class StudyRecord:
    """一次检查：它的图片路径、标签与所属患者。"""

    study_id: str
    subject_id: str
    image_paths: tuple[Path, ...]  # 顺序与 CSV 中出现顺序一致，保证可复现
    labels: np.ndarray | None = None  # test split 没有标注，为 None
    views: tuple[int, ...] = ()  # 0 正位，1 侧位，2 未知

    @property
    def num_images(self) -> int:
        """这次检查有几张图。"""
        return len(self.image_paths)


def index_images(root: str | Path) -> dict[str, Path]:
    """递归扫描 root，返回「图片名（不含后缀） -> 图片路径」的映射。

    目录结构是 `<subject_id>/<study_id>/<dicom_id>.jpg`，文件名去掉后缀就是
    CSV 里的 Dicom_id。这里只按 stem 建索引，不校验目录层级是否和 CSV 的
    Subject_id / Study_id 对得上——CSV 才是唯一可信来源，图片路径只用来读像素。

    Dicom_id 全局唯一，所以两个不同文件名的 stem 撞车一定是目录放错了
    （例如把两次检查的图混在同一个文件夹），直接报错比默默覆盖好。
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"图片根目录不存在：{root}")

    index: dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        stem = path.stem
        duplicated = index.get(stem)
        if duplicated is not None:
            raise ValueError(
                f"图片名重复：{duplicated} 与 {path} 的文件名都是 {stem!r}，"
                "Dicom_id 本应全局唯一"
            )
        index[stem] = path
    return index


def read_split(
    csv_path: str | Path, images_root: str | Path, labeled: bool = True
) -> list[StudyRecord]:
    """读取一份标注 CSV，按 study 聚合并配上图片路径。

    labeled=True 时要求 CSV 有 Predict_class 列并解析成标签；test split 没有
    这一列，传 labeled=False，记录里的 labels 就是 None。

    返回的列表顺序 = study 在 CSV 里第一次出现的顺序。这里不做随机打乱
    （打乱是 DataLoader 的事），同样的输入每次得到同样的顺序，便于复现和排查。
    """
    csv_path = Path(csv_path)
    # keep_default_na=False 让缺失值保持空串而不是 NaN，避免 ID 列出现 "nan"。
    frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding=CSV_ENCODING)

    required = REQUIRED_COLUMNS + (("Predict_class",) if labeled else ())
    missing_columns = [name for name in required if name not in frame.columns]
    if missing_columns:
        raise ValueError(
            f"{csv_path} 缺少必需的列 {missing_columns}，实际列为 {list(frame.columns)}"
        )

    duplicated_dicom = frame.loc[
        frame["Dicom_id"].duplicated(keep=False), "Dicom_id"
    ].unique()
    if len(duplicated_dicom) > 0:
        preview = "、".join(str(value) for value in duplicated_dicom[:5])
        raise ValueError(
            f"{csv_path} 含重复的 Dicom_id（例如 {preview}）；"
            "一行对应一次检查，重复行会让同一张图被计两次"
        )

    if "ViewPosition" not in frame.columns:
        frame["ViewPosition"] = ""
    images = index_images(images_root)

    # 先把缺图一次性找全再报错：边遍历边报错会在第一张图就中断，看不出缺口规模。
    missing_images = [dicom for dicom in frame["Dicom_id"] if dicom not in images]
    if missing_images:
        preview = "、".join(missing_images[:5])
        raise FileNotFoundError(
            f"{len(missing_images)} 张图片在 {images_root} 下找不到（例如 {preview}）；"
            "请检查配置里的 images 是否指向了正确的图片根目录"
        )

    # 同一 study 的行归到一起：study_id -> [(dicom_id, 图片路径, 原始标签串), ...]
    grouped_rows: dict[str, list[tuple[str, Path, str]]] = {}
    subjects: dict[str, str] = {}
    views_by_study: dict[str, list[int]] = {}
    for row in frame.itertuples(index=False):
        previous_subject = subjects.get(row.Study_id)
        if previous_subject is not None and previous_subject != row.Subject_id:
            raise ValueError(
                f"同一 study {row.Study_id} 对应多个 Subject_id："
                f"{previous_subject!r} 与 {row.Subject_id!r}"
            )
        grouped_rows.setdefault(row.Study_id, []).append(
            (row.Dicom_id, images[row.Dicom_id], row.Predict_class if labeled else "")
        )
        view = row.ViewPosition.strip().upper()
        view_id = 0 if view in {"AP", "PA"} else 1 if view in {"LATERAL", "LL"} else 2
        views_by_study.setdefault(row.Study_id, []).append(view_id)
        subjects[row.Study_id] = row.Subject_id

    records: list[StudyRecord] = []
    for study_id, rows in grouped_rows.items():
        labels = _study_labels(study_id, rows) if labeled else None
        records.append(
            StudyRecord(
                study_id=study_id,
                subject_id=subjects[study_id],
                image_paths=tuple(path for _, path, _ in rows),
                labels=labels,
                views=tuple(views_by_study[study_id]),
            )
        )
    return records


def _study_labels(study_id: str, rows: list[tuple[str, Path, str]]) -> np.ndarray:
    """解析一个 study 内所有图片的标签，并校验它们完全一致。

    以第一张图的标签为准；后续任何一张不一致就报错，消息里同时给出两个
    原始标签串和对应的 Dicom_id——拿它去 CSV 里一搜就能定位到出问题的行。
    """
    first_dicom, _, first_text = rows[0]
    labels = parse_label_string(first_text)
    for dicom_id, _, text in rows[1:]:
        if not np.array_equal(labels, parse_label_string(text)):
            raise ValueError(
                f"同一 study 的图片标签不一致：检查 {study_id} 下 "
                f"{first_dicom} 是 {first_text!r}，{dicom_id} 是 {text!r}"
            )
    return labels
