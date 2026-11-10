"""按允许的数据范围自动统计患者检查数；不读取影像、标签或预测。"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from data.index import StudyRecord

ID_COLUMNS = ("Subject_id", "Study_id")
COUNT_COLUMN = "patient_study_count"


def _validate_ids(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    if frame.empty:
        raise ValueError("患者计数元数据或预测不能为空")
    if not frame.columns.is_unique:
        raise ValueError("元数据/预测存在重复列名")
    for column in columns:
        if column not in frame:
            raise ValueError(f"缺少字段：{column}")
        # 不将整数转成字符串掩盖前导零已经丢失的情况。
        if not frame[column].map(lambda value: isinstance(value, str) and bool(value.strip())
                                 and value == value.strip()).all():
            raise ValueError(f"{column} 必须为无首尾空白的非空字符串，读取 CSV 时须使用 dtype=str")


def read_patient_metadata(path: str | Path) -> pd.DataFrame:
    """只读取 ID 列，忽略标签与其他元数据。"""
    return pd.read_csv(path, usecols=list(ID_COLUMNS), dtype=str, keep_default_na=False,
                       encoding="utf-8")


def build_patient_counts(metadata: pd.DataFrame) -> pd.DataFrame:
    """返回 Subject_id、Study_id、patient_study_count，每次检查一行。"""
    _validate_ids(metadata, ID_COLUMNS)
    mapping = metadata.loc[:, list(ID_COLUMNS)].drop_duplicates().copy()
    if mapping["Study_id"].duplicated().any():
        raise ValueError("同一个 Study_id 对应多个 Subject_id")
    mapping[COUNT_COLUMN] = mapping.groupby("Subject_id", sort=False)["Study_id"].transform("size")
    return mapping.reset_index(drop=True)


def counts_from_records(records: Iterable[StudyRecord]) -> pd.DataFrame:
    """由当前 loader 的完整记录自动计数，避免混入其他 split。"""
    return build_patient_counts(pd.DataFrame(
        [{"Subject_id": record.subject_id, "Study_id": record.study_id} for record in records],
        columns=list(ID_COLUMNS),
    ))


def align_patient_counts(predictions: pd.DataFrame, counts: pd.DataFrame) -> np.ndarray:
    """严格按预测的 Study_id 顺序对齐，拒绝缺失、额外 ID 或患者映射冲突。"""
    _validate_ids(predictions, ("Study_id",))
    _validate_ids(counts, ID_COLUMNS)
    if predictions["Study_id"].duplicated().any() or counts["Study_id"].duplicated().any():
        raise ValueError("预测与计数表必须每个 Study_id 恰好一行")
    if set(predictions["Study_id"]) != set(counts["Study_id"]):
        raise ValueError("预测与计数表的 Study_id 集合不同")
    if COUNT_COLUMN not in counts:
        raise ValueError(f"计数表缺少 {COUNT_COLUMN}")
    aligned = counts.set_index("Study_id").loc[predictions["Study_id"]]
    if "Subject_id" in predictions:
        _validate_ids(predictions, ID_COLUMNS)
        if not np.array_equal(predictions["Subject_id"].to_numpy(), aligned["Subject_id"].to_numpy()):
            raise ValueError("预测与计数表的 Subject_id 映射不一致")
    values = aligned[COUNT_COLUMN].to_numpy()
    if values.dtype.kind not in "iuf" or not np.isfinite(values).all() or (values < 1).any():
        raise ValueError("患者检查数必须为正整数")
    if not np.equal(values, np.floor(values)).all():
        raise ValueError("患者检查数必须为正整数")
    # 计数表必须覆盖当前完整输入，不能手工塞入其他范围的计数。
    expected = counts.groupby("Subject_id", sort=False)["Study_id"].transform("size").to_numpy()
    if not np.array_equal(counts[COUNT_COLUMN].to_numpy(), expected):
        raise ValueError("计数值与元数据中的不同 Study_id 数不一致")
    return values.astype(np.int64)
