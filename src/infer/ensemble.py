"""多模型 study-level 预测融合。

本模块只处理已经完成各自 study 聚合的宽表，不负责加载模型或读取图片。
这样可以把融合公式和 ID/概率校验与命令行编排分开测试。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from common.constants import CLASS_IDS


def normalize_weights(
    weights: Sequence[float] | None,
    count: int,
) -> np.ndarray:
    """校验并归一化模型权重。"""
    if count < 1:
        raise ValueError("模型数量必须至少为 1")
    if weights is None:
        values = np.ones(count, dtype=np.float64)
    else:
        values = np.asarray(list(weights), dtype=np.float64)
        if values.ndim != 1 or len(values) != count:
            raise ValueError(f"权重数量必须与模型数量一致，当前为 {len(values)} 与 {count}")
    if not np.isfinite(values).all():
        raise ValueError("权重必须是有限数字")
    if (values < 0).any():
        raise ValueError("权重必须为非负数")
    total = float(values.sum())
    if total <= 0:
        raise ValueError("至少需要一个权重为正")
    return values / total


def _columns(method: str) -> list[str]:
    if method == "probability":
        return [f"p{class_id}" for class_id in CLASS_IDS]
    if method == "logit":
        return [f"z{class_id}" for class_id in CLASS_IDS]
    raise ValueError("fusion method 只能是 'probability' 或 'logit'")


def _validate_frame(frame: pd.DataFrame, columns: list[str], index: int) -> None:
    if "Study_id" not in frame.columns:
        raise ValueError(f"第 {index} 个模型结果缺少 Study_id")
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"第 {index} 个模型结果缺少列 {missing[0]}")
    if frame["Study_id"].duplicated().any():
        raise ValueError(f"第 {index} 个模型结果含重复 Study_id")
    values = frame[columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"第 {index} 个模型结果含 NaN 或无穷值")
    if columns[0].startswith("p") and ((values < 0).any() or (values > 1).any()):
        raise ValueError(f"第 {index} 个模型概率必须落在 [0, 1]")


def fuse_prediction_frames(
    frames: Sequence[pd.DataFrame],
    weights: Sequence[float] | None,
    method: str,
) -> pd.DataFrame:
    """按 Study_id 融合多个概率或 logits 宽表，返回概率宽表。"""
    if not frames:
        raise ValueError("至少需要一个模型结果")
    columns = _columns(method)
    for index, frame in enumerate(frames, start=1):
        _validate_frame(frame, columns, index)

    first_ids = frames[0]["Study_id"].tolist()
    first_set = set(first_ids)
    aligned: list[np.ndarray] = []
    for index, frame in enumerate(frames, start=1):
        if set(frame["Study_id"]) != first_set:
            raise ValueError(f"第 {index} 个模型结果的 Study_id 与第 1 个模型不一致")
        indexed = frame.set_index("Study_id").loc[first_ids]
        aligned.append(indexed[columns].to_numpy(dtype=np.float64))

    normalized = normalize_weights(weights, len(frames))
    stacked = np.stack(aligned, axis=0)
    fused = np.tensordot(normalized, stacked, axes=(0, 0))
    if method == "logit":
        fused = 1.0 / (1.0 + np.exp(-fused))
    fused = np.asarray(fused, dtype=np.float64)
    if not np.isfinite(fused).all() or (fused < 0).any() or (fused > 1).any():
        raise ValueError("融合结果必须是 [0, 1] 内的有限概率")

    result = pd.DataFrame(fused, columns=[f"p{class_id}" for class_id in CLASS_IDS])
    result.insert(0, "Study_id", first_ids)
    return result
