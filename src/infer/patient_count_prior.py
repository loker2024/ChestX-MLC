"""冻结的患者检查次数偏置；仅支持 train/valid 离线研究。

计数来自当前完整输入而非 batch。正式 test 的跨检查统计尚未获许可，
所有面向用户的入口都必须先调用 require_research_split。
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from data.patient_counts import align_patient_counts

BETA_GRID = (0.0, 0.05, 0.10, 0.20)
PROBABILITY_COLUMNS = tuple(f"p{c}" for c in range(1, 11))


def sha256_file(path: str | Path) -> str:
    """分块计算文件摘要，不将权重整体载入内存。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_research_split(split: str) -> None:
    """在 test 读数据/加载模型之前失败，不用 JSON 字段冒充主办方许可。"""
    if split not in {"train", "valid"}:
        raise ValueError("患者检查次数后处理目前仅支持 train/valid 离线研究；test 跨检查计数须先取得主办方书面许可")


def recipe_fingerprint(configs: Sequence[Any], weights: Sequence[float] | None = None,
                       fusion: str = "probability") -> str:
    """绑定基础模型配方；排除不同机器的数据和输出目录。"""
    if not configs or fusion not in {"probability", "logit"}:
        raise ValueError("配方至少需要一个模型且 fusion 必须为 probability/logit")
    values = np.ones(len(configs), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    if (values.shape != (len(configs),) or not np.isfinite(values).all()
            or (values < 0).any() or not np.isfinite(values.sum()) or values.sum() <= 0):
        raise ValueError("配方融合权重非法")
    members = []
    for config in configs:
        raw = config.to_dict()
        train = dict(raw.get("train", {}))
        for key in ("save_last", "tensorboard"):
            train.pop(key, None)
        members.append({"model": raw.get("model", {}), "image": raw.get("image", {}),
                        "train": train, "seed": raw.get("seed", 42)})
    payload = {"members": members, "weights": (values / values.sum()).tolist(), "fusion": fusion}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PatientCountPrior:
    """第一期固定函数，仅 beta 为候选强度，其余数学定义不可调。"""

    beta: float = 0.0
    schema_version: int = 1
    method: str = "signed_capped_log_count"
    count_scope: str = "split_distinct_studies"
    cap: int = 10
    class_signs: tuple[int, ...] = (1, 1, 1, 1, 1, 1, 1, 1, 1, -1)
    probability_eps: float = 1e-6
    selection_source: str = "official_train_oof"
    deployment_status: str = "research_only"
    recipe_sha256: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.beta, bool) or not isinstance(self.beta, (float, int)) or self.beta not in BETA_GRID:
            raise ValueError(f"beta 必须在固定候选网格 {BETA_GRID} 中")
        if (type(self.schema_version) is not int or self.schema_version != 1
                or self.method != "signed_capped_log_count" or self.count_scope != "split_distinct_studies"
                or type(self.cap) is not int or self.cap != 10
                or tuple(self.class_signs) != (1,) * 9 + (-1,)
                or any(type(value) is not int for value in self.class_signs)
                or isinstance(self.probability_eps, bool) or self.probability_eps != 1e-6):
            raise ValueError("患者次数参数的 schema、统计范围或固定数学定义不一致")
        if self.selection_source != "official_train_oof" or self.deployment_status != "research_only":
            raise ValueError("目前仅接受 official_train_oof / research_only 参数")
        if not isinstance(self.recipe_sha256, str) or (self.recipe_sha256 and
                (len(self.recipe_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.recipe_sha256))):
            raise ValueError("recipe_sha256 必须为 SHA-256 十六进制摘要")
        if not isinstance(self.provenance, dict):
            raise ValueError("provenance 必须为对象")


def load_prior(path: str | Path) -> PatientCountPrior:
    """严格读取参数文件，禁止遗漏固定字段或忽略未知字段。"""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    names = {item.name for item in fields(PatientCountPrior)}
    if not isinstance(raw, dict) or set(raw) != names:
        raise ValueError("患者次数参数字段缺失或包含未知字段")
    if not isinstance(raw["class_signs"], list):
        raise ValueError("class_signs 必须为十类整数数组")
    raw["class_signs"] = tuple(raw["class_signs"])
    return PatientCountPrior(**raw)


def save_prior(prior: PatientCountPrior, path: str | Path) -> Path:
    """以 UTF-8 保存冻结定义与可追溯的选择记录。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(asdict(prior), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    return target


def validate_prior_recipe(prior: PatientCountPrior, configs: Sequence[Any],
                          weights: Sequence[float] | None = None, fusion: str = "probability") -> None:
    """推理入口不接受无来源或不匹配的模型配方。"""
    if not prior.recipe_sha256 or prior.recipe_sha256 != recipe_fingerprint(configs, weights, fusion):
        raise ValueError("患者次数参数与当前基础模型/融合配方不匹配，须在对应配方 OOF 上选型")
    for key in ("source_csv_sha256", "manifest_sha256", "oof_sha256"):
        value = prior.provenance.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"患者次数参数缺少有效来源摘要：{key}")


def validate_prior_checkpoints(prior: PatientCountPrior, checkpoints: Sequence[dict[str, Any]],
                               weights: Sequence[float] | None = None,
                               fusion: str = "probability") -> None:
    """核对实际权重存档的配方，避免仅修改外部 YAML 就套用旧 best 权重。"""
    from common.config import config_from_dict

    saved_configs = []
    for checkpoint in checkpoints:
        raw = checkpoint.get("config")
        metadata = checkpoint.get("metadata") or {}
        if not isinstance(raw, dict):
            raise ValueError("患者次数复核要求 checkpoint 保存完整 config")
        config = config_from_dict(raw)
        if (config.data.full_train or config.data.resplit.enabled or checkpoint.get("epoch") != 5
                or metadata.get("weights_kind") != "fixed_last_ema"
                or not metadata.get("source_last_sha256") or not metadata.get("ema_updates")
                or not metadata.get("pretrained_loaded") or metadata.get("warm_start_from")):
            raise ValueError("患者次数复核仅接受 train-only 固定第5轮 EMA，不能使用早停 best 或含 valid 的模型")
        saved_configs.append(config)
    validate_prior_recipe(prior, saved_configs, weights, fusion)


def adjust_probabilities(probabilities: np.ndarray, counts: np.ndarray,
                         prior: PatientCountPrior) -> np.ndarray:
    """按标量/[N] 次数调整 [10]/[N,10] 得分，保留零强度与 n=1 的原值。"""
    p = np.asarray(probabilities, dtype=np.float64)
    n = np.asarray(counts)
    if p.ndim not in (1, 2) or p.shape[-1] != 10 or p.size == 0:
        raise ValueError("概率必须为非空 [10] 或 [N,10]")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("概率必须为 [0,1] 内的有限数")
    expected = () if p.ndim == 1 else (len(p),)
    if (n.shape != expected or n.dtype.kind not in "iuf" or not np.isfinite(n).all()
            or (n < 1).any() or not np.equal(n, np.floor(n)).all()):
        raise ValueError("检查次数必须为正整数，形状须与概率匹配")
    result = p.copy()
    if prior.beta == 0:
        return result
    matrix = result.reshape(-1, 10)
    g = np.log1p(np.minimum(n.reshape(-1), prior.cap)) - math.log(2.0)
    # 直接按次数判定，避免不同数学库计算 log(2) 的舍入差触发 n=1 行。
    active = n.reshape(-1) > 1
    selected = np.clip(matrix[active], prior.probability_eps, 1.0 - prior.probability_eps)
    logits = np.log(selected) - np.log1p(-selected)
    logits += prior.beta * g[active, None] * np.asarray(prior.class_signs)[None, :]
    matrix[active] = 1.0 / (1.0 + np.exp(-logits))
    return result


def apply_prior_frame(predictions: pd.DataFrame, counts: pd.DataFrame,
                      prior: PatientCountPrior) -> pd.DataFrame:
    """宽表入口：原顺序、标签和额外列均保留，不原地修改输入。"""
    if any(column not in predictions for column in PROBABILITY_COLUMNS):
        raise ValueError("预测缺少 p1..p10 概率列")
    aligned = align_patient_counts(predictions, counts)
    adjusted = adjust_probabilities(predictions[list(PROBABILITY_COLUMNS)].to_numpy(), aligned, prior)
    result = predictions.copy()
    result[list(PROBABILITY_COLUMNS)] = adjusted
    return result
