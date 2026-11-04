"""离线教师缓存与多标签软目标损失。"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

COLUMNS = ["Study_id", *[f"p{i}" for i in range(1, 11)]]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def records_digest(records: list[Any]) -> str:
    rows = [(r.study_id, r.subject_id, [p.name for p in r.image_paths], list(r.views)) for r in records]
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _validate(frame: pd.DataFrame) -> None:
    if list(frame.columns) != COLUMNS or frame.empty:
        raise ValueError("教师缓存列顺序错误或缓存为空")
    ids = frame["Study_id"]
    if ids.isna().any() or ids.eq("").any() or ids.duplicated().any():
        raise ValueError("教师缓存 Study_id 缺失或重复")
    values = frame[COLUMNS[1:]].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("教师概率必须是 [0, 1] 内的有限数")


def write_teacher_cache(path: Path, frame: pd.DataFrame, manifest: dict[str, Any]) -> None:
    _validate(frame)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"教师缓存已存在: {path}")
    temp = Path(tempfile.mkdtemp(prefix=f".{path.name}.", dir=path.parent))
    try:
        csv = temp / "predictions.csv"
        frame.to_csv(csv, index=False, encoding="utf-8", float_format="%.17g")
        data = {**manifest, "version": 1, "predictions_sha256": sha256_file(csv)}
        (temp / "manifest.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.rename(temp, path)
    finally:
        if temp.exists():
            shutil.rmtree(temp)


class TeacherCache:
    def __init__(self, values: dict[str, np.ndarray], digest: str, manifest: dict[str, Any]):
        self.values = values
        self.digest = digest
        self.manifest = manifest

    @classmethod
    def load(cls, path: Path, study_ids: list[str], train_csv_sha256: str, *, index_sha256: str | None = None) -> TeacherCache:
        path = Path(path)
        manifest_path, csv = path / "manifest.json", path / "predictions.csv"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("version") != 1 or manifest.get("predictions_sha256") != sha256_file(csv):
            raise ValueError("教师缓存摘要不匹配")
        if manifest.get("train_csv_sha256") != train_csv_sha256:
            raise ValueError("教师缓存训练数据摘要不匹配")
        if index_sha256 is not None and (manifest.get("index_sha256") != index_sha256 or manifest.get("fusion") != "logit_clipped_probability" or manifest.get("weights") != [4, 2, 3] or manifest.get("split") != "train"):
            raise ValueError("教师缓存索引或融合来源不匹配")
        frame = pd.read_csv(csv, dtype={"Study_id": str}, keep_default_na=False)
        _validate(frame)
        ids = frame["Study_id"].tolist()
        if len(study_ids) != len(set(study_ids)) or set(ids) != set(study_ids):
            raise ValueError("教师缓存与训练 Study_id 不一致")
        values = {row[0]: np.asarray(row[1:], dtype=np.float32) for row in frame.itertuples(index=False, name=None)}
        digest = hashlib.sha256((sha256_file(manifest_path) + sha256_file(csv)).encode()).hexdigest()
        return cls(values, digest, manifest)

    def batch(self, study_ids: list[str], device: torch.device) -> torch.Tensor:
        return torch.as_tensor(np.stack([self.values[item] for item in study_ids]), device=device)


def distillation_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits.float(), targets.float())
