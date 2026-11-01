"""患者级重划分及可核验的运行清单。"""
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np

from common.config import Config
from data.index import StudyRecord, read_split

MANIFEST_NAME = "split_manifest.json"


def _records(cfg: Config) -> tuple[list[StudyRecord], set[str]]:
    train = read_split(cfg.data.train.csv, cfg.data.train.images)
    valid = read_split(cfg.data.valid.csv, cfg.data.valid.images)
    for field in ("study_id", "subject_id"):
        a = {getattr(r, field) for r in train}
        b = {getattr(r, field) for r in valid}
        if a & b:
            raise ValueError(f"合并数据的 {field} 跨来源重复：{sorted(a & b)[:5]}")
    a = {p.stem for r in train for p in r.image_paths}
    b = {p.stem for r in valid for p in r.image_paths}
    if a & b:
        raise ValueError(f"合并数据的 Dicom_id 重复：{sorted(a & b)[:5]}")
    return train + valid, {r.study_id for r in train}


def _manifest(cfg: Config, records: list[StudyRecord], train_ids: set[str]) -> dict:
    subjects = sorted({r.subject_id for r in records})
    n_valid = math.ceil(len(subjects) * cfg.data.resplit.valid_fraction)
    if n_valid < 1 or n_valid >= len(subjects):
        raise ValueError("患者数不足以划分出非空训练集和验证集")
    rng = np.random.default_rng(cfg.data.resplit.seed)
    selected = set(rng.choice(subjects, size=n_valid, replace=False).tolist())
    entries = []
    for r in sorted(records, key=lambda item: item.study_id):
        entries.append({"study_id": r.study_id, "subject_id": r.subject_id,
                        "dicom_ids": [p.stem for p in r.image_paths],
                        "labels": r.labels.tolist(), "views": list(r.views),
                        "source": ""})
    # 来源按官方 train 的 study 集合确定；图片路径本身可在不同机器上变化。
    for entry in entries:
        entry["source"] = "train" if entry["study_id"] in train_ids else "valid"
        entry["split"] = "valid" if entry["subject_id"] in selected else "train"
    return {"version": 1, "seed": cfg.data.resplit.seed,
            "valid_fraction": cfg.data.resplit.valid_fraction, "studies": entries}


def prepare_resplit(cfg: Config, manifest_path: str | Path, *, resume: bool = False) -> dict:
    path = Path(manifest_path)
    if resume and not path.is_file():
        raise FileNotFoundError(f"续训缺少划分清单：{path}")
    records, train_ids = _records(cfg)
    manifest = _manifest(cfg, records, train_ids)
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != manifest:
            raise ValueError("划分清单与当前数据或配置不一致")
        return saved
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(manifest, file, ensure_ascii=False, indent=2)
            file.write("\n")
        Path(temporary_name).replace(path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    return manifest


def load_resplit_records(cfg: Config, split: str, manifest_path: str | Path | None) -> list[StudyRecord]:
    if manifest_path is None or not Path(manifest_path).is_file():
        raise FileNotFoundError("重划分数据必须提供训练时的 split_manifest.json")
    records, train_ids = _records(cfg)
    saved = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if saved != _manifest(cfg, records, train_ids):
        raise ValueError("划分清单与当前数据或配置不一致")
    ids = {e["study_id"] for e in saved["studies"] if e["split"] == split}
    return [r for r in records if r.study_id in ids]


def manifest_digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checkpoint_manifest_path(cfg: Config, checkpoint: dict, checkpoint_path: str | Path) -> Path | None:
    """重划分权重必须绑定相邻清单，不允许悄悄回退官方验证集。"""
    raw = checkpoint.get("config") or {}
    enabled = raw.get("data", {}).get("resplit", {}).get("enabled", False)
    if enabled != cfg.data.resplit.enabled:
        raise ValueError("checkpoint 与评估配置的重划分设置不一致")
    if not enabled:
        return None
    path = Path(checkpoint_path).parent / MANIFEST_NAME
    expected = (checkpoint.get("metadata") or {}).get("split_manifest_sha256")
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint 缺少划分清单：{path}")
    if not expected or manifest_digest(path) != expected:
        raise ValueError("checkpoint 与划分清单的 SHA-256 不一致")
    return path
