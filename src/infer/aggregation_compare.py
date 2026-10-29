"""固定 checkpoint 的聚合空间对照：逐图 logits 缓存、G0 / G1 配对评估。

背景（RUN-008）：赛题按「检查（study）」计分，一次检查有 1~7 张胸片。现有
推理路径 `infer.predict.predict_loader` 的口径是 G0：

    G0 = sigmoid(mean(z_i))      # 先在 logits 空间等权平均，再 sigmoid

论文 *Ensemble of ConvNeXt V2 and MaxViT for Long-Tailed CXR Classification
with View-Based Aggregation* 的视图融合用的是概率空间平均，也就是：

    G1 = mean(sigmoid(z_i))      # 先逐图 sigmoid，再在概率空间等权平均

本模块固定同一个 checkpoint，只导出一次逐图 logits 到缓存，再离线重算
G0 / G1 做配对比较；不重新训练、不换视图权重、不做 TTA。

两个子命令：

- `export`：读 `best.pt` 与 valid 集，逐批前向得到 `[N, 10]` 逐图 logits，
  连同 offsets、标签与图片身份写成 `image_logits.npz` + `metadata.json`；
  同时用 `predict_loader` 独立跑一遍，校验本次导出的 G0 与现有路径一致。
- `compare`：只读缓存与 metadata（不加载模型、不读图片），离线重算 G0 / G1，
  写出预测表、汇总表和逐类表。缓存哈希不符、ID 或计数不一致时拒绝出报告。

缓存目录里的文件：

    image_logits.npz   logits[N,10]、offsets[B+1]、labels[B,10]、
                       study_ids[B]、subject_ids[B]、dicom_ids[N]、views[N]
    metadata.json      格式版本、代码与环境指纹、checkpoint / valid 哈希、
                       图片身份清单、NPZ 哈希、G0 基准差值

注意：`compare` 只能校验缓存自身没被改动，不能证明源图片在这之后没有变化；
重新导出时以 metadata 里的新指纹辨别数据版本。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from common.checkpoint import load_checkpoint
from common.config import Config, config_from_dict, load_config
from common.constants import CLASS_IDS, CSV_ENCODING, NUM_CLASSES
from common.metrics import macro_auc, macro_f1_curve, parse_thresholds, per_class_report
from common.utils import resolve_device
from data.dataset import build_loader
from infer.predict import load_checkpoint_model, predict_loader
from model.network import aggregate_logits

# 缓存格式版本：字段增删或语义变化时加一，compare 会拒绝旧版本缓存。
CACHE_FORMAT_VERSION = 1

IMAGE_LOGITS_FILE = "image_logits.npz"
METADATA_FILE = "metadata.json"
PREDICTIONS_G0_FILE = "predictions_g0.csv"
PREDICTIONS_G1_FILE = "predictions_g1.csv"
SUMMARY_CSV_FILE = "summary.csv"
SUMMARY_MD_FILE = "summary.md"
PER_CLASS_FILE = "per_class.csv"

PROBABILITY_COLUMNS = [f"p{class_id}" for class_id in CLASS_IDS]
LABEL_COLUMNS = [f"c{class_id}" for class_id in CLASS_IDS]

# 导出时对模型前向的硬性要求：计划规定 FP32 前向、关闭 autocast / TTA。
REQUIRED_PRECISION = "fp32"


# ---------------------------------------------------------------------------
# 核心数值：G0 / G1
# ---------------------------------------------------------------------------


def _validate_logits(logits: torch.Tensor) -> torch.Tensor:
    """校验逐图 logits 并转成 fp32；非法输入给出明确报错。"""
    if not torch.is_tensor(logits):
        raise TypeError(f"logits 必须是 torch.Tensor，实际是 {type(logits).__name__}")
    if logits.ndim != 2:
        raise ValueError(f"logits 必须是二维 [图片数, {NUM_CLASSES}]，当前为 {logits.ndim} 维")
    if logits.shape[1] != NUM_CLASSES:
        raise ValueError(f"logits 第二维必须是 {NUM_CLASSES} 类，当前为 {logits.shape[1]}")
    if logits.shape[0] == 0:
        raise ValueError("logits 不能为空，至少需要一张图片")
    if not torch.isfinite(logits).all():
        raise ValueError("logits 含 NaN 或 Inf，无法计算聚合概率")
    return logits.float()


def _validate_offsets(offsets: torch.Tensor, num_images: int) -> torch.Tensor:
    """校验 offsets：一维整数、首项 0、严格递增、末项等于图片数。"""
    if not torch.is_tensor(offsets):
        raise TypeError(f"offsets 必须是 torch.Tensor，实际是 {type(offsets).__name__}")
    if offsets.ndim != 1:
        raise ValueError(f"offsets 必须是一维，当前为 {offsets.ndim} 维")
    if offsets.dtype == torch.bool or offsets.is_floating_point():
        raise TypeError(f"offsets 必须是整数张量，当前 dtype 为 {offsets.dtype}")
    if offsets.numel() < 2:
        raise ValueError("offsets 至少要有两个元素（0 与图片数）")
    if int(offsets[0]) != 0:
        raise ValueError(f"offsets 首项必须是 0，当前为 {int(offsets[0])}")
    if int(offsets[-1]) != num_images:
        raise ValueError(f"offsets 末项必须等于图片数 {num_images}，当前为 {int(offsets[-1])}")
    if bool((offsets[1:] <= offsets[:-1]).any()):
        raise ValueError("offsets 必须严格递增，每个 study 至少一张图片")
    return offsets.to(torch.long)


def aggregate_pair(
    logits: torch.Tensor, offsets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """同一份逐图 logits 分别按 G0 / G1 聚合，返回两个 `[studies, 10]` fp32 张量。

    - G0：`sigmoid(aggregate_logits(logits))`，即现有推理路径的口径；
    - G1：`aggregate_logits(sigmoid(logits))`，先在概率空间逐图 sigmoid 再均值。

    两者只差「在哪个空间平均」：单图 study 必然相等，多图 study 可能改变排序。
    直接复用训练/推理同一个 `aggregate_logits`（无视图加权），避免第二套均值
    实现与线上口径分叉。
    """
    logits = _validate_logits(logits)
    offsets = _validate_offsets(offsets, logits.shape[0]).to(logits.device)

    g0 = torch.sigmoid(aggregate_logits(logits, offsets))
    g1 = aggregate_logits(torch.sigmoid(logits), offsets)
    return g0, g1


# ---------------------------------------------------------------------------
# 文件指纹与运行环境
# ---------------------------------------------------------------------------


def sha256_file(path: str | Path) -> str:
    """分块计算文件 SHA-256，避免把大权重整个读进内存。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode(CSV_ENCODING)).hexdigest()


def _repo_root() -> Path:
    """本文件位于 src/infer/，仓库根是上两级目录。"""
    return Path(__file__).resolve().parents[2]


def _git_info() -> dict[str, Any]:
    """记录代码指纹；git 不可用（例如打包环境）时返回 unknown 而不是崩溃。"""
    root = _repo_root()
    info: dict[str, Any] = {"commit": "unknown", "dirty": None, "status": []}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        return info
    info.update({"commit": commit, "dirty": bool(status), "status": status})
    return info


def _environment_info() -> dict[str, str]:
    """记录 Python / 关键库版本；README 里说明过版本差异会影响数值复现。"""
    info = {"python": platform.python_version(), "torch": torch.__version__}
    for name in ("timm", "numpy", "sklearn"):
        try:
            module = __import__(name)
            info[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            info[name] = "missing"
    return info


def _now() -> str:
    """本地时间（远程训练机为 Asia/Shanghai）带时区偏移，便于对齐实验记录。"""
    return datetime.now().astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# export：逐图 logits 导出与基准校验
# ---------------------------------------------------------------------------


def _load_run_config(run_dir: Path, checkpoint: dict[str, Any]) -> tuple[Config, str]:
    """优先用 best.pt 内保存的配置，旧权重缺配置时才回退 run 目录的 config.yaml。

    与 `run_compare.evaluate_run` 一致：续训可能写过新配置，必须按权重存档的
    配置重建数据与模型，否则预处理和骨干可能对不上。
    """
    raw_config = checkpoint.get("config")
    if raw_config is None:
        config_path = run_dir / "config.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"run 目录 {run_dir} 缺少 config.yaml，且 best.pt 里没有保存配置"
            )
        return load_config(config_path), "config.yaml（旧权重回退）"
    return config_from_dict(raw_config), "best.pt"


def _image_sha256(path: Path) -> str:
    return sha256_file(path)


def _check_output_dir(out_dir: Path) -> None:
    """目录已存在且非空时报错，避免静默覆盖上一次的缓存与报告。"""
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"输出目录 {out_dir} 已存在且非空；请换一个新目录，避免覆盖旧缓存"
        )


@torch.inference_mode()
def export_run(
    run_dir: str | Path,
    out_dir: str | Path,
    device: str = "auto",
) -> dict[str, Any]:
    """把固定 run 的 valid 逐图 logits 导出到 out_dir，并返回 metadata。

    流程：
    1. 读 best.pt 的配置（回退 config.yaml），只接受 `model.aggregation == mean`；
    2. 按 `build_loader(config, "valid", train=False)` 逐批前向，每批只调用一次
       模型，把 `model(images).float().cpu()` 攒成 `[N, 10]`；
    3. 用 batch 的 study_ids 与 dataset.records 对齐，再按 record 的图片路径展开
       图片身份（Dicom_id 取文件名去后缀）；
    4. 用 `predict_loader` 独立跑一次基准（额外一遍前向，不计入导出前向数），
       要求 G0 与现有路径的逐 study 概率最大绝对差 ≤ 1e-6；
    5. 写 image_logits.npz 与 metadata.json（含 NPZ SHA-256）。
    """
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    _check_output_dir(out_dir)
    checkpoint_path = run_dir / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"run 目录 {run_dir} 缺少 best.pt；export 只接受 run_train.py 的产物"
        )

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    config, config_source = _load_run_config(run_dir, checkpoint)
    if config.model.aggregation != "mean":
        raise ValueError(
            f"export 只接受 model.aggregation == 'mean' 的已训练模型，"
            f"当前为 {config.model.aggregation!r}；加权聚合不在本实验范围内"
        )

    torch_device = resolve_device(device)
    loader = build_loader(config, "valid", train=False, pin_memory=torch_device.type == "cuda")
    records = loader.dataset.records
    if not records:
        raise RuntimeError("valid 集为空，无法导出逐图 logits")

    model = load_checkpoint_model(config, checkpoint_path, torch_device)
    try:
        parameter_dtype = next(model.parameters()).dtype
    except StopIteration:  # 无参数的可测替身
        parameter_dtype = torch.float32
    if parameter_dtype != torch.float32:
        raise ValueError(
            f"导出要求 FP32 前向，但模型参数 dtype 为 {parameter_dtype}；"
            "请在导出前把模型转回 float32，不要用 AMP / fp16 权重"
        )

    logits_chunks: list[torch.Tensor] = []
    offsets: list[int] = [0]
    study_ids: list[str] = []
    subject_ids: list[str] = []
    labels: list[np.ndarray] = []
    views: list[int] = []
    dicom_ids: list[str] = []
    image_manifest: list[dict[str, Any]] = []
    cursor = 0
    num_model_forwards = 0

    for batch in loader:
        images = batch["images"].to(device=torch_device, non_blocking=True)
        chunk = model(images).float().cpu()
        num_model_forwards += 1

        batch_study_ids = list(batch["study_ids"])
        bounds = batch["offsets"]
        batch_studies = len(batch_study_ids)
        if batch_studies != int(bounds.numel()) - 1:
            raise ValueError("batch 的 study_ids 数量与 offsets 不一致")
        if chunk.shape[0] != int(bounds[-1]):
            raise ValueError(
                f"模型输出 {chunk.shape[0]} 行，但本批图片数为 {int(bounds[-1])}"
            )
        if chunk.shape[1] != NUM_CLASSES:
            raise ValueError(f"模型输出应为 {NUM_CLASSES} 类，当前为 {chunk.shape[1]}")
        if not torch.isfinite(chunk).all():
            raise ValueError("模型输出含 NaN 或 Inf，停止导出并排查权重 / 输入")

        expected = records[cursor : cursor + batch_studies]
        expected_ids = [record.study_id for record in expected]
        if expected_ids != batch_study_ids:
            raise ValueError(
                "batch 的 study_ids 与 dataset.records 顺序不一致："
                f"{batch_study_ids[:3]} != {expected_ids[:3]}"
            )

        for index, record in enumerate(expected):
            start = int(bounds[index])
            end = int(bounds[index + 1])
            if end - start != record.num_images:
                raise ValueError(
                    f"study {record.study_id} 的图片数不一致："
                    f"offsets 为 {end - start}，record 为 {record.num_images}"
                )
            study_views = record.views or (2,) * record.num_images
            if len(study_views) != record.num_images:
                raise ValueError(f"study {record.study_id} 的 views 长度与图片数不一致")
            if record.labels is None:
                raise ValueError(f"study {record.study_id} 没有标签，valid 集必须带标注")

            study_ids.append(record.study_id)
            subject_ids.append(record.subject_id)
            labels.append(np.asarray(record.labels))
            for image_index, image_path in enumerate(record.image_paths):
                image_path = Path(image_path)
                dicom_ids.append(image_path.stem)
                views.append(int(study_views[image_index]))
                image_manifest.append(
                    {
                        "dicom_id": image_path.stem,
                        "study_id": record.study_id,
                        "subject_id": record.subject_id,
                        "view": int(study_views[image_index]),
                        "path": str(image_path),
                        "sha256": _image_sha256(image_path),
                    }
                )
            offsets.append(offsets[-1] + record.num_images)

        logits_chunks.append(chunk)
        cursor += batch_studies

    if cursor != len(records):
        raise ValueError(f"只遍历了 {cursor} 个 study，但 valid 集有 {len(records)} 个")
    if not logits_chunks:
        raise RuntimeError("没有产生任何逐图 logits")

    logits_array = torch.cat(logits_chunks).numpy().astype(np.float32)
    offsets_array = np.asarray(offsets, dtype=np.int64)
    labels_array = np.stack([np.asarray(item, dtype=np.float64) for item in labels])
    if labels_array.shape != (len(study_ids), NUM_CLASSES):
        raise ValueError(f"标签形状应为 [{len(study_ids)}, {NUM_CLASSES}]")
    if not np.isin(labels_array, (0.0, 1.0)).all():
        raise ValueError("标签必须只含 0 / 1")
    labels_array = labels_array.astype(np.int64)

    if logits_array.shape[0] != offsets_array[-1]:
        raise ValueError("逐图 logits 行数与 offsets 末项不一致")
    if len(dicom_ids) != logits_array.shape[0]:
        raise ValueError("Dicom_id 数量与图片数不一致")

    # ---- 基准校验：本次导出的 G0 必须与现有 predict_loader 路径一致 ----
    g0, g1 = aggregate_pair(torch.from_numpy(logits_array), torch.from_numpy(offsets_array))
    g0_frame = pd.DataFrame(g0.numpy(), columns=PROBABILITY_COLUMNS)
    g0_frame.insert(0, "Study_id", study_ids)
    baseline = predict_loader(model, loader, device=torch_device, config=config)
    aligned_g0 = _align_frame(study_ids, g0_frame, "G0 导出结果")
    aligned_baseline = _align_frame(study_ids, baseline, "predict_loader 基准")
    difference = np.abs(
        aligned_g0[PROBABILITY_COLUMNS].to_numpy(dtype=np.float64)
        - aligned_baseline[PROBABILITY_COLUMNS].to_numpy(dtype=np.float64)
    )
    max_abs_diff = float(difference.max())
    mean_abs_diff = float(difference.mean())
    if max_abs_diff > 1e-6:
        raise RuntimeError(
            "G0 与 predict_loader 的独立结果最大绝对差 "
            f"{max_abs_diff:.3e} 超过 1e-6，本次实验无效；"
            "请先排查配置、预处理与运行环境，不要比较 G0 / G1"
        )

    y_true = labels_array.astype(np.float64)
    baseline_check = {
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "g0_macro_auc": float(macro_auc(y_true, g0.numpy())),
        "predict_loader_macro_auc": float(
            macro_auc(y_true, aligned_baseline[PROBABILITY_COLUMNS].to_numpy(np.float64))
        ),
        "predict_loader_model_forwards": len(loader),
    }

    # ---- 写缓存 ----
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / IMAGE_LOGITS_FILE
    np.savez(
        npz_path,
        logits=logits_array,
        offsets=offsets_array,
        labels=labels_array,
        study_ids=np.asarray(study_ids, dtype=np.str_),
        subject_ids=np.asarray(subject_ids, dtype=np.str_),
        dicom_ids=np.asarray(dicom_ids, dtype=np.str_),
        views=np.asarray(views, dtype=np.int64),
    )

    valid_csv = config.data.valid.csv
    metadata: dict[str, Any] = {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "created_at": _now(),
        "code": _git_info(),
        "environment": _environment_info(),
        "run": {"dir": str(run_dir), "config_source": config_source},
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "epoch": checkpoint.get("epoch"),
            "seed": checkpoint.get("seed", config.seed),
            "metrics": checkpoint.get("metrics"),
        },
        "config": config.to_dict(),
        "valid": {
            "csv": str(valid_csv),
            "csv_sha256": sha256_file(valid_csv) if valid_csv.is_file() else "missing",
            "studies": len(study_ids),
            "images": int(logits_array.shape[0]),
        },
        "class_ids": list(CLASS_IDS),
        "device": str(torch_device),
        "precision": REQUIRED_PRECISION,
        "num_batches": len(loader),
        "num_model_forwards": num_model_forwards,
        "baseline_check": baseline_check,
        "npz_sha256": sha256_file(npz_path),
        "images": image_manifest,
    }
    (out_dir / METADATA_FILE).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding=CSV_ENCODING
    )
    return metadata


# ---------------------------------------------------------------------------
# compare：只读缓存，离线重算 G0 / G1 并写报告
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CachedLogits:
    """校验通过的逐图缓存；字段顺序与 metadata 里记录的一致。"""

    logits: np.ndarray
    offsets: np.ndarray
    labels: np.ndarray
    study_ids: list[str]
    subject_ids: list[str]
    dicom_ids: list[str]
    views: np.ndarray
    metadata: dict[str, Any]


def _read_metadata(cache_dir: Path) -> dict[str, Any]:
    metadata_path = cache_dir / METADATA_FILE
    if not metadata_path.is_file():
        raise FileNotFoundError(f"缓存目录 {cache_dir} 缺少 {METADATA_FILE}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding=CSV_ENCODING))
    except json.JSONDecodeError as error:
        raise ValueError(f"{metadata_path} 不是合法 JSON：{error}") from error
    if not isinstance(metadata, dict):
        raise ValueError(f"{metadata_path} 顶层应该是对象")
    version = metadata.get("cache_format_version")
    if version != CACHE_FORMAT_VERSION:
        raise ValueError(
            f"缓存格式版本为 {version!r}，本代码只支持 {CACHE_FORMAT_VERSION}；"
            "请用当前代码重新 export"
        )
    return metadata


def _require_array(data: Any, name: str, ndim: int) -> np.ndarray:
    try:
        array = np.asarray(data[name])
    except KeyError as error:
        raise ValueError(f"缓存 {IMAGE_LOGITS_FILE} 缺少数组 {name!r}") from error
    if array.ndim != ndim:
        raise ValueError(f"缓存数组 {name} 应为 {ndim} 维，当前为 {array.ndim} 维")
    return array


def load_cache(cache_dir: str | Path, verify_hash: bool = True) -> CachedLogits:
    """读取并校验缓存：NPZ 哈希、格式版本、维度、ID 唯一性与标签取值。

    任何一项不符都直接报错——缓存被手改过或与 metadata 不配对时，继续算指标
    只会得到无法解释的数字。
    """
    cache_dir = Path(cache_dir)
    metadata = _read_metadata(cache_dir)
    npz_path = cache_dir / IMAGE_LOGITS_FILE
    if not npz_path.is_file():
        raise FileNotFoundError(f"缓存目录 {cache_dir} 缺少 {IMAGE_LOGITS_FILE}")

    if verify_hash:
        expected = metadata.get("npz_sha256")
        actual = sha256_file(npz_path)
        if expected is None or expected != actual:
            raise ValueError(
                f"{npz_path} 的 SHA-256 {actual} 与 metadata 记录 {expected} 不符；"
                "缓存可能被改动，请重新导出"
            )

    with np.load(npz_path, allow_pickle=False) as data:
        logits = _require_array(data, "logits", 2).astype(np.float64)
        offsets = _require_array(data, "offsets", 1).astype(np.int64)
        labels = _require_array(data, "labels", 2)
        study_ids = [str(item) for item in _require_array(data, "study_ids", 1).tolist()]
        subject_ids = [str(item) for item in _require_array(data, "subject_ids", 1).tolist()]
        dicom_ids = [str(item) for item in _require_array(data, "dicom_ids", 1).tolist()]
        views = _require_array(data, "views", 1).astype(np.int64)

    if logits.shape[1] != NUM_CLASSES:
        raise ValueError(f"logits 第二维应为 {NUM_CLASSES}，当前为 {logits.shape[1]}")
    if not np.isfinite(logits).all():
        raise ValueError("logits 含 NaN 或 Inf")
    if offsets.size < 2 or int(offsets[0]) != 0:
        raise ValueError("offsets 至少两个元素且首项为 0")
    if int(offsets[-1]) != logits.shape[0]:
        raise ValueError(f"offsets 末项应为图片数 {logits.shape[0]}，当前为 {int(offsets[-1])}")
    if bool((np.diff(offsets) <= 0).any()):
        raise ValueError("offsets 必须严格递增")
    studies = len(offsets) - 1
    if len(study_ids) != studies or len(subject_ids) != studies:
        raise ValueError("study_ids / subject_ids 数量与 offsets 不一致")
    if len(dicom_ids) != logits.shape[0] or len(views) != logits.shape[0]:
        raise ValueError("dicom_ids / views 数量与图片数不一致")
    if len(set(study_ids)) != studies:
        raise ValueError("study_ids 存在重复")
    if len(set(dicom_ids)) != len(dicom_ids):
        raise ValueError("dicom_ids 存在重复")
    if not set(views.tolist()) <= {0, 1, 2}:
        raise ValueError("views 只能取 0（正位）/ 1（侧位）/ 2（未知）")
    if labels.shape != (studies, NUM_CLASSES):
        raise ValueError(f"labels 形状应为 [{studies}, {NUM_CLASSES}]，当前为 {labels.shape}")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("labels 必须只含 0 / 1")
    if metadata.get("valid", {}).get("studies") not in (None, studies):
        raise ValueError("metadata 的 study 数与缓存不一致")
    if metadata.get("valid", {}).get("images") not in (None, logits.shape[0]):
        raise ValueError("metadata 的图片数与缓存不一致")

    return CachedLogits(
        logits=logits.astype(np.float32),
        offsets=offsets,
        labels=labels.astype(np.int64),
        study_ids=study_ids,
        subject_ids=subject_ids,
        dicom_ids=dicom_ids,
        views=views,
        metadata=metadata,
    )


def _align_frame(study_ids: list[str], frame: pd.DataFrame, what: str) -> pd.DataFrame:
    """按缓存里的 Study_id 顺序重排预测表；重复 / 缺失 / 额外 ID 一律报错。

    `Study_id` 在前导零上很敏感，统一按字符串比较。缓存顺序就是 metadata 里
    记录的加载顺序，重排后标签与预测才能逐行对上。
    """
    if "Study_id" not in frame.columns:
        raise ValueError(f"{what} 缺少 Study_id 列")
    ids = frame["Study_id"].astype(str).tolist()
    if len(set(ids)) != len(ids):
        raise ValueError(f"{what} 存在重复 Study_id")
    expected = set(study_ids)
    seen = set(ids)
    missing = expected - seen
    extra = seen - expected
    if missing or extra:
        raise ValueError(
            f"{what} 的 Study_id 与缓存不匹配：缺失 {len(missing)} 个、额外 {len(extra)} 个"
        )
    return frame.set_index("Study_id").loc[list(study_ids)].reset_index()


def _metrics_block(
    y_true: np.ndarray, y_score: np.ndarray, thresholds: list[float]
) -> dict[str, Any]:
    """一个子集的指标；空子集返回 NaN / 0 并标记 empty，不填 0 冒充结果。"""
    if y_true.shape[0] == 0:
        return {
            "status": "empty",
            "macro_auc": float("nan"),
            "valid_auc_classes": 0,
            "f1": {threshold: float("nan") for threshold in thresholds},
        }
    per_class = per_class_report(y_true, y_score)
    curve = macro_f1_curve(y_true, y_score, thresholds)
    return {
        "status": "ok",
        "macro_auc": float(macro_auc(y_true, y_score)),
        "valid_auc_classes": int(per_class["auc"].notna().sum()),
        "f1": {float(row.threshold): float(row.macro_f1) for row in curve.itertuples()},
    }


def _subset_summary(
    name: str,
    y_true: np.ndarray,
    g0: np.ndarray,
    g1: np.ndarray,
    thresholds: list[float],
) -> dict[str, Any]:
    """一个子集（全体 / 单图 / 多图）的 G0、G1、Δ 与概率差。"""
    g0_block = _metrics_block(y_true, g0, thresholds)
    g1_block = _metrics_block(y_true, g1, thresholds)
    if y_true.shape[0] == 0:
        max_abs = mean_abs = float("nan")
    else:
        difference = np.abs(g1 - g0)
        max_abs = float(difference.max())
        mean_abs = float(difference.mean())

    row: dict[str, Any] = {
        "subset": name,
        "status": "empty" if y_true.shape[0] == 0 else "ok",
        "studies": int(y_true.shape[0]),
        "g0_macro_auc": g0_block["macro_auc"],
        "g1_macro_auc": g1_block["macro_auc"],
        "delta_macro_auc": g1_block["macro_auc"] - g0_block["macro_auc"],
        "g0_valid_auc_classes": g0_block["valid_auc_classes"],
        "g1_valid_auc_classes": g1_block["valid_auc_classes"],
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
    }
    for threshold in thresholds:
        suffix = f"@{threshold:g}"
        row[f"g0_f1{suffix}"] = g0_block["f1"][threshold]
        row[f"g1_f1{suffix}"] = g1_block["f1"][threshold]
        row[f"delta_f1{suffix}"] = g1_block["f1"][threshold] - g0_block["f1"][threshold]
    return row


def _format_value(value: Any) -> str:
    """Markdown 单元格：NaN 写成「无结果」，浮点保留 4 位。"""
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    if isinstance(value, float) and not np.isfinite(value):
        return "无结果"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _to_markdown(table: pd.DataFrame) -> str:
    """手写 Markdown 表格，避免为了 to_markdown 引入 tabulate。"""
    columns = list(table.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(_format_value(value) for value in row) + " |")
    return "\n".join(lines)


def _summary_markdown(
    summary: pd.DataFrame,
    per_class: pd.DataFrame,
    metadata: dict[str, Any],
    thresholds: list[float],
    cache_dir: Path,
) -> str:
    """人读版报告：定义、数据指纹、汇总、逐类与判读提示。"""
    valid = metadata.get("valid", {})
    checkpoint = metadata.get("checkpoint", {})
    baseline = metadata.get("baseline_check", {})
    overall = summary[summary["subset"] == "all"].iloc[0]
    delta = float(overall["delta_macro_auc"])

    lines = [
        "# RUN-008 聚合空间对照（G0 / G1）",
        "",
        f"- 生成时间：{_now()}",
        f"- 缓存目录：{cache_dir}（格式版本 "
        f"{metadata.get('cache_format_version')}）",
        f"- 权重：`{checkpoint.get('path', '未知')}`（SHA-256 `{checkpoint.get('sha256', '未知')}`，"
        f"epoch {checkpoint.get('epoch')}，seed {checkpoint.get('seed')}）",
        f"- 验证集：{valid.get('studies', '未知')} studies / {valid.get('images', '未知')} 图"
        f"（CSV SHA-256 `{valid.get('csv_sha256', '未知')}`）",
        "- 定义：G0 = `sigmoid(mean(logits_i))`；G1 = `mean(sigmoid(logits_i))`",
        f"- 阈值网格：{', '.join(f'{value:g}' for value in thresholds)}",
        "- 判读口径：主要看全体验证集的 G1−G0 macro-AUC；单图 study 两者应完全一致。",
        "",
        "## 汇总",
        "",
        _to_markdown(summary),
        "",
        "## 逐类指标（全体验证集）",
        "",
        "AUC 对退化类记 NaN，不参与平均；support 是该类正样本数。",
        "",
        _to_markdown(per_class),
        "",
        "## 基准一致性",
        "",
        f"- 本次导出的 G0 与 predict_loader 逐 study 概率最大绝对差："
        f"{_format_value(baseline.get('max_abs_diff'))}",
        f"- 平均绝对差：{_format_value(baseline.get('mean_abs_diff'))}",
        f"- G0 macro-AUC（缓存重算）：{_format_value(baseline.get('g0_macro_auc'))}；"
        f"predict_loader 独立结果：{_format_value(baseline.get('predict_loader_macro_auc'))}",
        "",
        "## 判读提示",
        "",
    ]
    if np.isfinite(delta) and delta >= 0.005:
        lines.append(
            f"- 全体 G1−G0 = {delta:+.4f}，达到预定的 +0.005 实用门槛；"
            "可在相同代码与精度下继续 RUN-008-A（seed 2026）复核。"
        )
    elif np.isfinite(delta):
        lines.append(
            f"- 全体 G1−G0 = {delta:+.4f}，未达 +0.005 门槛。"
            "按计划停止本项，默认保持 G0；正的小幅差异只记为趋势，不据此切换聚合。"
        )
    else:
        lines.append("- 全体验证集 AUC 无定义或为空，本次结果不可用于判读。")
    lines.extend(
        [
            "- +0.005 是预先固定的项目实用门槛，不是统计显著性检验。",
            "- compare 只校验缓存自身未被改动，不证明源图片之后没有变化。",
            "",
        ]
    )
    return "\n".join(lines)


def compare_cache(
    cache_dir: str | Path, thresholds: list[float]
) -> dict[str, Any]:
    """只读缓存，离线重算 G0 / G1，写出预测表、summary 与逐类报告。

    不加载模型、不读图片：模型前向只发生在 export。阈值只影响 F1，AUC 与
    阈值无关；主要判读始终基于全体验证集。
    """
    cache_dir = Path(cache_dir)
    cached = load_cache(cache_dir)
    metadata = cached.metadata

    g0_tensor, g1_tensor = aggregate_pair(
        torch.from_numpy(cached.logits), torch.from_numpy(cached.offsets)
    )
    g0 = g0_tensor.numpy().astype(np.float64)
    g1 = g1_tensor.numpy().astype(np.float64)

    study_ids = cached.study_ids
    subject_ids = cached.subject_ids
    study_labels = cached.labels.astype(np.float64)

    # 先在 Study_id 上对齐标签和两种预测；任何 ID 不一致都在这里停下。
    labels_frame = pd.DataFrame(study_labels, columns=LABEL_COLUMNS)
    labels_frame.insert(0, "Study_id", study_ids)
    aligned_labels = _align_frame(study_ids, labels_frame, "缓存标签")

    g0_frame = pd.DataFrame(g0, columns=PROBABILITY_COLUMNS)
    g0_frame.insert(0, "Study_id", study_ids)
    g1_frame = pd.DataFrame(g1, columns=PROBABILITY_COLUMNS)
    g1_frame.insert(0, "Study_id", study_ids)
    aligned_g0 = _align_frame(study_ids, g0_frame, "G0 预测")
    aligned_g1 = _align_frame(study_ids, g1_frame, "G1 预测")

    y_true = aligned_labels[LABEL_COLUMNS].to_numpy(dtype=np.float64)
    g0_aligned = aligned_g0[PROBABILITY_COLUMNS].to_numpy(dtype=np.float64)
    g1_aligned = aligned_g1[PROBABILITY_COLUMNS].to_numpy(dtype=np.float64)

    # 子集：按 study 的图片数区分单图 / 多图；空子集标 empty 而不是填 0。
    counts = np.diff(cached.offsets)
    subsets = {
        "all": np.ones(len(counts), dtype=bool),
        "single": counts == 1,
        "multi": counts > 1,
    }
    summary_rows = [
        _subset_summary(name, y_true[mask], g0_aligned[mask], g1_aligned[mask], thresholds)
        for name, mask in subsets.items()
    ]
    summary = pd.DataFrame(summary_rows)

    # 逐类：只报 support、两种 AUC 与差值；退化类 AUC 为 NaN。
    g0_report = per_class_report(y_true, g0_aligned)
    g1_report = per_class_report(y_true, g1_aligned)
    per_class = pd.DataFrame(
        {
            "class_id": g0_report["class_id"].astype(int),
            "support": g0_report["support"].astype(int),
            "g0_auc": g0_report["auc"],
            "g1_auc": g1_report["auc"],
        }
    )
    per_class["delta_auc"] = per_class["g1_auc"] - per_class["g0_auc"]

    # 预测表：Subject_id, Study_id, p1..p10，一 study 一行；顺序与缓存一致。
    def _write_predictions(path: Path, frame: pd.DataFrame) -> None:
        output = frame.copy()
        output.insert(0, "Subject_id", subject_ids)
        output.to_csv(path, index=False, encoding=CSV_ENCODING)

    _write_predictions(cache_dir / PREDICTIONS_G0_FILE, aligned_g0)
    _write_predictions(cache_dir / PREDICTIONS_G1_FILE, aligned_g1)
    summary.to_csv(cache_dir / SUMMARY_CSV_FILE, index=False, encoding=CSV_ENCODING)
    per_class.to_csv(cache_dir / PER_CLASS_FILE, index=False, encoding=CSV_ENCODING)
    markdown = _summary_markdown(summary, per_class, metadata, thresholds, cache_dir)
    (cache_dir / SUMMARY_MD_FILE).write_text(markdown, encoding=CSV_ENCODING)

    return {
        "summary": summary,
        "per_class": per_class,
        "markdown": markdown,
        "g0": g0,
        "g1": g1,
        "cache": cached,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="固定 checkpoint 的聚合空间对照：导出逐图 logits，比较 G0 / G1"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser(
        "export", help="导出 valid 逐图 logits 缓存（每个 study 只做一次前向）"
    )
    export_parser.add_argument("--run", required=True, help="训练 run 目录（含 best.pt）")
    export_parser.add_argument("--device", default="auto", help="推理设备，默认 auto")
    export_parser.add_argument("--out", required=True, help="缓存输出目录（必须为空或不存在）")

    compare_parser = subparsers.add_parser(
        "compare", help="只读缓存，离线重算 G0 / G1 并写报告"
    )
    compare_parser.add_argument("--cache", required=True, help="export 写出的缓存目录")
    compare_parser.add_argument(
        "--thresholds",
        default="0.2,0.3,0.4,0.5,0.6,0.7",
        help="逗号分隔的阈值网格，用来算各阈值 F1（默认 0.2,0.3,0.4,0.5,0.6,0.7）",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "export":
        metadata = export_run(args.run, args.out, device=args.device)
        baseline = metadata["baseline_check"]
        print(
            f"已导出 {metadata['valid']['studies']} studies / "
            f"{metadata['valid']['images']} 图到 {args.out}"
        )
        print(
            f"G0 与 predict_loader 最大绝对差 {baseline['max_abs_diff']:.3e}；"
            f"G0 macro-AUC {baseline['g0_macro_auc']:.6f}"
        )
        return

    thresholds = parse_thresholds(args.thresholds)
    result = compare_cache(args.cache, thresholds)
    summary = result["summary"]
    overall = summary[summary["subset"] == "all"].iloc[0]
    print(summary.to_string(index=False))
    print()
    print(
        f"G1−G0 全体 macro-AUC：{overall['delta_macro_auc']:+.6f}"
        if np.isfinite(overall["delta_macro_auc"])
        else "G1−G0 全体 macro-AUC：无结果"
    )
    for name in (SUMMARY_CSV_FILE, SUMMARY_MD_FILE, PER_CLASS_FILE):
        print(f"已写出：{Path(args.cache) / name}")


if __name__ == "__main__":
    main()
