"""导出三模型概率融合的 train+valid 全量教师软标签。

教师：外部工作簿 SW-02（torchvision swin_v2_t @384）+ D-010-a（ConvNeXt-Tiny
@320）+ D-010-d（EfficientNetV2-S @320）。三者在官方 train、valid 合并后的
全部 study 上各自推理（D-010 两位教师直接复用其 full_train 配置的全量 loader，
SW-02 用 exp_S001a 的数据口径构造全量 loader），按 Study_id 对齐后逐类做等权
概率平均，写成离线蒸馏缓存，供 full_train 学生训练消费。

与 `run_distill_teacher.py`（A-FUSE-5 固定 4:2:3 裁剪 logit 融合）的区别：
本脚本面向 train+valid 全量、等权概率融合，教师可以包含仓库外的权重；融合
口径与 A-FUSE-7 的提交融合（probability, weights 1:1）保持一致。

用法（在仓库根目录下执行）：

    python src/run_distill_teacher_fuse.py \
        --repo-checkpoint outputs/20261107-122517_D-010-a/final_ema.pt \
                          outputs/20261107-130808_D-010-d/final_ema.pt \
        --swin-checkpoint outputs/external_SW-02_ASL_EMA/best_ema.pth \
        --swin-config configs/exp_S001a.yaml \
        --out outputs/distill/FUSE3_teacher
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from common.checkpoint import load_checkpoint
from common.config import Config, config_from_dict, load_config
from common.constants import CLASS_IDS
from common.utils import resolve_device
from data.dataset import StudyDataset, build_loader, collate_studies
from data.resplit import _records
from data.transforms import build_eval_transform
from infer.ensemble import fuse_probability_vectors
from infer.predict import load_checkpoint_model, predict_loader
from train.distillation import records_digest, sha256_file, write_teacher_cache


def build_sw02_model(path: Path, num_classes: int) -> nn.Module:
    """按 torchvision 原生结构严格加载 SW-02 的 EMA 权重。"""
    import torchvision

    model = torchvision.models.swin_v2_t(weights=None, num_classes=num_classes)
    # SW-02 的 head 是 Sequential(Dropout(0.1), Linear(768, 10))，state_dict 里是 head.1.*。
    model.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(768, num_classes))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state_dict"], strict=True)
    print(f"SW-02 权重: {path} | arch={payload.get('architecture')} epoch={payload.get('epoch')} variant={payload.get('variant')}")
    return model


def build_sw02_loader(config: Config) -> DataLoader:
    """用 exp_S001a 的图像口径构造 train+valid 全量 loader（不打乱）。

    这里不走 build_loader 的 full_train 分支，因为 exp_S001a 是全量训练之前的
    普通配置（full_train=false）；但 `data.resplit._records` 只读两份 CSV，
    与 full_train 分支返回的记录内容和顺序完全一致。
    """
    records, _ = _records(config)
    transform = build_eval_transform(
        config.image.size,
        config.image.letterbox,
        config.image.normalize,
        config.image.in_channels,
        config.image.xrv_maxval,
    )
    dataset = StudyDataset(records, transform, preserve_mode=config.image.normalize == "xrv")
    return DataLoader(
        dataset,
        batch_size=config.eval.batch_studies,
        shuffle=False,
        num_workers=config.train.num_workers,
        collate_fn=collate_studies,
        pin_memory=False,
    )


@torch.inference_mode()
def predict_study_mean(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    """逐图 sigmoid 后按 study 求均值（SW-02 存档的 G0 聚合口径）。"""
    model.eval()
    study_ids: list[str] = []
    chunks: list[np.ndarray] = []
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        logits = model(images)
        offsets = batch["offsets"]
        probabilities = torch.sigmoid(logits.float())
        counts = len(offsets) - 1
        means = torch.stack(
            [probabilities[offsets[i]: offsets[i + 1]].mean(dim=0) for i in range(counts)]
        )
        study_ids.extend(batch["study_ids"])
        chunks.append(means.cpu().numpy())
    frame = pd.DataFrame(np.concatenate(chunks), columns=[f"p{i}" for i in CLASS_IDS])
    frame.insert(0, "Study_id", study_ids)
    return frame


def fuse_frames(frames: list[pd.DataFrame], ids: list[str], weights: list[float] | None) -> pd.DataFrame:
    """按基准 Study_id 顺序对齐后，逐类做加权概率平均。"""
    columns = [f"p{i}" for i in CLASS_IDS]
    aligned: list[np.ndarray] = []
    for frame in frames:
        if frame["Study_id"].duplicated().any() or set(frame["Study_id"]) != set(ids):
            raise ValueError("教师输出的 Study_id 集合不一致")
        aligned.append(frame.set_index("Study_id").loc[ids, columns].to_numpy())
    values = [
        fuse_probability_vectors([array[index] for array in aligned], weights, "probability")
        for index in range(len(ids))
    ]
    result = pd.DataFrame(np.asarray(values), columns=columns)
    result.insert(0, "Study_id", ids)
    return result


def _git_head() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="导出三模型概率融合的全量教师软标签")
    parser.add_argument(
        "--repo-checkpoint",
        nargs="+",
        required=True,
        metavar="CHECKPOINT",
        help="仓库格式教师权重（按顺序给出，需内嵌 full_train 配置）",
    )
    parser.add_argument("--swin-checkpoint", required=True, help="外部 SW-02 教师权重")
    parser.add_argument("--swin-config", default="configs/exp_S001a.yaml", help="SW-02 数据与预处理口径配置")
    parser.add_argument("--out", required=True, help="新建缓存目录")
    parser.add_argument(
        "--weights",
        default=None,
        help="教师权重，按 repo 教师顺序、再 SW-02 的顺序给，逗号分隔；默认等权",
    )
    args = parser.parse_args()

    device = resolve_device("auto")
    repo_paths = [Path(value) for value in args.repo_checkpoint]
    swin_path = Path(args.swin_checkpoint)
    checkpoints = [load_checkpoint(path, map_location="cpu") for path in repo_paths]
    if any(not payload.get("config") for payload in checkpoints):
        raise ValueError("仓库教师 checkpoint 必须内嵌配置")
    configs = [config_from_dict(payload["config"]) for payload in checkpoints]
    if any(not config.data.full_train for config in configs):
        raise ValueError("仓库教师必须是 full_train 配置，否则 train loader 只覆盖 18,513 个 study")

    train_loader = build_loader(configs[0], "train", train=False, pin_memory=device.type == "cuda")
    base_records = train_loader.dataset.records
    index_sha256 = records_digest(base_records)
    base_ids = [record.study_id for record in base_records]
    print(f"全量教师推理 study 数: {len(base_ids)} | 基准 index_sha256: {index_sha256[:16]}…")

    frames: list[pd.DataFrame] = []
    kinds: list[dict[str, str]] = []
    for path, payload, config in zip(repo_paths, checkpoints, configs):
        started = time.time()
        loader = build_loader(config, "train", train=False, pin_memory=device.type == "cuda")
        if records_digest(loader.dataset.records) != index_sha256:
            raise ValueError(f"仓库教师数据索引与基准不一致: {path}")
        model = load_checkpoint_model(config, path, device)
        frames.append(predict_loader(model, loader, device, config))
        del model
        print(f"教师 {path.parent.name}/{path.name} 完成，{time.time() - started:.1f}s")
        kinds.append({"kind": "repo", "path": str(path.resolve()), "sha256": sha256_file(path)})

    started = time.time()
    swin_config = load_config(args.swin_config)
    swin_loader = build_sw02_loader(swin_config)
    if records_digest(swin_loader.dataset.records) != index_sha256:
        raise ValueError("SW-02 全量数据索引与基准不一致")
    swin_model = build_sw02_model(swin_path, swin_config.model.num_classes).to(device)
    frames.append(predict_study_mean(swin_model, swin_loader, device))
    del swin_model
    print(f"教师 SW-02 完成，{time.time() - started:.1f}s")
    kinds.append({"kind": "swin", "path": str(swin_path.resolve()), "sha256": sha256_file(swin_path)})

    if args.weights:
        weights = [float(value) for value in args.weights.split(",")]
        if len(weights) != len(frames):
            raise ValueError(f"权重数量 {len(weights)} 与教师数量 {len(frames)} 不一致")
    else:
        weights = None
    result = fuse_frames(frames, base_ids, weights)
    if len(result) != len(base_ids) or result["Study_id"].duplicated().any():
        raise ValueError("融合结果 Study_id 数量或唯一性错误")

    manifest = {
        "version": 1,
        "split": "train_full",
        "fusion": "probability_mean",
        "weights": weights if weights is not None else [1.0] * len(frames),
        "classes": list(CLASS_IDS),
        "train_csv_sha256": sha256_file(configs[0].data.train.csv),
        "index_sha256": index_sha256,
        "study_count": len(base_ids),
        "checkpoints": kinds,
        "git_head": _git_head(),
    }
    write_teacher_cache(Path(args.out), result, manifest)
    print(f"教师缓存已生成: {args.out} | study 数: {len(result)} | 融合: probability_mean | 权重: {manifest['weights']}")


if __name__ == "__main__":
    main()
