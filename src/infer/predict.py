"""推理：把一个 DataLoader 里的 study 跑成「一行一个 study」的概率宽表。

推理和验证共用同一套口径：每张图各过一次网络得到 [N, 10] logits，
再用 offsets 把同一 study 的多张图 logits 求均值，最后 sigmoid 成概率。
这样训练时看到的分数、验证时算的指标、提交时写的概率完全一致。

这里只产出宽表（Study_id + p1..p10），不直接写 submission.csv：
宽表适合算指标、调阈值、排查单个 study；真正的长表提交格式由
infer.submission 负责转换。

本模块同时提供 `predict_image` / `predict_study` 两个「不经 DataLoader」的
推理入口，供 `src/cost.py` 计时和一致性校验复用。三者（predict_loader、
predict_image、predict_study）都经 `_forward_probabilities` 做前向、聚合与
sigmoid，保证提交概率、计时口径、校验口径不分叉。
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from common.checkpoint import load_checkpoint
from common.config import Config, config_from_dict
from common.constants import CLASS_IDS
from data.dataset import load_image
from data.transforms import build_eval_transform
from model.network import aggregate_logits, build_model


def load_checkpoint_model(
    config: Config, checkpoint_path: str | Path, device: torch.device
) -> nn.Module:
    """按配置建模型、装入 checkpoint 权重，返回已经放到 device 上的 eval 模型。

    建模型时强制 pretrained=False：权重马上会被 checkpoint 覆盖，没必要再下载
    一遍 ImageNet 预训练权重（离线环境也下载不了）。模型结构仍由 config 决定，
    所以推理用的配置文件必须和训练时一致（backbone、类别数）。
    """
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    if checkpoint.get("config") is not None:
        saved = config_from_dict(checkpoint["config"])
        fields = {"image": ("size", "letterbox", "normalize", "in_channels", "xrv_maxval"),
                  "model": ("arch", "backbone", "num_classes", "dropout", "aggregation", "frontal_weight")}
        differences = [f"{section}.{field}" for section, names in fields.items()
                       for field in names if getattr(getattr(config, section), field) != getattr(getattr(saved, section), field)]
        if differences:
            raise ValueError("推理配置与 checkpoint 不一致：" + ", ".join(differences))
    else:
        import warnings
        warnings.warn("旧 checkpoint 缺少存档配置，无法验证推理口径", RuntimeWarning)
    # 只替换 pretrained 字段，backbone 和 num_classes 保持配置里的值。
    model_config = replace(config.model, pretrained=False)
    model = build_model(replace(config, model=model_config))

    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


def validate_submission_model(config: Config, checkpoint_path: str | Path) -> None:
    """禁止将研究专用的 TxRV 模型用于提交。"""
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    saved = config_from_dict(checkpoint["config"]) if checkpoint.get("config") else config
    if config.model.arch == "xrv_densenet121" or saved.model.arch == "xrv_densenet121":
        raise ValueError("TxRV 模型仅用于研究，不能生成赛题提交或参与提交融合")


def build_eval_inference_transform(
    config: Config,
) -> Callable[[Image.Image], torch.Tensor]:
    """返回与 `build_loader(config, split, train=False)` 完全相同的推理预处理。

    cost.py 不走 DataLoader（计时要求 batch=1 串行、且要控制每张图的边界），
    但预处理必须和训练/提交时一致，所以这里复用同一个 build_eval_transform。
    """
    return build_eval_transform(
        config.image.size,
        config.image.letterbox,
        config.image.normalize,
        config.image.in_channels,
        config.image.xrv_maxval,
    )


def _model_dtype(model: nn.Module) -> torch.dtype:
    """模型参数的 dtype；fp16 推理时把输入也转成半精度。"""
    try:
        return next(model.parameters()).dtype
    except StopIteration:  # 没有参数的可测替身（比如全用缓冲区的假模型）
        return torch.float32


def _forward_probabilities(
    model: nn.Module,
    images: torch.Tensor,
    offsets: torch.Tensor,
    device: torch.device,
    views: torch.Tensor | None = None,
    frontal_weight: float = 0.7,
) -> torch.Tensor:
    """CPU 上的 images/offsets -> CPU 上的 `[B, 10]` fp32 概率。

    这是全仓库唯一的「前向 + study 聚合 + sigmoid」实现：
    搬设备、按模型 dtype 转换输入、aggregate_logits 求 study 均值、sigmoid 后
    再转回 fp32 CPU 张量。predict_loader / predict_image / predict_study 都走它。
    """
    return torch.sigmoid(
        _forward_logits(model, images, offsets, device, views, frontal_weight)
    )


def _forward_logits(
    model: nn.Module,
    images: torch.Tensor,
    offsets: torch.Tensor,
    device: torch.device,
    views: torch.Tensor | None = None,
    frontal_weight: float = 0.7,
) -> torch.Tensor:
    """CPU 上的 images/offsets -> CPU 上的 `[B, 10]` fp32 study logits."""
    model_images = images.to(device=device, dtype=_model_dtype(model), non_blocking=True)
    model_offsets = offsets.to(device=device, non_blocking=True)
    study_logits = aggregate_logits(model(model_images), model_offsets,
                                    views.to(device) if views is not None else None, frontal_weight)
    # 先转 float32 再搬 CPU：以后即使模型用半精度推理，融合也使用稳定的 fp32 logits。
    return study_logits.float().cpu()


@torch.inference_mode()
def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: Config | None = None,
) -> pd.DataFrame:
    """对一个 DataLoader 做推理，返回 `Study_id, p1..p10` 的宽表。

    每个 batch 的图片按 data.dataset.collate_studies 的约定首尾相接，
    offsets 给出每个 study 的图片区间；同一 study 多张图的 logits 求均值后
    sigmoid，得到这次检查的 10 个概率。行顺序与 DataLoader 顺序一致
    （test loader 不打乱，也就是 CSV 里 study 第一次出现的顺序）。

    概率列名沿用赛题的类别 ID：p1 对应类别 1，p10 对应类别 10。
    """
    model.eval()
    study_ids: list[str] = []
    probability_batches: list[torch.Tensor] = []

    for batch in loader:
        probabilities = _forward_probabilities(
            model, batch["images"], batch["offsets"], device,
            batch["views"] if config is not None and config.model.aggregation == "view_weighted" else None,
            config.model.frontal_weight if config is not None else 0.7,
        )
        study_ids.extend(batch["study_ids"])
        probability_batches.append(probabilities)

    if not probability_batches:
        raise RuntimeError("推理集为空，无法生成预测")
    probabilities = torch.cat(probability_batches).numpy()
    frame = pd.DataFrame(probabilities, columns=[f"p{class_id}" for class_id in CLASS_IDS])
    frame.insert(0, "Study_id", study_ids)
    return frame


@torch.inference_mode()
def predict_loader_logits(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: Config | None = None,
) -> pd.DataFrame:
    """对 DataLoader 做 study-level 推理，返回 `Study_id, z1..z10` logits 宽表。"""
    model.eval()
    study_ids: list[str] = []
    logit_batches: list[torch.Tensor] = []

    for batch in loader:
        logits = _forward_logits(
            model,
            batch["images"],
            batch["offsets"],
            device,
            batch["views"] if config is not None and config.model.aggregation == "view_weighted" else None,
            config.model.frontal_weight if config is not None else 0.7,
        )
        study_ids.extend(batch["study_ids"])
        logit_batches.append(logits)

    if not logit_batches:
        raise RuntimeError("推理集为空，无法生成 logits")
    logits = torch.cat(logit_batches).numpy()
    frame = pd.DataFrame(logits, columns=[f"z{class_id}" for class_id in CLASS_IDS])
    frame.insert(0, "Study_id", study_ids)
    return frame


@torch.inference_mode()
def predict_image(
    model: nn.Module,
    image_path: str | Path,
    transform: Callable[[Image.Image], torch.Tensor],
    device: torch.device,
) -> np.ndarray:
    """单张图片的 10 类概率（batch=1，`[10]` fp32 numpy）。

    读图 + 预处理 + 前向 + sigmoid 全在这里，cost.py 用它做端到端计时：
    计时区间包住这次调用的全部工作，测的就是「读文件到算出结果」的墙钟时间。
    单图 forward 与 DataLoader 里多图 forward 在 eval 模式下逐图独立，结果一致。
    """
    model.eval()
    # convert_rgb=False：灰度图保留 L 模式，由 Letterbox 在 resize 之后转 RGB，
    # 结果与先转 RGB 再 resize 逐像素一致，但单通道插值只需 1/3 的运算量。
    tensor = transform(load_image(Path(image_path), convert_rgb=False)).unsqueeze(0)
    offsets = torch.tensor([0, 1], dtype=torch.long)
    return _forward_probabilities(model, tensor, offsets, device)[0].numpy()


@torch.inference_mode()
def predict_study(
    model: nn.Module,
    image_paths: Sequence[str | Path],
    transform: Callable[[Image.Image], torch.Tensor],
    device: torch.device,
    views: Sequence[int] | None = None,
    frontal_weight: float = 0.7,
) -> np.ndarray:
    """一次检查（study）的 10 类概率：该 study 所有图片 logits 求均值后 sigmoid。

    与 `predict_loader` 的聚合口径完全一致（同样的 aggregate_logits + sigmoid），
    用于 cost.py 把 `pictures_100` 里的图片按 study 分组后与 submission.csv 比对。
    `pictures_100` 里一个 study 只有部分图片时，这里得到的是「这批图」的均值，
    与提交时「全部图片」的均值会有差异——cost.py 的日志和文档都显式说明这一点。
    """
    paths = [Path(path) for path in image_paths]
    if not paths:
        raise ValueError("一个 study 至少要有一张图片才能推理")
    model.eval()
    images = torch.stack(
        # 与 predict_image 一致：灰度图保留 L 模式、在 resize 后再转 RGB。
        [transform(load_image(path, convert_rgb=False)) for path in paths]
    )
    offsets = torch.tensor([0, len(paths)], dtype=torch.long)
    return _forward_probabilities(model, images, offsets, device,
                                  torch.tensor(views) if views is not None else None, frontal_weight)[0].numpy()
