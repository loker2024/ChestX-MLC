"""推理：把一个 DataLoader 里的 study 跑成「一行一个 study」的概率宽表。

推理和验证共用同一套口径：每张图各过一次网络得到 [N, 10] logits，
再用 offsets 把同一 study 的多张图 logits 求均值，最后 sigmoid 成概率。
这样训练时看到的分数、验证时算的指标、提交时写的概率完全一致。

这里只产出宽表（Study_id + p1..p10），不直接写 submission.csv：
宽表适合算指标、调阈值、排查单个 study；真正的长表提交格式由
infer.submission 负责转换。
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from common.checkpoint import load_checkpoint
from common.config import Config
from common.constants import CLASS_IDS
from model.network import aggregate_logits, build_model


def load_checkpoint_model(
    config: Config, checkpoint_path: str | Path, device: torch.device
) -> nn.Module:
    """按配置建模型、装入 checkpoint 权重，返回已经放到 device 上的 eval 模型。

    建模型时强制 pretrained=False：权重马上会被 checkpoint 覆盖，没必要再下载
    一遍 ImageNet 预训练权重（离线环境也下载不了）。模型结构仍由 config 决定，
    所以推理用的配置文件必须和训练时一致（backbone、类别数）。
    """
    # 只替换 pretrained 字段，backbone 和 num_classes 保持配置里的值。
    model_config = replace(config.model, pretrained=False)
    model = build_model(replace(config, model=model_config))

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


@torch.inference_mode()
def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
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
        images = batch["images"].to(device, non_blocking=True)
        offsets = batch["offsets"].to(device, non_blocking=True)
        study_logits = aggregate_logits(model(images), offsets)

        study_ids.extend(batch["study_ids"])
        # 先转 float32 再搬 CPU：以后即使模型用半精度推理，概率也按 fp32 落表。
        probability_batches.append(torch.sigmoid(study_logits.float()).cpu())

    if not probability_batches:
        raise RuntimeError("推理集为空，无法生成预测")
    probabilities = torch.cat(probability_batches).numpy()
    frame = pd.DataFrame(probabilities, columns=[f"p{class_id}" for class_id in CLASS_IDS])
    frame.insert(0, "Study_id", study_ids)
    return frame
