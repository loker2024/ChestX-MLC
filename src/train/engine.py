"""训练引擎：单 epoch 的训练 / 验证循环、学习率 warmup、早停与最优权重挑选。

完整的一次训练由 `src/run_train.py` 编排（读配置、建 loader、建模型），
本模块只负责“给定模型和数据，把它训练好”这件事：

    for epoch in range(epochs):
        训练一个 epoch（前向 + 反向 + 更新参数）
        在 valid 上算 loss / macro-AUC / macro-F1
        按 macro-AUC 决定要不要“记住”这一轮的权重
        连续若干轮都没有变好，就提前结束

几个容易踩坑的地方（下面的注释还会再提）：
- 模型吃的是“单张图”，但标签和 AUC 是“按检查（study）”算的，所以先把同一
  study 的多张图 logits 用 `aggregate_logits` 平均成一个向量，再算 loss 和指标。
- 有些类别在 valid 里全阴或全阳，AUC 没有意义会返回 NaN；这一轮就当作“没有
  变好”，绝不能让 NaN 抢走最优权重。
- 混合精度（AMP）只在 CUDA 上开启。GradScaler 在非 CUDA 设备上会被禁用，
  之后所有 scale / step / update 调用都等价于普通的优化器操作。
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from common.config import Config
from common.metrics import macro_auc, macro_f1
from model.losses import build_loss
from model.network import aggregate_logits


def _amp_enabled(config: Config, device: torch.device) -> bool:
    """项目只在 CUDA 上开混合精度：CPU / MPS 上没收益，还容易遇到算子不支持。"""
    return config.train.amp and device.type == "cuda"


def _build_scheduler(optimizer: Optimizer, warmup_epochs: int) -> LambdaLR:
    """线性 warmup：前 warmup_epochs 轮把学习率从 lr/warmup 线性涨到 lr，之后不变。

    LambdaLR 的 lr_lambda 收到的是“已经走过的步数”。这里每个 epoch 只调用一次
    scheduler.step()，所以“步数”就是 epoch 序号（从 0 开始）：
    warmup_epochs=2 时，第 1 轮 lr*0.5、第 2 轮 lr*1.0，之后一直是 lr。
    """
    if warmup_epochs <= 0:
        return LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    return LambdaLR(optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / warmup_epochs))


def _train_labels(train_loader: DataLoader) -> np.ndarray:
    """把训练集里所有 study 的标签堆成 [study 数, 类别数]，给 pos_weight 用。"""
    return np.stack([record.labels for record in train_loader.dataset.records]).astype(np.float32)


def _snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    """把当前权重在 CPU 上拷一份。训练中不带在 GPU 上，省显存也避免互相影响。"""
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def train_one_epoch(
    config: Config,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> float:
    """训练一个 epoch，返回按 study 数加权的平均 loss。"""
    model.train()
    use_amp = scaler.is_enabled()  # 只有 CUDA 上才会是 True
    total_loss = 0.0
    total_studies = 0

    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        offsets = batch["offsets"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            # 每张图各出一个 10 维 logits，再按 study 平均 —— 和计算指标的口径一致。
            study_logits = aggregate_logits(model(images), offsets)
            loss = criterion(study_logits, labels)

        scaler.scale(loss).backward()
        if config.train.grad_clip > 0:
            # 裁剪必须放在“反缩放”之后、optimizer.step() 之前，
            # 否则裁的是放大过的梯度，阈值 5.0 就失去意义了。
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        num_studies = len(batch["study_ids"])
        total_loss += loss.item() * num_studies
        total_studies += num_studies

    return total_loss / total_studies


@torch.inference_mode()
def evaluate(
    config: Config,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """在带标签的数据集（也是 valid）上跑一遍，返回 loss 与按 study 计算的指标。"""
    model.eval()
    use_amp = _amp_enabled(config, device)
    total_loss = 0.0
    total_studies = 0
    logits_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []

    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        offsets = batch["offsets"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            study_logits = aggregate_logits(model(images), offsets)

        num_studies = len(batch["study_ids"])
        total_loss += criterion(study_logits, labels).item() * num_studies
        total_studies += num_studies

        # 先转 float32 再搬 CPU：混合精度下 logits 可能是 fp16，直接 sigmoid 精度不够。
        logits_list.append(study_logits.float().cpu())
        labels_list.append(labels.float().cpu())

    y_score = torch.sigmoid(torch.cat(logits_list)).numpy()
    y_true = torch.cat(labels_list).numpy()
    return {
        "loss": total_loss / total_studies,
        "macro_auc": macro_auc(y_true, y_score),
        "macro_f1": macro_f1(y_true, y_score, config.eval.threshold),
    }


def fit(
    config: Config,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    """完整训练流程：逐 epoch 训练 / 验证，按 valid macro-AUC 保留最优权重。

    返回字典：
        history       每个 epoch 的记录（epoch / lr / 三个指标），用来写 metrics.csv
        best_epoch    最优的那一轮序号（从 1 开始）
        best_macro_auc 最优的 valid macro-AUC
        best_metrics  最优那一轮的完整验证指标
        best_state    最优那一轮的权重（在 CPU 上），训练结束由调用方加载回模型
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
    )
    scheduler = _build_scheduler(optimizer, config.train.warmup_epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=_amp_enabled(config, device))
    criterion = build_loss(config, _train_labels(train_loader))

    # best_state 先放一份初始权重：万一整场训练 AUC 都是 NaN，至少还有东西可存。
    best_state = _snapshot(model)
    best_auc = float("nan")
    best_epoch = 0
    best_metrics: dict[str, float] = {}
    history: list[dict[str, float]] = []
    stale_epochs = 0  # 连续多少个 epoch 没有提升

    for epoch in range(1, config.train.epochs + 1):
        # 学习率在训练这一步用的是“当前值”，所以先记下来再更新。
        lr = optimizer.param_groups[0]["lr"]
        train_loss = train_one_epoch(
            config, model, train_loader, criterion, optimizer, scaler, device
        )
        valid_metrics = evaluate(config, model, val_loader, criterion, device)
        auc = valid_metrics["macro_auc"]

        history.append(
            {
                "epoch": epoch,
                "lr": lr,
                "train_loss": train_loss,
                "valid_loss": valid_metrics["loss"],
                "valid_macro_auc": auc,
                "valid_macro_f1": valid_metrics["macro_f1"],
            }
        )
        print(
            f"epoch {epoch:2d}/{config.train.epochs} | lr {lr:.2e} | "
            f"train_loss {train_loss:.4f} | valid_loss {valid_metrics['loss']:.4f} | "
            f"macro_auc {auc:.4f} | macro_f1 {valid_metrics['macro_f1']:.4f}"
        )

        # 第一轮（best_auc 是 NaN）只要 AUC 有效就算提升；之后必须严格变大。
        improved = not math.isnan(auc) and (math.isnan(best_auc) or auc > best_auc)
        if improved:
            best_auc = auc
            best_epoch = epoch
            best_metrics = valid_metrics
            best_state = _snapshot(model)
            stale_epochs = 0
        else:
            stale_epochs += 1
            # early_stop_patience=0 表示关闭早停。
            patience = config.train.early_stop_patience
            if patience and stale_epochs >= patience:
                print(f"连续 {stale_epochs} 个 epoch 没有提升，提前结束训练")
                break

        scheduler.step()

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_macro_auc": best_auc,
        "best_metrics": best_metrics,
        "best_state": best_state,
    }
