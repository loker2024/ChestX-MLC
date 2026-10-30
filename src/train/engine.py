"""训练引擎：单 epoch 的训练 / 验证循环、学习率 warmup、早停与最优权重挑选。

完整的一次训练由 `src/run_train.py` 编排（读配置、建 loader、建模型），
本模块只负责“给定模型和数据，把它训练好”这件事：

    for epoch in range(epochs):
        训练一个 epoch（前向 + 反向 + 更新参数）
        在 valid 上算 loss / macro-AUC / macro-F1
        按 macro-AUC 决定要不要“记住”这一轮的权重
        连续若干轮都没有变好，就提前结束

断点续训：进度（已完成轮次、最优快照、历史记录、早停计数）放在 `TrainProgress`，
优化器 / 学习率调度 / AMP scaler 放在 `TrainRuntime`；`fit` 接受前者和三个
`*_state` 接着跑，并通过 `on_epoch_end` 每轮把状态交给调用方落盘。

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
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from common.config import Config
from common.metrics import macro_auc, macro_f1
from model.losses import build_loss
from model.network import aggregate_logits, set_frozen_backbone_eval


@dataclass
class TrainProgress:
    """可以脱离进程保存、之后再接上的训练进度。

    这里只放“纯 Python 对象”——权重快照 `best_state` 是 CPU 上的张量字典，
    其余都是数字、字典和列表，所以调用方可以原样塞进 checkpoint。
    优化器 / 学习率调度 / AMP scaler 的状态在 `TrainRuntime` 里，由回调按需取。
    """

    epoch: int = 0  # 已完成的 epoch 数，下一轮从 epoch + 1 开始
    best_auc: float = float("nan")
    best_epoch: int = 0
    best_metrics: dict[str, float] = field(default_factory=dict)
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = field(default_factory=list)
    stale_epochs: int = 0  # 连续多少个 epoch 没有提升


@dataclass
class TrainRuntime:
    """训练期间才有状态的对象，每个 epoch 结束时交给回调去序列化。"""

    optimizer: Optimizer
    scheduler: LambdaLR
    scaler: torch.amp.GradScaler


# 每轮训练结束后调用，参数是当前进度和带状态的运行时对象。
EpochEndHook = Callable[[TrainProgress, TrainRuntime], None]


def _amp_enabled(config: Config, device: torch.device) -> bool:
    """项目只在 CUDA 上开混合精度：CPU / MPS 上没收益，还容易遇到算子不支持。"""
    return config.train.amp and device.type == "cuda"


def _build_scheduler(optimizer: Optimizer, config: Config) -> LambdaLR:
    """按 `train.lr_scheduler` 构造学习率调度。

    LambdaLR 的 lr_lambda 收到的是“已经走过的步数”。这里每个 epoch 只调用一次
    scheduler.step()，所以“步数”就是 epoch 序号（从 0 开始）：

    - `warmup`（默认）：前 warmup_epochs 轮把学习率从 lr/warmup 线性涨到 lr，
      之后不变。warmup_epochs=2 时，第 1 轮 lr*0.5、第 2 轮 lr*1.0，之后一直是 lr。
    - `cosine`：同样的线性 warmup，之后按半个余弦周期从 lr 衰减到 lr_min，
      最后一轮正好落在 lr_min（例如 lr=3e-4、lr_min=1e-5、epochs=25）。
    """
    warmup_epochs = config.train.warmup_epochs
    if config.train.lr_scheduler == "cosine":
        total_epochs = config.train.epochs
        lr_floor = config.train.lr_min / config.train.lr
        # 预热结束后开始衰减，分母用“最后一轮的步数 - warmup”，保证末轮正好到 lr_min。
        decay_steps = max(1, total_epochs - 1 - warmup_epochs)

        def cosine_lr(step: int) -> float:
            if warmup_epochs > 0 and step < warmup_epochs:
                return (step + 1) / warmup_epochs
            progress = min(1.0, max(0.0, (step - warmup_epochs) / decay_steps))
            return lr_floor + (1.0 - lr_floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

        return LambdaLR(optimizer, lr_lambda=cosine_lr)
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
    if config.model.freeze_backbone:
        set_frozen_backbone_eval(model)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
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
            study_logits = aggregate_logits(model(images), offsets,
                batch["views"].to(device, non_blocking=True) if config.model.aggregation == "view_weighted" else None,
                config.model.frontal_weight)
            loss = criterion(study_logits, labels)

        scaler.scale(loss).backward()
        if config.train.grad_clip > 0:
            # 裁剪必须放在“反缩放”之后、optimizer.step() 之前，
            # 否则裁的是放大过的梯度，阈值 5.0 就失去意义了。
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(trainable_parameters, config.train.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        num_studies = len(batch["study_ids"])
        total_loss += loss.item() * num_studies
        total_studies += num_studies

    if total_studies == 0:
        raise RuntimeError("训练集为空，无法计算 loss")
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
            study_logits = aggregate_logits(model(images), offsets,
                batch["views"].to(device, non_blocking=True) if config.model.aggregation == "view_weighted" else None,
                config.model.frontal_weight)

        num_studies = len(batch["study_ids"])
        total_loss += criterion(study_logits, labels).item() * num_studies
        total_studies += num_studies

        # 先转 float32 再搬 CPU：混合精度下 logits 可能是 fp16，直接 sigmoid 精度不够。
        logits_list.append(study_logits.float().cpu())
        labels_list.append(labels.float().cpu())

    if total_studies == 0:
        raise RuntimeError("验证集为空，无法计算指标")
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
    *,
    progress: TrainProgress | None = None,
    optimizer_state: dict[str, Any] | None = None,
    scheduler_state: dict[str, Any] | None = None,
    scaler_state: dict[str, Any] | None = None,
    on_epoch_end: EpochEndHook | None = None,
) -> dict[str, Any]:
    """完整训练流程：逐 epoch 训练 / 验证，按 valid macro-AUC 保留最优权重。

    返回字典：
        history       每个 epoch 的记录（epoch / lr / 三个指标），用来写 metrics.csv
        best_epoch    最优的那一轮序号（从 1 开始）
        best_macro_auc 最优的 valid macro-AUC
        best_metrics  最优那一轮的完整验证指标
        best_state    最优那一轮的权重（在 CPU 上），训练结束由调用方加载回模型
        progress      结束时的进度对象，续训时原样传回来即可

    断点续训：把上次的 `progress` 连同三个 `*_state`（优化器 / 调度器 / AMP）
    一起传进来，训练就从 `progress.epoch + 1` 接着跑，最优权重、历史记录和
    早停计数都沿用旧值。`on_epoch_end` 在调度器推进后调用，保存的状态可直接
    用于下一轮；旧版断点的调度器尚未推进，恢复时按已完成轮次补上一步。
    """
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("模型没有可训练参数")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
    )
    scheduler = _build_scheduler(optimizer, config)
    scaler = torch.amp.GradScaler(device.type, enabled=_amp_enabled(config, device))
    # 先恢复优化器再恢复调度器：LambdaLR 的 last_epoch 决定下一轮的学习率。
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    if scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
    if scaler_state is not None:
        scaler.load_state_dict(scaler_state)
    runtime = TrainRuntime(optimizer=optimizer, scheduler=scheduler, scaler=scaler)

    criterion = build_loss(config, _train_labels(train_loader), device=device).to(device)
    progress = TrainProgress() if progress is None else progress
    if scheduler_state is not None and scheduler.last_epoch == progress.epoch - 1:
        scheduler.step()  # 旧版 last.pt 是在 scheduler.step() 之前保存的。
    # best_state 记录当前最优权重；续训时沿用存档里的，不能拿当前权重覆盖。
    # 若整场 AUC 都无效，函数末尾会显式报错，而不是保存随机权重。
    if progress.best_state is None:
        progress.best_state = _snapshot(model)

    for epoch in range(progress.epoch + 1, config.train.epochs + 1):
        # 学习率在训练这一步用的是“当前值”，所以先记下来再更新。
        lr = optimizer.param_groups[0]["lr"]
        train_loss = train_one_epoch(
            config, model, train_loader, criterion, optimizer, scaler, device
        )
        valid_metrics = evaluate(config, model, val_loader, criterion, device)
        auc = valid_metrics["macro_auc"]

        progress.history.append(
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
        improved = not math.isnan(auc) and (
            math.isnan(progress.best_auc) or auc > progress.best_auc
        )
        if improved:
            progress.best_auc = auc
            progress.best_epoch = epoch
            progress.best_metrics = valid_metrics
            progress.best_state = _snapshot(model)
            progress.stale_epochs = 0
        else:
            progress.stale_epochs += 1

        progress.epoch = epoch
        scheduler.step()
        if on_epoch_end is not None:
            on_epoch_end(progress, runtime)

        # early_stop_patience=0 表示关闭早停。早停也要先落盘再退出，
        # 否则最后一轮的指标和进度会丢。
        patience = config.train.early_stop_patience
        if not improved and patience and progress.stale_epochs >= patience:
            print(f"连续 {progress.stale_epochs} 个 epoch 没有提升，提前结束训练")
            break

    if progress.best_epoch == 0:
        raise RuntimeError(
            "验证集所有类别的 AUC 都是 NaN（可能全为阳性或全为阴性），无法挑选最优权重；"
            "请检查 valid 划分或增加验证样本"
        )

    return {
        "history": progress.history,
        "best_epoch": progress.best_epoch,
        "best_macro_auc": progress.best_auc,
        "best_metrics": progress.best_metrics,
        "best_state": progress.best_state,
        "progress": progress,
    }
