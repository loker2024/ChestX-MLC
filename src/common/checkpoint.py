"""模型权重的保存与加载。

checkpoint 里不只存权重，还可以存优化器状态、epoch、指标、配置、seed 与额外 metadata。
这样拿到任意一个 best.pt，都能回答“它是在什么配置、什么种子下训练出来的”，
复现和排错时不用再翻当时的命令行。
"""

import os
import tempfile
from pathlib import Path
from typing import Any

import torch

from common.config import Config


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    epoch: int | None = None,
    metrics: dict[str, float] | None = None,
    config: Config | None = None,
    seed: int | None = None,
    metadata: dict[str, Any] | None = None,
    extras: dict[str, Any] | None = None,
) -> Path:
    """把训练状态保存到 path，返回实际写入的路径。

    只有 model 是必填的：训练中途存快照时把优化器和 epoch 也带上，
    恢复训练才能接上学习率调度和动量；只做推理时可以只保存权重。
    断点续训还会带上 `scheduler`（学习率走到了哪）、`scaler`（AMP 的缩放系数）
    和 `extras`（训练进度、最优权重快照等不固定结构的内容）。

    父目录不存在会自动创建，所以 `outputs/<run>/best.pt` 这种路径不用手动建目录。
    """
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "metrics": metrics,
        "config": config.to_dict() if config is not None else None,
        "seed": seed,
        "metadata": metadata,
        "extras": extras,
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


def load_checkpoint(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    """读取 checkpoint，返回保存时的字典。

    默认加载到 CPU，之后按需 `model.to(device)`；如果在 GPU 上直接恢复,
    可以传 map_location="cuda" 省一次搬运。

    返回的 dict 至少含 "model"；其余键在保存时没提供就是 None。推理时典型用法：

        ckpt = load_checkpoint("outputs/xxx/best.pt")
        model.load_state_dict(ckpt["model"])
    """
    # checkpoint 由本仓库自己生成，除张量外还含配置、指标等普通 Python 对象，
    # 所以关掉 PyTorch 2.6+ 默认的 weights_only 受限反序列化。
    # 注意：只加载自己产出或可信来源的 .pt 文件。
    return torch.load(Path(path), map_location=map_location, weights_only=False)
