"""训练入口：读配置 → 建数据 → 建模型 → 训练 → 保存最优权重。

用法（在仓库根目录下执行）：

    python src/run_train.py --config configs/default.yaml --seed 42
    python src/run_train.py --resume outputs/20261027-083159_example

每次训练都会在 outputs/ 下新建一个带时间戳的目录，里面固定放这些东西：

    config.yaml   本次训练实际使用的完整配置（以后复盘只看这个目录就够了）
    train.log     训练日志（和终端输出一模一样）
    metrics.csv   每个 epoch 的 loss / macro-AUC / macro-F1
    best.pt       valid macro-AUC 最高的那一轮权重
    last.pt       每个 epoch 结束时的完整状态，用于断点续训
    tb/           TensorBoard 曲线（仅在装了 tensorboard 且 train.tensorboard
                  不为 false 时生成；没装就跳过，不影响训练）
    tb_ema/       EMA 影子权重的独立曲线面板，只在 train.ema_enabled 时生成

曲线查看：`tensorboard --logdir outputs`，面板名就是 run 目录名加 `/tb`；
开了 EMA 时会多一个并列面板 `<run 目录名>/tb_ema`，tag 与 raw 面板同名。

`--resume` 既可以给 run 目录，也可以直接给其中的 last.pt。续训会复用原 run
目录、追加写 train.log、沿用已完成的 epoch 与最优权重，不会新建目录重头再来。
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import math
import random
import sys
import warnings
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

import pandas as pd
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from common.checkpoint import load_checkpoint, save_checkpoint
from common.config import Config, config_from_dict, load_config
from common.constants import CSV_ENCODING
from common.tracking import TENSORBOARD_DIR_NAME, TENSORBOARD_EMA_DIR_NAME, build_writer
from common.utils import make_run_dir, resolve_device, seed_everything
from data.dataset import build_loader
from data.resplit import MANIFEST_NAME, prepare_resplit, manifest_digest, checkpoint_manifest_path
from model.losses import build_loss
from model.network import build_model
from train.engine import TrainProgress, TrainRuntime, evaluate, fit, train_labels

LAST_CHECKPOINT_NAME = "last.pt"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。argv 显式传入是为了测试时复用，平时直接传 None 即可。"""
    parser = argparse.ArgumentParser(description="训练 ChestX-MLC 胸片多标签分类模型")
    parser.add_argument(
        "--config",
        default=None,
        help="YAML 配置文件路径，默认 configs/default.yaml；续训时省略则用 checkpoint 里存档的配置",
    )
    parser.add_argument("--seed", type=int, default=None, help="随机种子，覆盖配置文件里的值")
    parser.add_argument("--tag", default=None, help="输出目录后缀，默认由主干网络和图像尺寸拼成")
    parser.add_argument(
        "--resume",
        default=None,
        help=f"断点续训：run 目录或其中的 {LAST_CHECKPOINT_NAME}，省略则从头训练",
    )
    parser.add_argument(
        "--init-from",
        default=None,
        help="warm start：用某个 checkpoint（如 best.pt）的模型权重初始化后重新训练，与 --resume 互斥",
    )
    return parser.parse_args(argv)


class _Tee:
    """把 print 的内容同时写进终端和日志文件，省得再手写一遍日志。"""

    def __init__(self, stream: TextIO, log_file: TextIO) -> None:
        self._stream = stream
        self._log_file = log_file

    def write(self, text: str) -> int:
        self._stream.write(text)
        return self._log_file.write(text)

    def flush(self) -> None:
        self._stream.flush()
        self._log_file.flush()


def _resolve_resume(path: str | Path) -> tuple[Path, dict[str, Any]]:
    """把 --resume 解析成 (run 目录, last.pt 内容)。

    只有训练过程中生成的 last.pt 才能续训：普通 best.pt 没有优化器、调度器
    和进度信息，拿它“续训”等于换个初始化重新开始，这里直接报错说清楚。
    """
    resume_path = Path(path)
    if resume_path.is_dir():
        run_dir = resume_path
        checkpoint_path = run_dir / LAST_CHECKPOINT_NAME
    else:
        checkpoint_path = resume_path
        run_dir = resume_path.parent

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"找不到 {checkpoint_path}；只有训练过程中保存过 {LAST_CHECKPOINT_NAME} 的 run 才能续训"
        )

    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint.get("extras") is None:
        raise ValueError(f"{checkpoint_path} 不是训练断点（缺少 extras），无法续训")
    if checkpoint.get("config") is None:
        raise ValueError(f"{checkpoint_path} 里没有存档配置，无法重建训练环境")
    return run_dir, checkpoint


def _resolve_init_from(path: str | Path) -> tuple[Path, dict[str, Any], str]:
    """把 --init-from 解析成 (checkpoint 路径, checkpoint 内容, 权重 SHA-256)。

    与 --resume 不同，这里只取模型权重做初始化：训练从第 1 轮重新开始，
    优化器、调度器和 epoch 进度都不复用。
    """
    checkpoint_path = Path(path)
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"找不到 {checkpoint_path}；--init-from 需要现有 checkpoint 文件")
    checkpoint = load_checkpoint(checkpoint_path)
    if "model" not in checkpoint:
        raise ValueError(f"{checkpoint_path} 里没有模型权重，无法用于 warm start")
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return checkpoint_path, checkpoint, digest.hexdigest()


def _progress_from_checkpoint(checkpoint: dict[str, Any]) -> TrainProgress:
    """把 last.pt 里的进度还原成 TrainProgress（权重快照本来就在 CPU 上）。"""
    extras = checkpoint["extras"]
    return TrainProgress(
        epoch=int(checkpoint.get("epoch") or 0),
        best_auc=float(extras["best_auc"]),
        best_epoch=int(extras["best_epoch"]),
        best_metrics=dict(extras["best_metrics"]),
        best_state=extras["best_state"],
        history=[dict(record) for record in extras["history"]],
        stale_epochs=int(extras["stale_epochs"]),
        ema_state=extras.get("ema_state"),
        ema_best_auc=float(extras.get("ema_best_auc", float("nan"))),
        ema_best_epoch=int(extras.get("ema_best_epoch", 0)),
        ema_best_metrics=dict(extras.get("ema_best_metrics", {})),
        ema_best_state=extras.get("ema_best_state"),
    )


def _resume_checkpoint_config(checkpoint: dict[str, Any]) -> Config:
    """旧断点未记录 EMA 开关时，沿用当时默认关闭的训练语义。"""
    saved = checkpoint["config"]
    if "ema_enabled" not in saved.get("train", {}):
        saved = {**saved, "train": {**saved.get("train", {}), "ema_enabled": False}}
    return config_from_dict(saved)


def _validate_resume_config(config: Config, saved: Config) -> None:
    """只允许调整轮数和运行环境，保留训练及验证口径。"""
    current_values = config.to_dict()
    saved_values = saved.to_dict()
    for values in (current_values, saved_values):
        values.pop("device")
        values.pop("output")
        if saved.train.lr_scheduler != "cosine":
            values["train"].pop("epochs")
        values["train"].pop("num_workers")
        # tensorboard 只是观测开关，改它不会影响训练轨迹，不该拦续训。
        values["train"].pop("tensorboard")
        values["eval"].pop("batch_studies")
    differences = []
    for section, value in current_values.items():
        if isinstance(value, dict):
            for field, field_value in value.items():
                if field_value != saved_values[section][field]:
                    differences.append(f"{section}.{field}")
        elif value != saved_values[section]:
            differences.append(section)
    if differences:
        raise ValueError("续训配置与断点不兼容：" + ", ".join(differences))


def _write_metrics(path: Path, history: list[dict[str, float]]) -> None:
    """把当前全部 epoch 记录写成 metrics.csv（每轮重写，中断也不丢已完成的轮次）。

    各 epoch 的键不一定完全相同（只有最优那一轮会带 `*_fp32` 复评列），
    pandas 会按并集生成列、缺失处留空，所以列顺序以首次出现的顺序为准。
    """
    if not history:
        return
    pd.DataFrame(history).to_csv(path, index=False, encoding=CSV_ENCODING)


def _recheck_best_metrics(
    config: Config,
    model: torch.nn.Module,
    val_loader: DataLoader,
    train_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """用 fp32 前向把最优权重在 valid 上复评一次，作为「提交口径」的指标。

    训练期的验证跟着 `train.amp` 走 AMP（fp16）前向，而 `run_eval`、`run_infer`
    与提交都是 fp32 前向。两条路径的 logits 只差 1e-4~1e-3，但阈值 0.5 附近
    翻掉个位数样本就能让 macro-F1 差到千分位、macro-AUC 差到万分之一，
    于是 `metrics.csv` 与重算值长期对不上。这里只在训练结束后付一次 fp32
    验证的开销，把三处数字统一。

    重建 criterion 只为拿到 valid loss：pos_weight 只作用于训练损失，
    对 AUC / F1 没有影响。
    """
    criterion = build_loss(config, train_labels(train_loader), device=device).to(device)
    metrics = evaluate(config, model, val_loader, criterion, device, use_amp=False)
    return {key: value for key, value in metrics.items() if key != "per_class"}


def _record_fp32_metrics(
    history: list[dict[str, float]], best_epoch: int, metrics: dict[str, float]
) -> None:
    """把复评结果写进最优那一轮，列名带 `_fp32` 后缀。

    不覆盖原来的列：那几列是训练期 AMP 下的真实观测，逐轮曲线和早停判读
    都依赖它们；两套数并列才看得出精度口径造成的差有多大。
    """
    for record in history:
        if record.get("epoch") == best_epoch:
            record["valid_loss_fp32"] = metrics["loss"]
            record["valid_macro_auc_fp32"] = metrics["macro_auc"]
            record["valid_macro_f1_fp32"] = metrics["macro_f1"]
            return


def _report_fp32_delta(
    best_epoch: int, amp_metrics: dict[str, float], fp32_metrics: dict[str, float]
) -> None:
    """打印 AMP 与 fp32 两套指标，差异大到异常时能立刻看出来。"""

    def _format(key: str) -> str:
        fp32_value = float(fp32_metrics[key])
        amp_value = float(amp_metrics.get(key, float("nan")))
        if math.isnan(amp_value):
            return f"{fp32_value:.4f}（训练期 AMP 无有效值）"
        return f"{fp32_value:.4f}（AMP {amp_value:.4f}，Δ {fp32_value - amp_value:+.4f}）"

    print(
        f"最优权重 fp32 复评（epoch {best_epoch}）："
        f"macro_auc {_format('macro_auc')} | macro_f1 {_format('macro_f1')}"
    )


def _checkpoint_metadata(
    model: torch.nn.Module,
    config: Config,
    device: torch.device,
    pretrained_loaded: bool,
    previous: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """生成可核对模型训练口径的 checkpoint 元数据。"""
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    metadata: dict[str, Any] = {
        "pretrained_loaded": pretrained_loaded,
        "device": str(device),
        "freeze_backbone": config.model.freeze_backbone,
        "total_params": total_params,
        "trainable_params": trainable_params,
    }
    # B 组（TxRV 骨干）额外记录权重来源、路径与 SHA-256：checkpoint 能自证
    # 「用的确实是 densenet121-res224-all 的医学预训练权重」。
    pretrained_info = getattr(model, "pretrained_info", None)
    if callable(pretrained_info):
        metadata.update(pretrained_info())
        if previous is not None:
            for key in ("arch", "weights_name", "weights_path", "weights_sha256", "pretrained_loaded"):
                if key in previous:
                    metadata[key] = previous[key]
    return metadata


def _merge_metadata(metadata: dict[str, Any], extra: dict[str, Any] | None) -> dict[str, Any]:
    """把 warm start 来源等额外字段并进 checkpoint 元数据。"""
    if extra:
        metadata.update(extra)
    return metadata


def _capture_rng_state(device: torch.device | None = None) -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "mps": torch.mps.get_rng_state() if device is not None and device.type == "mps" else None}


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    cuda = state.get("cuda")
    if cuda is not None:
        if not torch.cuda.is_available() or len(cuda) != torch.cuda.device_count():
            warnings.warn("CUDA 设备与断点不一致，无法精确恢复 CUDA 随机状态", RuntimeWarning)
        else:
            torch.cuda.set_rng_state_all(cuda)
    if state.get("mps") is not None:
        if torch.backends.mps.is_available():
            torch.mps.set_rng_state(state["mps"].cpu())
        else:
            warnings.warn("MPS 设备不可用，无法恢复 MPS 随机状态", RuntimeWarning)


def _save_last_checkpoint(
    path: Path,
    model: torch.nn.Module,
    progress: TrainProgress,
    runtime: TrainRuntime,
    config: Config,
    device: torch.device,
    pretrained_loaded: bool,
    previous_metadata: dict[str, Any] | None = None,
    extra_metadata: dict[str, Any] | None = None,
    split_digest: str | None = None,
) -> None:
    """保存可续训的完整状态：权重 + 优化器/调度器/AMP + 进度与最优快照。"""
    metadata = _checkpoint_metadata(model, config, device, pretrained_loaded, previous_metadata)
    if extra_metadata is not None:
        metadata.update(extra_metadata)
    if split_digest:
        metadata["split_manifest_sha256"] = split_digest
    save_checkpoint(
        path,
        model,
        optimizer=runtime.optimizer,
        scheduler=runtime.scheduler,
        scaler=runtime.scaler,
        epoch=progress.epoch,
        metrics=progress.best_metrics,
        config=config,
        seed=config.seed,
        metadata=metadata,
        extras={
            "best_auc": progress.best_auc,
            "best_epoch": progress.best_epoch,
            "best_metrics": progress.best_metrics,
            "best_state": progress.best_state,
            "history": progress.history,
            "stale_epochs": progress.stale_epochs,
            "rng_state": _capture_rng_state(device),
            "ema_state": progress.ema_state,
            "ema_best_auc": progress.ema_best_auc,
            "ema_best_epoch": progress.ema_best_epoch,
            "ema_best_metrics": progress.ema_best_metrics,
            "ema_best_state": progress.ema_best_state,
            **({"criterion_state": runtime.criterion.state_dict()} if config.train.loss == "aucm" else {}),
        },
    )


def run_training(
    config: Config,
    device: torch.device,
    run_dir: Path,
    *,
    checkpoint: dict[str, Any] | None = None,
    init_from: tuple[Path, dict[str, Any], str] | None = None,
) -> dict[str, Any]:
    """真正的编排：建 loader、建模型、训练、写 metrics.csv、保存最优权重。

    `checkpoint` 是 `--resume` 读到的 last.pt；给了就从里面恢复权重、优化器
    状态和训练进度，接着跑剩下的 epoch。
    `init_from` 是 `--init-from` 读到的 (路径, checkpoint, SHA-256)；只把模型
    权重装进新模型，训练从第 1 轮重新开始（warm start）。
    """
    if checkpoint is not None:
        _validate_resume_config(config, _resume_checkpoint_config(checkpoint))
    if config.train.aucm_require_bce_init and checkpoint is None:
        if init_from is None:
            raise ValueError("本 AUCM 配置需要 --init-from 指定 BCE checkpoint")
        source_config = init_from[1].get("config")
        if source_config is None:
            raise ValueError("BCE 初始化 checkpoint 缺少配置，无法核对训练口径")
        source = config_from_dict(source_config)
        if source.train.loss != "bce" or source.model != config.model or source.data != config.data or source.image != config.image or source.seed != config.seed:
            raise ValueError("初始化 checkpoint 必须来自数据、图像和模型口径一致的 BCE 训练")
    if config.train.loss == "aucm" and config.data.resplit.enabled:
        raise ValueError("AUCM 暂不支持 resplit，必须使用固定划分")
    if config.train.loss == "aucm" and not config.train.aucm_require_bce_init and init_from is not None:
        raise ValueError("全程 AUCM 配置不接受 --init-from")
    # 曲线开关先于数据加载处理：mode=true 却没装 tensorboard 时应当立刻失败，
    # 而不是等几分钟建完 loader 才报错。auto 模式下不装就安静跳过。
    writer = build_writer(run_dir / TENSORBOARD_DIR_NAME, config.train.tensorboard)
    if writer is not None:
        print(
            f"训练曲线: {run_dir / TENSORBOARD_DIR_NAME} | "
            f"查看方式: tensorboard --logdir {run_dir.parent}"
        )
    # EMA 影子权重单独一个面板：开着训练曲线且开了 EMA 才建，否则保持 None。
    ema_writer = (
        build_writer(run_dir / TENSORBOARD_EMA_DIR_NAME, config.train.tensorboard)
        if config.train.ema_enabled and writer is not None
        else None
    )
    if ema_writer is not None:
        print(f"EMA 曲线: {run_dir / TENSORBOARD_EMA_DIR_NAME}（独立面板）")
    pin_memory = device.type == "cuda"
    manifest_path = run_dir / MANIFEST_NAME if config.data.resplit.enabled else None
    split_digest = None
    if manifest_path is not None:
        prepare_resplit(config, manifest_path, resume=checkpoint is not None)
        split_digest = manifest_digest(manifest_path)
        if checkpoint is not None:
            checkpoint_manifest_path(config, checkpoint, run_dir / "last.pt")
    train_loader = build_loader(config, "train", train=True, pin_memory=pin_memory, manifest_path=manifest_path)
    val_loader = build_loader(config, "valid", pin_memory=pin_memory, manifest_path=manifest_path)
    if config.train.distill_enabled:
        from train.distillation import TeacherCache, records_digest, sha256_file
        cache = TeacherCache.load(Path(config.train.distill_cache),
                                  [record.study_id for record in train_loader.dataset.records],
                                  sha256_file(config.data.train.csv),
                                  index_sha256=records_digest(train_loader.dataset.records))
        previous_digest = (checkpoint.get("metadata") or {}).get("distill_cache_sha256") if checkpoint else None
        if checkpoint is not None and previous_digest != cache.digest:
            raise ValueError("续训教师缓存摘要缺失或不一致")
        train_loader.dataset.teacher_cache = cache
        print(f"教师缓存 SHA-256: {cache.digest}")
    else:
        cache = None
    if manifest_path is not None:
        labels = np.stack([record.labels for record in val_loader.dataset.records])
        for i, positives in enumerate(labels.sum(axis=0), 1):
            negative = len(labels) - int(positives)
            print(f"验证集类别 {i}: 正 {int(positives)} / 负 {negative}" +
                  ("（AUC 无定义）" if positives == 0 or negative == 0 else ""))
    print(f"训练 study 数: {len(train_loader.dataset)} | 验证 study 数: {len(val_loader.dataset)}")

    # 续训或 warm start 时权重马上会被覆盖，构建模型不必再加载（或联网下载）预训练权重。
    build_config = (
        config
        if checkpoint is None and init_from is None
        else replace(config, model=replace(config.model, pretrained=False))
    )
    model = build_model(build_config).to(device)
    pretrained_loaded = bool(getattr(model, "pretrained_loaded", False))
    if config.train.loss == "aucm" and checkpoint is None and init_from is None and config.model.pretrained and not pretrained_loaded:
        raise RuntimeError("AUCM 预训练权重未加载，停止实验")
    warm_metadata: dict[str, Any] | None = None
    if init_from is not None:
        init_path, init_checkpoint, init_sha256 = init_from
        model.load_state_dict(init_checkpoint["model"])
        warm_metadata = {
            "warm_start_from": str(init_path),
            "warm_start_sha256": init_sha256,
            "warm_start_epoch": init_checkpoint.get("epoch"),
            "warm_start_metrics": init_checkpoint.get("metrics") or {},
        }
        print(
            f"warm start: 从 {init_path} 加载模型权重 "
            f"(source epoch {init_checkpoint.get('epoch')}, sha256 {init_sha256[:12]}…)"
        )
    # B 组把权重来源 / SHA-256 打进训练日志，作为「权重确实加载成功」的留痕；
    # 默认 timm 骨干没有 pretrained_info，不受影响。
    progress: TrainProgress | None = None
    optimizer_state = scheduler_state = scaler_state = None
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        progress = _progress_from_checkpoint(checkpoint)
        optimizer_state = checkpoint.get("optimizer")
        scheduler_state = checkpoint.get("scheduler")
        scaler_state = checkpoint.get("scaler")
        # 预训练是否真正加载过要以最初那次训练的记录为准，别被本次跳过加载覆盖。
        stored = (checkpoint.get("metadata") or {}).get("pretrained_loaded")
        if stored is not None:
            pretrained_loaded = bool(stored)
        rng_state = checkpoint["extras"].get("rng_state")
        if rng_state is None:
            warnings.warn("旧断点缺少随机状态，续训不能精确重现连续训练", RuntimeWarning)
        else:
            _restore_rng_state(rng_state)
        saved_config = _resume_checkpoint_config(checkpoint)
        old_device = (checkpoint.get("metadata") or {}).get("device")
        if saved_config.train.num_workers != config.train.num_workers or (old_device and old_device != str(device)):
            warnings.warn("续训设备或 num_workers 已改变，随机数恢复不保证训练轨迹一致", RuntimeWarning)
    if callable(getattr(model, "pretrained_info", None)):
        info = _checkpoint_metadata(model, config, device, pretrained_loaded, checkpoint.get("metadata") if checkpoint is not None else None)
        print(f"预训练权重: arch={info['arch']} name={info['weights_name']} "
              f"loaded={info['pretrained_loaded']} sha256={info['weights_sha256']}")
        print(f"权重文件: {info['weights_path']}")
    print(f"预训练权重: {'已加载' if pretrained_loaded else '未加载（随机初始化或配置关闭）'}")
    if progress is not None:
        print(
            f"从断点继续：已完成 {progress.epoch}/{config.train.epochs} 个 epoch，"
            f"最优 epoch {progress.best_epoch}（valid macro-AUC {progress.best_auc:.4f}）"
        )
        if progress.epoch >= config.train.epochs:
            print(f"该 run 已跑满 {config.train.epochs} 个 epoch，只重新导出 best.pt")

    if cache is not None:
        warm_metadata = {**(warm_metadata or {}), "distill_cache_sha256": cache.digest}

    def on_epoch_end(current: TrainProgress, runtime: TrainRuntime) -> None:
        # 先写指标再存权重：两个文件都以“当前 epoch 已完成”为准。
        _write_metrics(run_dir / "metrics.csv", current.history)
        _save_last_checkpoint(
            run_dir / LAST_CHECKPOINT_NAME,
            model,
            current,
            runtime,
            config,
            device,
            pretrained_loaded,
            checkpoint.get("metadata") if checkpoint is not None else None,
            extra_metadata=warm_metadata,
            split_digest=split_digest,
        )

    try:
        result = fit(
            config,
            model,
            train_loader,
            val_loader,
            device,
            progress=progress,
            optimizer_state=optimizer_state,
            scheduler_state=scheduler_state,
            scaler_state=scaler_state,
            **({"criterion_state": checkpoint["extras"].get("criterion_state") if checkpoint else None} if config.train.loss == "aucm" else {}),
            on_epoch_end=on_epoch_end,
            writer=writer,
            ema_writer=ema_writer,
        )
    finally:
        # 训练抛异常也要关掉 writer，否则 events 文件的最后一段可能没落盘。
        if writer is not None:
            writer.close()
        if ema_writer is not None:
            ema_writer.close()

    # best_state 是所有 epoch 里由 fit 挑出来的最优权重，最后才落盘，
    # 这样 best.pt 永远是完整文件，不会出现“写到一半”的坏权重。
    model.load_state_dict(result["best_state"])
    # 落盘前用 fp32 复评一次：训练期验证是 AMP 前向，与 run_eval / 提交的
    # fp32 路径不同源，直接拿 metrics.csv 里的数当提交分数会差千分位。
    amp_metrics = dict(result["best_metrics"])
    result["best_metrics"] = _recheck_best_metrics(config, model, val_loader, train_loader, device)
    _record_fp32_metrics(result["history"], result["best_epoch"], result["best_metrics"])
    _write_metrics(run_dir / "metrics.csv", result["history"])
    _report_fp32_delta(result["best_epoch"], amp_metrics, result["best_metrics"])
    checkpoint_path = save_checkpoint(
        run_dir / "best.pt",
        model,
        epoch=result["best_epoch"],
        metrics=result["best_metrics"],
        config=config,
        seed=config.seed,
        metadata={
            **_merge_metadata(
                _checkpoint_metadata(model, config, device, pretrained_loaded, checkpoint.get("metadata") if checkpoint is not None else None),
                warm_metadata,
            ),
            **({"split_manifest_sha256": split_digest} if split_digest else {}),
        },
    )
    print(f"最优权重已保存: {checkpoint_path}")
    progress = result["progress"]
    if config.train.ema_enabled and progress.ema_best_state is not None:
        model.load_state_dict(progress.ema_best_state)
        ema_metrics = _recheck_best_metrics(config, model, val_loader, train_loader, device)
        save_checkpoint(run_dir / "best_ema.pt", model, epoch=progress.ema_best_epoch,
                        metrics=ema_metrics, config=config, seed=config.seed,
                        metadata={**_merge_metadata(_checkpoint_metadata(model, config, device, pretrained_loaded, checkpoint.get("metadata") if checkpoint is not None else None), warm_metadata),
                                  **({"split_manifest_sha256": split_digest} if split_digest else {}), "weights_kind": "ema"})
        model.load_state_dict(result["best_state"])
        print(f"EMA 最优权重已保存: {run_dir / 'best_ema.pt'}")
    elif config.train.ema_enabled:
        warnings.warn("EMA 验证 AUC 均无效，未生成 best_ema.pt", RuntimeWarning)
    return result


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    resume_dir: Path | None = None
    checkpoint: dict[str, Any] | None = None
    if args.resume:
        resume_dir, checkpoint = _resolve_resume(args.resume)
    init_from: tuple[Path, dict[str, Any], str] | None = None
    if args.init_from:
        if checkpoint is not None:
            raise ValueError("--init-from 和 --resume 不能同时使用")
        init_from = _resolve_init_from(args.init_from)

    # 续训时优先用 last.pt 里存档的配置：它记录的就是这次训练真正的超参。
    if checkpoint is not None and args.config is None:
        config = _resume_checkpoint_config(checkpoint)
        config_source = f"{Path(args.resume)} 中存档的配置"
    else:
        config_path = args.config or "configs/default.yaml"
        config = load_config(config_path)
        config_source = config_path
    if args.seed is not None:
        config = replace(config, seed=args.seed)

    if checkpoint is not None:
        _validate_resume_config(config, _resume_checkpoint_config(checkpoint))

    seed_everything(config.seed)
    device = resolve_device(config.device)

    if resume_dir is not None:
        # 复用原 run 目录：产物、日志都在这里，续训不该另开一份。
        run_dir = resume_dir
    else:
        tag = args.tag or f"{config.model.backbone}_s{config.image.size}"
        run_dir = make_run_dir(config.output.run_dir, tag)

    config_text = yaml.safe_dump(config.to_dict(), allow_unicode=True, sort_keys=False)
    if resume_dir is None:
        # 先把配置存档再训练：以后翻到某个 run 目录，能立刻知道当时用的完整设置。
        (run_dir / "config.yaml").write_text(config_text, encoding=CSV_ENCODING)
    else:
        # 保留最初的 config.yaml 不动，另存一份本次续训实际生效的配置。
        (run_dir / "config_resume.yaml").write_text(config_text, encoding=CSV_ENCODING)

    # 续训追加写日志，保证一个 run 的 train.log 是完整连续的。
    log_mode = "a" if resume_dir is not None else "w"
    with (run_dir / "train.log").open(log_mode, encoding=CSV_ENCODING) as log_file:
        with (
            contextlib.redirect_stdout(_Tee(sys.stdout, log_file)),
            contextlib.redirect_stderr(_Tee(sys.stderr, log_file)),
        ):
            if resume_dir is not None:
                print(f"===== 续训 {datetime.now():%Y-%m-%d %H:%M:%S} =====")
            print(f"配置文件: {config_source}")
            print(f"运行目录: {run_dir}")
            print(f"设备: {device} | 随机种子: {config.seed}")
            if init_from is not None:
                print(f"warm start 来源: {init_from[0]} (sha256 {init_from[2]})")
            try:
                result = run_training(config, device, run_dir, checkpoint=checkpoint, init_from=init_from)
            except KeyboardInterrupt:
                print(
                    f"训练被中断：每个 epoch 结束都会保存 {run_dir / LAST_CHECKPOINT_NAME}，"
                    f"已完成的轮次不会丢；用 --resume {run_dir} 继续"
                )
                raise
            print(
                f"训练结束：最优 epoch {result['best_epoch']}，"
                f"valid macro-AUC {result['best_metrics']['macro_auc']:.4f}"
                f"（fp32 复评口径；权重挑选时 AMP 为 {result['best_macro_auc']:.4f}）"
            )


if __name__ == "__main__":
    main()
