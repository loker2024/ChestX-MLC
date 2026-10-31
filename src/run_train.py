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

`--resume` 既可以给 run 目录，也可以直接给其中的 last.pt。续训会复用原 run
目录、追加写 train.log、沿用已完成的 epoch 与最优权重，不会新建目录重头再来。
"""

from __future__ import annotations

import argparse
import contextlib
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

from common.checkpoint import load_checkpoint, save_checkpoint
from common.config import Config, config_from_dict, load_config
from common.constants import CSV_ENCODING
from common.utils import make_run_dir, resolve_device, seed_everything
from data.dataset import build_loader
from model.network import build_model
from train.engine import TrainProgress, TrainRuntime, fit

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
    )


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
    """把当前全部 epoch 记录写成 metrics.csv（每轮重写，中断也不丢已完成的轮次）。"""
    if not history:
        return
    pd.DataFrame(history).to_csv(path, index=False, encoding=CSV_ENCODING)


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
) -> None:
    """保存可续训的完整状态：权重 + 优化器/调度器/AMP + 进度与最优快照。"""
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
        metadata=_checkpoint_metadata(model, config, device, pretrained_loaded, previous_metadata),
        extras={
            "best_auc": progress.best_auc,
            "best_epoch": progress.best_epoch,
            "best_metrics": progress.best_metrics,
            "best_state": progress.best_state,
            "history": progress.history,
            "stale_epochs": progress.stale_epochs,
            "rng_state": _capture_rng_state(device),
        },
    )


def run_training(
    config: Config,
    device: torch.device,
    run_dir: Path,
    *,
    checkpoint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """真正的编排：建 loader、建模型、训练、写 metrics.csv、保存最优权重。

    `checkpoint` 是 `--resume` 读到的 last.pt；给了就从里面恢复权重、优化器
    状态和训练进度，接着跑剩下的 epoch。
    """
    if checkpoint is not None:
        _validate_resume_config(config, config_from_dict(checkpoint["config"]))
    pin_memory = device.type == "cuda"
    train_loader = build_loader(config, "train", train=True, pin_memory=pin_memory)
    val_loader = build_loader(config, "valid", pin_memory=pin_memory)
    print(f"训练 study 数: {len(train_loader.dataset)} | 验证 study 数: {len(val_loader.dataset)}")

    # 续训时权重马上会被 last.pt 覆盖，构建模型不必再加载（或联网下载）预训练权重。
    build_config = (
        config
        if checkpoint is None
        else replace(config, model=replace(config.model, pretrained=False))
    )
    model = build_model(build_config).to(device)
    pretrained_loaded = bool(getattr(model, "pretrained_loaded", False))
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
        saved_config = config_from_dict(checkpoint["config"])
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
        )

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
        on_epoch_end=on_epoch_end,
    )

    # best_state 是所有 epoch 里由 fit 挑出来的最优权重，最后才落盘，
    # 这样 best.pt 永远是完整文件，不会出现“写到一半”的坏权重。
    model.load_state_dict(result["best_state"])
    checkpoint_path = save_checkpoint(
        run_dir / "best.pt",
        model,
        epoch=result["best_epoch"],
        metrics=result["best_metrics"],
        config=config,
        seed=config.seed,
        metadata=_checkpoint_metadata(model, config, device, pretrained_loaded, checkpoint.get("metadata") if checkpoint is not None else None),
    )
    print(f"最优权重已保存: {checkpoint_path}")
    return result


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    resume_dir: Path | None = None
    checkpoint: dict[str, Any] | None = None
    if args.resume:
        resume_dir, checkpoint = _resolve_resume(args.resume)

    # 续训时优先用 last.pt 里存档的配置：它记录的就是这次训练真正的超参。
    if checkpoint is not None and args.config is None:
        config = config_from_dict(checkpoint["config"])
        config_source = f"{Path(args.resume)} 中存档的配置"
    else:
        config_path = args.config or "configs/default.yaml"
        config = load_config(config_path)
        config_source = config_path
    if args.seed is not None:
        config = replace(config, seed=args.seed)

    if checkpoint is not None:
        _validate_resume_config(config, config_from_dict(checkpoint["config"]))

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
            try:
                result = run_training(config, device, run_dir, checkpoint=checkpoint)
            except KeyboardInterrupt:
                print(
                    f"训练被中断：每个 epoch 结束都会保存 {run_dir / LAST_CHECKPOINT_NAME}，"
                    f"已完成的轮次不会丢；用 --resume {run_dir} 继续"
                )
                raise
            print(
                f"训练结束：最优 epoch {result['best_epoch']}，"
                f"valid macro-AUC {result['best_macro_auc']:.4f}"
            )


if __name__ == "__main__":
    main()
