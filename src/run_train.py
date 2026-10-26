"""训练入口：读配置 → 建数据 → 建模型 → 训练 → 保存最优权重。

用法（在仓库根目录下执行）：

    python src/run_train.py --config configs/default.yaml --seed 42

每次训练都会在 outputs/ 下新建一个带时间戳的目录，里面固定放四样东西：

    config.yaml   本次训练实际使用的完整配置（以后复盘只看这个目录就够了）
    train.log     训练日志（和终端输出一模一样）
    metrics.csv   每个 epoch 的 loss / macro-AUC / macro-F1
    best.pt       valid macro-AUC 最高的那一轮权重
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, TextIO

import pandas as pd
import torch
import yaml

from common.checkpoint import save_checkpoint
from common.config import Config, load_config
from common.constants import CSV_ENCODING
from common.utils import make_run_dir, resolve_device, seed_everything
from data.dataset import build_loader
from model.network import build_model
from train.engine import fit


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。argv 显式传入是为了测试时复用，平时直接传 None 即可。"""
    parser = argparse.ArgumentParser(description="训练 ChestX-MLC 胸片多标签分类模型")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML 配置文件路径")
    parser.add_argument("--seed", type=int, default=None, help="随机种子，覆盖配置文件里的值")
    parser.add_argument("--tag", default=None, help="输出目录后缀，默认由主干网络和图像尺寸拼成")
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


def run_training(config: Config, device: torch.device, run_dir: Path) -> dict[str, Any]:
    """真正的编排：建 loader、建模型、训练、写 metrics.csv、保存最优权重。"""
    pin_memory = device.type == "cuda"
    train_loader = build_loader(config, "train", train=True, pin_memory=pin_memory)
    val_loader = build_loader(config, "valid", pin_memory=pin_memory)
    print(f"训练 study 数: {len(train_loader.dataset)} | 验证 study 数: {len(val_loader.dataset)}")

    model = build_model(config).to(device)
    pretrained_loaded = bool(getattr(model, "pretrained_loaded", False))
    print(f"预训练权重: {'已加载' if pretrained_loaded else '未加载（随机初始化或配置关闭）'}")
    result = fit(config, model, train_loader, val_loader, device)

    # 一行一个 epoch，列名就是 engine 返回记录里的键，方便之后直接画曲线。
    pd.DataFrame(result["history"]).to_csv(
        run_dir / "metrics.csv", index=False, encoding=CSV_ENCODING
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
        metadata={"pretrained_loaded": pretrained_loaded, "device": str(device)},
    )
    print(f"最优权重已保存: {checkpoint_path}")
    return result


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    if args.seed is not None:
        config = replace(config, seed=args.seed)

    seed_everything(config.seed)
    device = resolve_device(config.device)
    tag = args.tag or f"{config.model.backbone}_s{config.image.size}"
    run_dir = make_run_dir(config.output.run_dir, tag)

    # 先把配置存档再训练：以后翻到某个 run 目录，能立刻知道当时用的完整设置。
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config.to_dict(), allow_unicode=True, sort_keys=False),
        encoding=CSV_ENCODING,
    )

    with (run_dir / "train.log").open("w", encoding=CSV_ENCODING) as log_file:
        with (
            contextlib.redirect_stdout(_Tee(sys.stdout, log_file)),
            contextlib.redirect_stderr(_Tee(sys.stderr, log_file)),
        ):
            print(f"配置文件: {args.config}")
            print(f"运行目录: {run_dir}")
            print(f"设备: {device} | 随机种子: {config.seed}")
            result = run_training(config, device, run_dir)
            print(
                f"训练结束：最优 epoch {result['best_epoch']}，"
                f"valid macro-AUC {result['best_macro_auc']:.4f}"
            )


if __name__ == "__main__":
    main()
