"""训练引擎测试：用 2 个 study / 3 张小图的合成数据跑通训练循环。

这里不追求指标好看，只验证“训练确实在动”：loss 是有限数、参数被更新、
warmup 的学习率按预期变化、早停会在 AUC 停滞时停下。
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from common.config import Config, load_config
from data.dataset import build_loader
from model.losses import build_loss
from model.network import build_model
from train import engine
from train.engine import fit


def _make_config(
    config_path: Path,
    epochs: int = 1,
    warmup_epochs: int = 0,
    early_stop_patience: int = 0,
) -> Config:
    """在测试配置基础上关掉预训练、缩短轮数，CPU 上也能几秒跑完。"""
    config = load_config(config_path)
    return replace(
        config,
        model=replace(config.model, pretrained=False),
        train=replace(
            config.train,
            epochs=epochs,
            warmup_epochs=warmup_epochs,
            early_stop_patience=early_stop_patience,
        ),
    )


@pytest.fixture
def config(tmp_path: Path, labeled_split: dict, write_config) -> Config:
    """train / valid / test 都指向同一份 2 个 study 的合成数据。"""
    split = (labeled_split["csv"], labeled_split["images"])
    config_path = write_config({"train": split, "valid": split, "test": split})
    return _make_config(config_path)


def _loaders(config: Config):
    """建训练和验证 loader；训练集走增强分支，验证集走纯评估分支。"""
    return build_loader(config, "train", train=True), build_loader(config, "valid")


def test_fit_one_epoch_updates_weights(config: Config) -> None:
    device = torch.device("cpu")
    torch.manual_seed(0)
    train_loader, val_loader = _loaders(config)
    model = build_model(config)
    before = {name: value.clone() for name, value in model.state_dict().items()}

    result = fit(config, model, train_loader, val_loader, device)

    assert len(result["history"]) == 1
    record = result["history"][0]
    assert math.isfinite(record["train_loss"])
    assert math.isfinite(record["valid_loss"])
    assert 0.0 <= record["valid_macro_auc"] <= 1.0
    assert 0.0 <= record["valid_macro_f1"] <= 1.0
    # 训练一个 epoch 之后，参数必须真的被反向传播更新过。
    assert any(not torch.equal(value, before[name]) for name, value in model.state_dict().items())
    assert result["best_epoch"] == 1
    assert set(result["best_state"]) == set(model.state_dict())


def test_warmup_ramps_learning_rate(config: Config) -> None:
    # 配置要求 warmup_epochs < epochs，所以用 3 轮训练、前 2 轮 warmup。
    config = replace(config, train=replace(config.train, epochs=3, warmup_epochs=2))
    train_loader, val_loader = _loaders(config)

    result = fit(config, build_model(config), train_loader, val_loader, torch.device("cpu"))

    # 第 1 轮 lr/2（预热刚起步），第 2 轮涨到 lr，第 3 轮保持 lr。
    lrs = [record["lr"] for record in result["history"]]
    assert lrs == pytest.approx([config.train.lr / 2, config.train.lr, config.train.lr])


def test_evaluate_reports_loss_and_metrics(config: Config) -> None:
    device = torch.device("cpu")
    _, val_loader = _loaders(config)
    model = build_model(config)

    metrics = engine.evaluate(config, model, val_loader, build_loss(config), device)

    assert set(metrics) == {"loss", "macro_auc", "macro_f1"}
    assert metrics["loss"] > 0.0
    assert 0.0 <= metrics["macro_auc"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0


def test_early_stopping_stops_when_auc_stops_improving(config: Config, monkeypatch) -> None:
    config = replace(config, train=replace(config.train, epochs=5, early_stop_patience=2))
    train_loader, val_loader = _loaders(config)

    # 让 valid 的 AUC 永远停在 0.5：第 1 轮算“初始最优”，
    # 之后连续 2 轮没有提升，训练应当只跑 3 个 epoch 就停下。
    monkeypatch.setattr(
        engine,
        "evaluate",
        lambda *args, **kwargs: {"loss": 0.5, "macro_auc": 0.5, "macro_f1": 0.5},
    )
    result = fit(config, build_model(config), train_loader, val_loader, torch.device("cpu"))

    assert len(result["history"]) == 3
    assert result["best_epoch"] == 1
