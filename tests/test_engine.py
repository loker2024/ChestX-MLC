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


def test_fit_with_probability_space_aggregation(config: Config) -> None:
    """A-008 口径：概率空间聚合也要能训练——loss 有限、参数更新、指标可算。

    概率空间分支返回的是 `logit(mean(σ(z)))`，仍交给 BCEWithLogitsLoss，
    所以这里同时验证了「梯度能穿过 sigmoid / logit / clamp 回到骨干」。
    """
    g1 = replace(
        config,
        model=replace(
            config.model, aggregation="view_weighted_probability", frontal_weight=0.7
        ),
    )
    device = torch.device("cpu")
    torch.manual_seed(0)
    train_loader, val_loader = _loaders(g1)
    model = build_model(g1)
    before = {name: value.clone() for name, value in model.state_dict().items()}

    result = fit(g1, model, train_loader, val_loader, device)

    record = result["history"][0]
    assert math.isfinite(record["train_loss"])
    assert math.isfinite(record["valid_loss"])
    assert 0.0 <= record["valid_macro_auc"] <= 1.0
    assert any(
        not torch.equal(value, before[name]) for name, value in model.state_dict().items()
    )


def test_frozen_fit_only_passes_trainable_parameters_to_optimizer(
    config: Config, monkeypatch
) -> None:
    config = replace(config, model=replace(config.model, freeze_backbone=True))
    train_loader, val_loader = _loaders(config)
    model = build_model(config)
    named_parameters = dict(model.named_parameters())
    before = {
        name: parameter.detach().clone()
        for name, parameter in named_parameters.items()
        if not parameter.requires_grad
    }
    before_batch_norm_buffers = {
        name: value.clone()
        for name, value in model.state_dict().items()
        if name.endswith("running_mean") or name.endswith("running_var")
    }
    captured: list[torch.nn.Parameter] = []
    original_adamw = torch.optim.AdamW

    def recording_adamw(parameters, **kwargs):
        captured.extend(list(parameters))
        return original_adamw(captured, **kwargs)

    monkeypatch.setattr(engine.torch.optim, "AdamW", recording_adamw)
    fit(config, model, train_loader, val_loader, torch.device("cpu"))

    expected = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert {id(parameter) for parameter in captured} == {
        id(parameter) for parameter in expected
    }
    for name, value in before.items():
        assert torch.equal(model.state_dict()[name], value)
    for name, value in before_batch_norm_buffers.items():
        assert torch.equal(model.state_dict()[name], value)


def test_warmup_ramps_learning_rate(config: Config) -> None:
    # 配置要求 warmup_epochs < epochs，所以用 3 轮训练、前 2 轮 warmup。
    config = replace(config, train=replace(config.train, epochs=3, warmup_epochs=2))
    train_loader, val_loader = _loaders(config)

    result = fit(config, build_model(config), train_loader, val_loader, torch.device("cpu"))

    # 第 1 轮 lr/2（预热刚起步），第 2 轮涨到 lr，第 3 轮保持 lr。
    lrs = [record["lr"] for record in result["history"]]
    assert lrs == pytest.approx([config.train.lr / 2, config.train.lr, config.train.lr])


def test_cosine_scheduler_decays_learning_rate(config: Config) -> None:
    # 5 轮、前 2 轮 warmup，之后余弦衰减到 lr_min；末轮正好落在 lr_min。
    config = replace(
        config,
        train=replace(
            config.train,
            epochs=5,
            warmup_epochs=2,
            lr_scheduler="cosine",
            lr_min=config.train.lr * 0.1,
        ),
    )
    train_loader, val_loader = _loaders(config)

    result = fit(config, build_model(config), train_loader, val_loader, torch.device("cpu"))

    lr, lr_min = config.train.lr, config.train.lr_min
    # 第 1 轮 lr/2、第 2 轮 lr（预热完成）、第 3 轮仍是峰值 lr（余弦起点）、
    # 第 4 轮走到半个余弦周期、第 5 轮落到 lr_min。
    expected = [lr / 2, lr, lr, lr_min + (lr - lr_min) * 0.5, lr_min]
    lrs = [record["lr"] for record in result["history"]]
    assert lrs == pytest.approx(expected)


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


def test_fit_raises_when_all_valid_auc_are_nan(config: Config, monkeypatch) -> None:
    """valid 全部类别都退化时不能静默保存随机初始权重，必须让训练失败。"""
    train_loader, val_loader = _loaders(config)
    monkeypatch.setattr(
        engine,
        "evaluate",
        lambda *args, **kwargs: {
            "loss": 0.5,
            "macro_auc": float("nan"),
            "macro_f1": 0.5,
        },
    )

    with pytest.raises(RuntimeError, match="AUC"):
        fit(config, build_model(config), train_loader, val_loader, torch.device("cpu"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")
def test_fit_with_pos_weight_on_cuda(config: Config) -> None:
    """回归：pos_weight 必须和数据/模型在同一块 CUDA 上，否则训练第一步就会 device mismatch。"""
    config = replace(
        config,
        train=replace(
            config.train,
            pos_weight=True,
            epochs=1,
            warmup_epochs=0,
            early_stop_patience=0,
        ),
    )
    train_loader, val_loader = _loaders(config)
    device = torch.device("cuda")
    model = build_model(config).to(device)

    result = fit(config, model, train_loader, val_loader, device)

    assert math.isfinite(result["history"][0]["train_loss"])
    assert math.isfinite(result["history"][0]["valid_loss"])
