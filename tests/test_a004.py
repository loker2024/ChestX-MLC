"""A-004 的配置和分类头 dropout 行为。"""
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from common.config import ModelConfig, config_from_dict, load_config
from model.network import build_model


CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_a004_changes_only_two_parameters():
    baseline = load_config(CONFIGS / "exp_A001.yaml").to_dict()
    candidate = load_config(CONFIGS / "exp_A004.yaml").to_dict()
    assert baseline["model"]["dropout"] == 0.0
    assert candidate["model"].pop("dropout") == 0.2
    baseline["model"].pop("dropout")
    assert baseline["train"].pop("weight_decay") == 0.01
    assert candidate["train"].pop("weight_decay") == 0.05
    assert candidate == baseline


def test_dropout_is_applied_and_checkpoint_config_roundtrips():
    config = load_config(CONFIGS / "exp_A004.yaml")
    config = replace(config, model=replace(config.model, pretrained=False))
    restored = config_from_dict(config.to_dict())
    model = build_model(restored)
    dropout = model.head.drop
    assert dropout.p == pytest.approx(0.2)
    x = torch.ones(1024)
    model.train()
    assert not torch.equal(dropout(x), x)
    model.eval()
    assert torch.equal(dropout(x), x)
    legacy = config.to_dict()
    del legacy["model"]["dropout"]
    assert config_from_dict(legacy).model.dropout == 0.0


@pytest.mark.parametrize("value", [-0.1, 1.0, float("nan")])
def test_invalid_dropout_rejected(value):
    with pytest.raises(ValueError):
        ModelConfig(dropout=value)


def test_xrv_dropout_is_not_silently_ignored():
    with pytest.raises(ValueError):
        ModelConfig(arch="xrv_densenet121", dropout=0.2)
