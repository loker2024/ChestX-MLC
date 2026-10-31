from dataclasses import replace
from pathlib import Path
import pytest
import torch
from common.checkpoint import save_checkpoint
from common.config import ModelConfig, config_from_dict, load_config
from infer.predict import load_checkpoint_model
from model.network import build_model

ROOT = Path(__file__).resolve().parents[1]


def test_config_and_experiment():
    base = load_config(ROOT / 'configs/exp_A001.yaml').to_dict()
    exp = load_config(ROOT / 'configs/exp_ml_decoder.yaml').to_dict()
    assert exp['model']['head'] == 'ml_decoder'
    exp['model']['head'] = 'standard'
    assert exp == base
    assert config_from_dict(base).model.head == 'standard'
    with pytest.raises(ValueError):
        ModelConfig(head='ml_decoder', backbone='resnet18')
    with pytest.raises(ValueError):
        ModelConfig(head='ml_decoder', backbone='convnext_tiny.fb_in1k', freeze_backbone=True)


def test_decoder_gradients_and_checkpoint(tmp_path):
    cfg = load_config(ROOT / 'configs/exp_ml_decoder.yaml')
    cfg = replace(cfg, model=replace(cfg.model, pretrained=False))
    model = build_model(cfg).train()
    images = torch.randn(2, 3, 64, 64)
    logits = model(images)
    assert logits.shape == (2, 10)
    torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits)).backward()
    assert model.stem[0].weight.grad is not None
    assert model.head.decoder.query.grad is not None
    model.eval()
    with torch.no_grad():
        expected = model(images)
    path = save_checkpoint(tmp_path / 'best.pt', model, config=cfg)
    loaded = load_checkpoint_model(cfg, path, torch.device('cpu'))
    with torch.no_grad():
        torch.testing.assert_close(loaded(images), expected)
    with pytest.raises(ValueError, match='head'):
        load_checkpoint_model(replace(cfg, model=replace(cfg.model, head='standard')), path, torch.device('cpu'))
