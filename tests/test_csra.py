"""CSRA 分类头的空间注意力、配置和 checkpoint 行为。"""
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from common.checkpoint import save_checkpoint
from common.config import ModelConfig, config_from_dict, load_config
from infer.predict import load_checkpoint_model
from model.network import build_model

ROOT = Path(__file__).resolve().parents[1]


def test_csra_computes_class_specific_spatial_attention():
    from model.csra import CSRAHead

    head = CSRAHead(2, 2, residual_weight=0.4, temperature=1.0)
    with torch.no_grad():
        head.classifier.weight.copy_(torch.tensor([[[[1.]], [[0.]]], [[[0.]], [[1.]]]]))
    features = torch.tensor([[[[1., 3.]], [[4., 0.]]]])
    expected = torch.tensor([[2.0 + 0.4 * (1 + 3 * torch.exp(torch.tensor(2.))) / (1 + torch.exp(torch.tensor(2.))),
                              2.0 + 0.4 * (4 * torch.exp(torch.tensor(4.))) / (1 + torch.exp(torch.tensor(4.)))]])
    torch.testing.assert_close(head(features), expected)


def test_config_legacy_and_invalid_options():
    cfg = load_config(ROOT / 'configs/exp_A001.yaml')
    raw = cfg.to_dict()
    raw['model'].pop('head', None)
    raw['model'].pop('csra_residual_weight', None)
    raw['model'].pop('csra_temperature', None)
    assert config_from_dict(raw).model.head == 'standard'
    with pytest.raises(ValueError):
        ModelConfig(head='unknown')
    with pytest.raises(ValueError):
        ModelConfig(head='csra', backbone='resnet18')
    with pytest.raises(ValueError):
        ModelConfig(head='csra', backbone='convnext_tiny.fb_in1k', freeze_backbone=True)
    with pytest.raises(ValueError):
        ModelConfig(head='csra', backbone='convnext_tiny.fb_in1k', csra_temperature=float('nan'))


def test_experiment_changes_only_head_settings():
    base = load_config(ROOT / 'configs/exp_A001.yaml').to_dict()
    exp = load_config(ROOT / 'configs/exp_csra.yaml').to_dict()
    assert exp['model']['head'] == 'csra'
    exp['model']['head'] = 'standard'
    assert exp == base


def test_convnext_csra_gradients_and_checkpoint_roundtrip(tmp_path):
    cfg = load_config(ROOT / 'configs/exp_csra.yaml')
    cfg = replace(cfg, model=replace(cfg.model, pretrained=False))
    model = build_model(cfg)
    model.train()
    images = torch.randn(2, 3, 64, 64)
    logits = model(images)
    assert logits.shape == (2, 10)
    torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits)).backward()
    assert model.stem[0].weight.grad is not None
    assert model.head.csra.classifier.weight.grad is not None
    model.eval()
    with torch.no_grad():
        expected = model(images)
    path = save_checkpoint(tmp_path / 'csra.pt', model, config=cfg)
    restored = load_checkpoint_model(cfg, path, torch.device('cpu'))
    with torch.no_grad():
        torch.testing.assert_close(restored(images), expected)
    wrong = replace(cfg, model=replace(cfg.model, csra_temperature=2.0))
    with pytest.raises(ValueError, match='csra_temperature'):
        load_checkpoint_model(wrong, path, torch.device('cpu'))


def test_pretrained_backbone_and_norm_preserved(monkeypatch):
    import timm
    from model import network

    cfg = load_config(ROOT / 'configs/exp_csra.yaml')
    source = timm.create_model(cfg.model.backbone, pretrained=False, num_classes=10)
    with torch.no_grad():
        source.stem[0].weight.fill_(0.125)
        source.head.norm.weight.fill_(1.5)
    monkeypatch.setattr(network.timm, 'create_model', lambda *args, **kwargs: source)
    result = build_model(cfg)
    assert result.pretrained_loaded is True
    assert torch.all(result.stem[0].weight == 0.125)
    assert torch.all(result.head.norm.weight == 1.5)
    assert result.head.csra.classifier.weight.requires_grad


def test_pretrained_failure_does_not_silently_fallback(monkeypatch):
    from model import network

    cfg = load_config(ROOT / 'configs/exp_csra.yaml')
    def fail(*args, **kwargs):
        raise OSError('missing weights')
    monkeypatch.setattr(network.timm, 'create_model', fail)
    with pytest.raises(RuntimeError, match='预训练骨干加载失败'):
        build_model(cfg)


def test_csra_train_step_preserves_study_aggregation():
    from model.network import aggregate_logits

    cfg = load_config(ROOT / 'configs/exp_csra.yaml')
    cfg = replace(cfg, model=replace(cfg.model, pretrained=False))
    model = build_model(cfg).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
    initial_head = model.head.csra.classifier.weight.detach().clone()
    initial_backbone = model.stem[0].weight.detach().clone()
    logits = model(torch.randn(3, 3, 64, 64))
    study = aggregate_logits(logits, torch.tensor([0, 2, 3]), torch.tensor([0, 1, 0]))
    torch.testing.assert_close(study[0], logits[0] * 0.7 + logits[1] * 0.3)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(study, torch.ones_like(study), pos_weight=torch.full((10,), 2.0))
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    assert not torch.equal(initial_head, model.head.csra.classifier.weight)
    assert not torch.equal(initial_backbone, model.stem[0].weight)


def test_csra_synthetic_training_and_inference(tmp_path, labeled_split, unlabeled_split, write_config):
    from data.dataset import build_loader
    from infer.predict import predict_loader
    from train.engine import fit

    path = write_config({
        'train': (labeled_split['csv'], labeled_split['images']),
        'valid': (labeled_split['csv'], labeled_split['images']),
        'test': (unlabeled_split['csv'], unlabeled_split['images']),
    })
    cfg = load_config(path)
    cfg = replace(cfg, model=replace(cfg.model, backbone='convnext_tiny.fb_in1k', head='csra', pretrained=False,
                                      aggregation='view_weighted'),
                  train=replace(cfg.train, epochs=1, warmup_epochs=0, early_stop_patience=0))
    device = torch.device('cpu')
    model = build_model(cfg)
    result = fit(cfg, model, build_loader(cfg, 'train', train=True), build_loader(cfg, 'valid'), device)
    assert torch.isfinite(torch.tensor(result['history'][0]['train_loss']))
    model.load_state_dict(result['best_state'])
    path = save_checkpoint(tmp_path / 'best.pt', model, config=cfg)
    predictions = predict_loader(load_checkpoint_model(cfg, path, device), build_loader(cfg, 'test'), device)
    assert len(predictions) == len(build_loader(cfg, 'test').dataset)
    assert predictions[[f'p{i}' for i in range(1, 11)]].to_numpy().min() >= 0
    assert predictions[[f'p{i}' for i in range(1, 11)]].to_numpy().max() <= 1
