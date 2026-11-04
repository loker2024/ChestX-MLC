import torch
from torch import nn

from common.config import ModelConfig, TrainConfig
from model.study_attention import StudyAttention
from train.ema import ModelEMA


def test_ema_updates_and_restores_buffers():
    model = nn.Sequential(nn.Linear(1, 1, bias=False), nn.BatchNorm1d(1))
    ema = ModelEMA(model, 0.5)
    old = ema.model[0].weight.detach().clone()
    with torch.no_grad():
        model[0].weight.add_(2)
        model[1].num_batches_tracked.fill_(3)
    ema.update(model)
    assert torch.allclose(ema.model[0].weight, old + 1)
    assert ema.model[1].num_batches_tracked.item() == 3
    clone = ModelEMA(model, 0.5)
    clone.load_state_dict(ema.state_dict())
    assert clone.updates == 1
    assert torch.equal(clone.model[0].weight, ema.model[0].weight)


def test_attention_is_per_study_and_differentiable():
    module = StudyAttention(4, hidden_dim=3)
    x = torch.randn(3, 4, requires_grad=True)
    out = module(x, torch.tensor([0, 1, 3]), torch.tensor([0, 1, 2]))
    assert out.shape == (2, 4)
    assert torch.allclose(out[0], x[0])
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_configuration_defaults_and_switches():
    assert TrainConfig().ema_enabled
    assert not TrainConfig(ema_enabled=False).ema_enabled
    assert ModelConfig(backbone='convnext_tiny.fb_in1k', aggregation='study_attention').uses_views


def test_attention_real_convnext_train_infer_checkpoint(tmp_path):
    from dataclasses import replace
    from common.config import load_config
    from common.checkpoint import save_checkpoint, load_checkpoint
    from model.network import build_model, forward_studies
    from infer.predict import _forward_logits

    config = load_config("configs/default.yaml")
    config = replace(config, model=replace(config.model, backbone='convnext_tiny.fb_in1k', pretrained=False, aggregation='study_attention'))
    model = build_model(config).eval()
    images = torch.randn(3, 3, 32, 32)
    offsets = torch.tensor([0, 1, 3])
    views = torch.tensor([0, 1, 2])
    with torch.no_grad():
        expected = forward_studies(model, images, offsets, views)
        actual = _forward_logits(model, images, offsets, torch.device('cpu'), views)
    assert expected.shape == (2, 10)
    assert torch.allclose(expected, actual)
    model.zero_grad()
    forward_studies(model, images, offsets, views).sum().backward()
    assert model.attention.score.weight.grad is not None
    assert torch.isfinite(model.attention.score.weight.grad).all()
    path = save_checkpoint(tmp_path / 'best.pt', model, config=config)
    restored = build_model(config).eval()
    restored.load_state_dict(load_checkpoint(path)['model'])
    with torch.no_grad():
        assert torch.allclose(expected, forward_studies(restored, images, offsets, views))


def test_ema_progress_checkpoint_roundtrip(tmp_path):
    from common.config import load_config
    from common.checkpoint import load_checkpoint
    from run_train import _progress_from_checkpoint, _save_last_checkpoint
    from train.engine import TrainProgress, TrainRuntime
    model = nn.Linear(1, 1)
    ema = ModelEMA(model, 0.5)
    ema.update(model)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
    scaler = torch.amp.GradScaler('cpu', enabled=False)
    state = {k: v.clone() for k, v in ema.model.state_dict().items()}
    progress = TrainProgress(epoch=1, ema_state={'model': state, 'updates': ema.updates}, ema_best_epoch=1, ema_best_auc=0.8, ema_best_state=state)
    path = tmp_path / 'last.pt'
    _save_last_checkpoint(path, model, progress, TrainRuntime(optimizer, scheduler, scaler), load_config("configs/default.yaml"), torch.device('cpu'), False)
    restored = _progress_from_checkpoint(load_checkpoint(path))
    assert restored.ema_state['updates'] == 1
    assert restored.ema_best_epoch == 1
    assert torch.equal(restored.ema_best_state['weight'], state['weight'])
