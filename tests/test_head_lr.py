"""差异学习率的配置、参数归属、调度和续训。"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import timm
import torch
from torch import nn

import run_train
from common.config import load_config
from data.dataset import build_loader
from model.network import build_model
from train import engine


@pytest.mark.parametrize("suffix", "abcd")
def test_c001_config_and_real_classifier_groups(suffix):
    cfg = load_config(Path(f"configs/exp_C001{suffix}.yaml"))
    model = timm.create_model(cfg.model.backbone, pretrained=False, num_classes=10)
    optimizer = engine._build_optimizer(model, cfg)
    groups = optimizer.param_groups
    assert len(groups) == 2
    assert groups[0]["lr"] == pytest.approx(0.00005)
    assert groups[1]["lr"] == pytest.approx(0.00015)
    head = {id(p) for p in model.get_classifier().parameters() if p.requires_grad}
    backbone = {id(p) for p in groups[0]["params"]}
    actual_head = {id(p) for p in groups[1]["params"]}
    assert head and head == actual_head and not head & backbone
    assert head | backbone == {id(p) for p in model.parameters() if p.requires_grad}
    scheduler = engine._build_scheduler(optimizer, cfg)
    lrs = []
    for _ in range(12):
        lrs.append([group["lr"] for group in groups])
        optimizer.step()
        scheduler.step()
    assert lrs[0] == pytest.approx([0.000025, 0.000075])
    assert lrs[1] == pytest.approx([0.00005, 0.00015])
    assert lrs[2] == pytest.approx([0.00005, 0.00015])
    assert lrs[-1] == pytest.approx([0.000005, 0.000015])
    assert all(head_lr == pytest.approx(3 * base_lr) for base_lr, head_lr in lrs)


@pytest.mark.parametrize("value", [0.0, -1.0, float('inf'), float('nan')])
def test_head_lr_invalid(value):
    cfg = load_config('configs/exp_A001.yaml')
    with pytest.raises(ValueError, match='head_lr'):
        replace(cfg.train, head_lr=value)


def test_unsupported_classifier_fails():
    cfg = load_config('configs/exp_C001a.yaml')
    with pytest.raises(ValueError, match='分类头'):
        engine._build_optimizer(nn.Linear(3, 2), cfg)


def test_resume_config_rejects_changed_head_lr():
    cfg = load_config('configs/exp_C001a.yaml')
    with pytest.raises(ValueError, match='train.head_lr'):
        run_train._validate_resume_config(replace(cfg, train=replace(cfg.train, head_lr=0.0002)), cfg)


def test_two_groups_with_real_loader_and_training(tmp_path, labeled_split, write_config):
    split = (labeled_split['csv'], labeled_split['images'])
    cfg = load_config(write_config({'train': split, 'valid': split, 'test': split}))
    cfg = replace(cfg, model=replace(cfg.model, pretrained=False),
                  train=replace(cfg.train, epochs=1, warmup_epochs=0, head_lr=0.0006))
    model = build_model(cfg)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    result = engine.fit(cfg, model, build_loader(cfg, 'train', train=True),
                        build_loader(cfg, 'valid'), torch.device('cpu'))
    record = result['history'][0]
    assert record['lr'] == pytest.approx(cfg.train.lr)
    assert record['head_lr'] == pytest.approx(0.0006)
    assert any(not torch.equal(before[name], value) for name, value in model.state_dict().items())
    path = tmp_path / 'metrics.csv'
    run_train._write_metrics(path, result['history'])
    assert 'head_lr' in path.read_text(encoding='utf-8').splitlines()[0]


def test_resume_two_groups_preserves_training_and_logging(monkeypatch):
    cfg = load_config('configs/exp_C001a.yaml')
    cfg = replace(cfg, train=replace(cfg.train, epochs=4, early_stop_patience=0, pos_weight=False, ema_enabled=False))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(2, 2)
            self.head = nn.Linear(2, 1)
        def get_classifier(self):
            return self.head
        def forward(self, x):
            return self.head(self.backbone(x))

    monkeypatch.setattr(engine, '_train_labels', lambda _: None)
    monkeypatch.setattr(engine, 'build_loss', lambda *a, **k: nn.MSELoss())
    def train(config, model, loader, criterion, optimizer, scaler, device):
        optimizer.zero_grad()
        loss = criterion(model(torch.ones(1, 2)), torch.ones(1, 1))
        loss.backward()
        optimizer.step()
        return loss.item()
    monkeypatch.setattr(engine, 'train_one_epoch', train)
    monkeypatch.setattr(engine, 'evaluate', lambda *a, **k: {'loss': 1.0, 'macro_auc': 0.5, 'macro_f1': 0.5})

    class Writer:
        def __init__(self): self.events = []
        def add_scalar(self, *args): self.events.append(args)
        def flush(self): pass

    torch.manual_seed(2)
    full_model = Model()
    full = engine.fit(cfg, full_model, None, None, torch.device('cpu'))
    torch.manual_seed(2)
    model = Model()
    saved = {}
    class Stop(Exception): pass
    def stop(progress, runtime):
        if progress.epoch == 2:
            saved.update(progress=deepcopy(progress), optimizer=deepcopy(runtime.optimizer.state_dict()), scheduler=deepcopy(runtime.scheduler.state_dict()))
            raise Stop()
    writer = Writer()
    with pytest.raises(Stop):
        engine.fit(cfg, model, None, None, torch.device('cpu'), on_epoch_end=stop, writer=writer)
    resumed = engine.fit(cfg, model, None, None, torch.device('cpu'), progress=saved['progress'], optimizer_state=saved['optimizer'], scheduler_state=saved['scheduler'], writer=writer)
    for a, b in zip(full['history'], resumed['history']):
        assert a['lr'] == pytest.approx(b['lr'])
        assert a['head_lr'] == pytest.approx(b['head_lr'])
    for name, value in full_model.state_dict().items():
        assert torch.equal(value, model.state_dict()[name])
    # 回调模拟中断发生在曲线写入前，第二轮 checkpoint 已保存但曲线缺失。
    assert [e[2] for e in writer.events if e[0] == 'train/head_lr'] == [1, 3, 4]
