from dataclasses import replace

import pytest
import torch

from common.config import load_config
from train import engine


def test_d006_configs_and_scheduler():
    configs = [load_config(f'configs/exp_D006{x}.yaml') for x in 'abc']
    assert all(c.train.early_stop_monitor == 'ema_auc' for c in configs)
    for cfg, expected in [(configs[0], [150, 300, 300, 272.307, 199.807, 110.193, 37.693, 10]),
                          (configs[1], [150, 300, 300, 227.5, 82.5, 10, 10, 10])]:
        model = torch.nn.Linear(1, 1)
        opt = engine._build_optimizer(model, cfg)
        scheduler = engine._build_scheduler(opt, cfg)
        observed = []
        for _ in range(8):
            observed.append(opt.param_groups[0]['lr'] * 1e6)
            opt.step(); scheduler.step()
        assert observed == pytest.approx(expected, abs=0.001)


def test_mixup_preserves_views_and_mixup_is_symmetric(monkeypatch):
    from train.study_mixup import mixup_studies
    monkeypatch.setattr(torch.distributions.Beta, 'sample', lambda self: torch.tensor(.25))
    images = torch.tensor([1., 10., 20., 2.]).reshape(4, 1, 1, 1)
    labels = torch.tensor([[1., 0.], [0., 1.]])
    result, target, count = mixup_studies(images, torch.tensor([0, 2, 4]), labels,
                                           torch.tensor([0, 1, 1, 0]), alpha=.2, probability=1.)
    assert result.flatten().tolist() == pytest.approx([1.75, 17.5, 12.5, 1.25]) or result.flatten().tolist() == pytest.approx([1.25, 12.5, 17.5, 1.75])
    assert sorted(target[:, 0].tolist()) == pytest.approx([.25, .75])
    assert images.flatten().tolist() == [1, 10, 20, 2]
    assert count == 2


def test_invalid_combinations():
    cfg = load_config('configs/exp_D006c.yaml')
    with pytest.raises(ValueError):
        replace(cfg, train=replace(cfg.train, loss='asl', pos_weight=False))
    with pytest.raises(ValueError):
        replace(cfg.train, early_stop_monitor='ema_auc', ema_enabled=False)
    with pytest.raises(ValueError):
        replace(cfg.model, drop_path_rate=float('nan'))


def test_mixup_unknown_odd_and_rng_restoration():
    from train.study_mixup import mixup_studies
    images = torch.arange(5., dtype=torch.float32).reshape(5, 1, 1, 1)
    labels = torch.eye(5)
    offsets = torch.arange(6)
    views = torch.tensor([0, 0, 0, 2, 1])
    state = torch.get_rng_state()
    first = mixup_studies(images, offsets, labels, views, alpha=.2, probability=1)
    torch.set_rng_state(state)
    second = mixup_studies(images, offsets, labels, views, alpha=.2, probability=1)
    assert torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])
    assert first[2] == 2
    assert torch.equal(first[0][3:], images[3:])
    state = torch.get_rng_state()
    untouched = mixup_studies(images, offsets, labels, views, alpha=.2, probability=0)
    assert untouched[0] is images and torch.equal(state, torch.get_rng_state())


def test_real_droppath_and_old_checkpoint():
    from model.network import build_model
    cfg = load_config('configs/exp_D006a.yaml')
    cfg = replace(cfg, model=replace(cfg.model, pretrained=False))
    torch.manual_seed(3)
    model = build_model(cfg).eval()
    rates = [m.drop_prob for m in model.modules() if hasattr(m, 'drop_prob')]
    assert max(rates) == pytest.approx(.1)
    old = build_model(replace(cfg, model=replace(cfg.model, drop_path_rate=0)))
    model.load_state_dict(old.state_dict())
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        assert torch.equal(model(x), model(x))


def test_mixup_training_batch_has_backward():
    from train.study_mixup import mixup_studies
    images = torch.tensor([[1.], [2.]])
    labels = torch.tensor([[1.], [0.]])
    images, labels, count = mixup_studies(images, torch.tensor([0, 1, 2]), labels, torch.tensor([0, 0]), alpha=.2, probability=1)
    layer = torch.nn.Linear(1, 1)
    loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.]))(layer(images), labels)
    loss.backward()
    assert count == 2 and torch.isfinite(loss) and torch.isfinite(layer.weight.grad).all()


@pytest.mark.parametrize('monitor, expected', [('raw_auc', 3), ('ema_auc', 5)])
def test_monitor_tracks_selected_series(monkeypatch, monitor, expected):
    from torch import nn
    cfg = load_config('configs/exp_D006a.yaml')
    cfg = replace(cfg, train=replace(cfg.train, early_stop_monitor=monitor, early_stop_patience=2, pos_weight=False))
    monkeypatch.setattr(engine, '_train_labels', lambda _: None)
    monkeypatch.setattr(engine, 'build_loss', lambda *a, **k: nn.MSELoss())
    monkeypatch.setattr(engine, 'train_one_epoch', lambda *a, **k: 1.)
    raw = iter([.80, .79, .78, .77, .76])
    ema = iter([.79, .80, .81, .81, .80])
    calls = iter(sum(([r, e] for r, e in zip(raw, ema)), []))
    monkeypatch.setattr(engine, 'evaluate', lambda *a, **k: {'loss': 1., 'macro_auc': next(calls), 'macro_f1': .5})
    result = engine.fit(cfg, nn.Linear(1, 1), None, None, torch.device('cpu'))
    assert result['progress'].epoch == expected
    assert result['history'][-1]['stale_epochs'] == 2
    assert result['progress'].monitor == monitor


def test_monitor_resume_preserves_stop_epoch(monkeypatch):
    from copy import deepcopy
    from torch import nn
    cfg = load_config('configs/exp_D006a.yaml')
    cfg = replace(cfg, train=replace(cfg.train, early_stop_patience=2, pos_weight=False))
    monkeypatch.setattr(engine, '_train_labels', lambda _: None)
    monkeypatch.setattr(engine, 'build_loss', lambda *a, **k: nn.MSELoss())
    def train(config, model, loader, criterion, optimizer, scaler, device, **kwargs):
        optimizer.zero_grad()
        model(torch.ones(1, 1)).sum().backward()
        optimizer.step()
        if kwargs.get('ema') is not None:
            kwargs['ema'].update(model)
        return 1.
    monkeypatch.setattr(engine, 'train_one_epoch', train)
    values = iter([.8, .79, .79, .8, .78, .81, .77, .81, .76, .8])
    monkeypatch.setattr(engine, 'evaluate', lambda *a, **k: {'loss': 1., 'macro_auc': next(values), 'macro_f1': .5})
    model = nn.Linear(1, 1)
    saved = {}
    class Interrupted(Exception): pass
    def stop(progress, runtime):
        if progress.epoch == 3:
            saved.update(progress=deepcopy(progress), optimizer=deepcopy(runtime.optimizer.state_dict()), scheduler=deepcopy(runtime.scheduler.state_dict()))
            raise Interrupted
    with pytest.raises(Interrupted):
        engine.fit(cfg, model, None, None, torch.device('cpu'), on_epoch_end=stop)
    result = engine.fit(cfg, model, None, None, torch.device('cpu'), progress=saved['progress'], optimizer_state=saved['optimizer'], scheduler_state=saved['scheduler'])
    assert result['progress'].epoch == 5
    assert result['progress'].stale_epochs == 2
    with pytest.raises(ValueError, match='监控'):
        engine.fit(replace(cfg, train=replace(cfg.train, early_stop_monitor='raw_auc')), model, None, None, torch.device('cpu'), progress=saved['progress'])


@pytest.mark.parametrize('name,value', [('mixup_alpha', float('inf')), ('mixup_alpha', -1), ('mixup_prob', float('nan')), ('mixup_prob', 1.1), ('cosine_end_epoch', 3), ('cosine_end_epoch', 9)])
def test_reject_invalid_training_options(name, value):
    cfg = load_config('configs/exp_D006c.yaml')
    with pytest.raises(ValueError):
        replace(cfg.train, **{name: value})


def test_cosine_resume_matches_uninterrupted():
    from copy import deepcopy
    cfg = load_config('configs/exp_D006b.yaml')
    cfg = replace(cfg, train=replace(cfg.train, head_lr=.0006))
    def optimizer():
        model = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Linear(1, 1))
        return torch.optim.AdamW([{'params': model[0].parameters(), 'lr': cfg.train.lr}, {'params': model[1].parameters(), 'lr': cfg.train.head_lr}])
    opt = optimizer()
    sched = engine._build_scheduler(opt, cfg)
    expected = []
    for step in range(8):
        expected.append([g['lr'] for g in opt.param_groups])
        opt.step(); sched.step()
        if step == 2:
            state = deepcopy(opt.state_dict()), deepcopy(sched.state_dict())
    resumed = optimizer()
    resumed_sched = engine._build_scheduler(resumed, cfg)
    resumed.load_state_dict(state[0]); resumed_sched.load_state_dict(state[1])
    for expected_pair in expected[3:]:
        assert [g['lr'] for g in resumed.param_groups] == pytest.approx(expected_pair)
        resumed.step(); resumed_sched.step()


def test_mixup_integrates_with_training_and_records_actual_coverage():
    cfg = load_config('configs/exp_D006c.yaml')
    cfg = replace(cfg, train=replace(cfg.train, mixup_prob=1., amp=False))
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(1, 10))
    batch = {'images': torch.tensor([[[[1.]]], [[[2.]]], [[[3.]]]]),
             'offsets': torch.tensor([0, 1, 2, 3]), 'labels': torch.zeros(3, 10),
             'views': torch.tensor([0, 0, 2]), 'study_ids': ['a', 'b', 'c']}
    batch['labels'][0, 0] = 1.
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    scaler = torch.amp.GradScaler('cpu', enabled=False)
    before = model[1].weight.detach().clone()
    stats = {}
    loss = engine.train_one_epoch(cfg, model, [batch], torch.nn.BCEWithLogitsLoss(), optimizer, scaler, torch.device('cpu'), loss_stats=stats)
    assert torch.isfinite(torch.tensor(loss)) and not torch.equal(before, model[1].weight)
    assert stats == {'mixup_mixed_studies': 2, 'mixup_total_studies': 3, 'mixup_fraction': pytest.approx(2 / 3)}
