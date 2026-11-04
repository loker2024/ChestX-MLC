"""Study 级 AUCM 的采样、损失和优化器状态契约。"""

import copy

import numpy as np
import pytest
import torch

pytest.importorskip("libauc", reason="D-005 可选依赖未安装")

from data.aucm_sampler import AUCMBatchSampler
from model.losses import AUCMLoss
from train.engine import build_aucm_optimizer


def test_experiment_configs_are_independent_and_require_correct_initialization():
    from common.config import load_config
    first = load_config('configs/exp_D005a.yaml')
    second = load_config('configs/exp_D005b.yaml')
    assert first.train.loss == second.train.loss == 'aucm'
    assert not first.train.pos_weight and not second.train.pos_weight
    assert not first.train.aucm_require_bce_init and second.train.aucm_require_bce_init
    a, b = first.to_dict(), second.to_dict()
    b['train']['aucm_require_bce_init'] = False
    assert a == b


def test_second_stage_requires_valid_bce_checkpoint(tmp_path):
    from common.config import load_config
    import run_train
    cfg = load_config('configs/exp_D005b.yaml')
    with pytest.raises(ValueError, match='--init-from'):
        run_train.run_training(cfg, torch.device('cpu'), tmp_path)
    with pytest.raises(ValueError, match='BCE'):
        run_train.run_training(cfg, torch.device('cpu'), tmp_path,
                               init_from=(tmp_path / 'x.pt', {'config': cfg.to_dict()}, 'abc'))


def test_sampler_balances_selected_task_and_is_epoch_reproducible():
    labels = np.array([[i % 2, (i // 2) % 2] for i in range(20)])
    sampler = AUCMBatchSampler(labels, batch_size=4, seed=42)
    sampler.set_epoch(2)
    batches = list(sampler)
    sampler.set_epoch(2)
    assert batches == list(sampler)
    assert {batch[0][1] for batch in batches} == {0, 1}
    for batch in batches:
        task = batch[0][1]
        assert all(t == task for _, t in batch)
        assert sum(labels[i, task] for i, _ in batch) == 2


def test_sampler_rejects_missing_classes():
    with pytest.raises(ValueError, match="正.*负"):
        AUCMBatchSampler(np.zeros((10, 2)), batch_size=4, seed=42)


def test_aucm_state_restores_training_update():
    torch.manual_seed(4)
    model = torch.nn.Linear(2, 2)
    criterion = AUCMLoss(num_labels=2, device=torch.device('cpu'))
    opt = build_aucm_optimizer(model, criterion, lr=0.001, weight_decay=0.0)
    x = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [0., 0.]])
    y = torch.tensor([[1., 0.], [0., 1.], [1., 0.], [0., 1.]])

    def step(m, c, o):
        o.zero_grad(set_to_none=True)
        c.clear_dual_grad()
        c(m(x), y, torch.tensor([0])).backward()
        o.step()

    step(model, criterion, opt)
    assert criterion.alpha[1].item() == 0
    assert criterion.a[1].item() == 0
    assert criterion.b[1].item() == 0
    model_state = copy.deepcopy(model.state_dict())
    loss_state = copy.deepcopy(criterion.state_dict())
    opt_state = copy.deepcopy(opt.state_dict())
    step(model, criterion, opt)

    restored = torch.nn.Linear(2, 2)
    restored.load_state_dict(model_state)
    other = AUCMLoss(num_labels=2, device=torch.device('cpu'))
    other.load_state_dict(loss_state)
    other_opt = build_aucm_optimizer(restored, other, lr=0.001, weight_decay=0.0)
    other_opt.load_state_dict(opt_state)
    step(restored, other, other_opt)
    for a, b in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(a, b)
    for key, value in criterion.state_dict().items():
        torch.testing.assert_close(value, other.state_dict()[key])


def test_aucm_validation_does_not_change_training_state():
    from dataclasses import replace
    from common.config import load_config
    from train.engine import evaluate
    cfg = load_config('configs/exp_D005a.yaml')
    cfg = replace(cfg, model=replace(cfg.model, aggregation='mean'))
    criterion = AUCMLoss(num_labels=10, device=torch.device('cpu'))
    state = copy.deepcopy(criterion.state_dict())
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 10))
    labels = torch.tensor([[1.] * 10, [0.] * 10])
    batch = {'images': torch.rand(2, 3, 2, 2), 'labels': labels, 'offsets': torch.tensor([0, 1, 2]),
             'study_ids': ['1', '2']}
    metrics = evaluate(cfg, model, [batch], criterion, torch.device('cpu'))
    assert np.isfinite(metrics['loss'])
    for key, value in state.items():
        torch.testing.assert_close(value, criterion.state_dict()[key])


def test_fit_resume_matches_uninterrupted_aucm():
    from dataclasses import replace
    from types import SimpleNamespace
    from torch.utils.data import DataLoader, Dataset
    from common.config import load_config
    from data.dataset import collate_studies
    from train.engine import fit

    class DatasetOfStudies(Dataset):
        def __init__(self):
            self.records = [SimpleNamespace(labels=np.full(10, i % 2, dtype=np.float32)) for i in range(4)]
        def __len__(self):
            return 4
        def __getitem__(self, index):
            task = None
            if isinstance(index, tuple):
                index, task = index
            item = {'images': torch.full((1, 3, 2, 2), float(index)), 'views': torch.tensor([0]),
                    'study_id': str(index), 'labels': torch.from_numpy(self.records[index].labels)}
            if task is not None:
                item['task_id'] = task
            return item

    ds = DatasetOfStudies()
    labels = np.stack([r.labels for r in ds.records])
    train = DataLoader(ds, batch_sampler=AUCMBatchSampler(labels, 4, 42), collate_fn=collate_studies)
    valid = DataLoader(ds, batch_size=4, collate_fn=collate_studies)
    cfg = load_config('configs/default.yaml')
    cfg = replace(cfg, model=replace(cfg.model, aggregation='mean'), train=replace(cfg.train,
        loss='aucm', pos_weight=False, head_lr=None, epochs=2, warmup_epochs=0,
        batch_studies=4, amp=False, ema_enabled=False, early_stop_patience=0))

    def model():
        return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 10))
    torch.manual_seed(5)
    full_model = model()
    torch.manual_seed(7)
    fit(cfg, full_model, train, valid, torch.device('cpu'))
    torch.manual_seed(5)
    part_model = model()
    captured = {}
    class Stop(Exception): pass
    def stop(progress, runtime):
        if progress.epoch == 1:
            captured.update(progress=copy.deepcopy(progress), optimizer=copy.deepcopy(runtime.optimizer.state_dict()),
                scheduler=copy.deepcopy(runtime.scheduler.state_dict()), criterion=copy.deepcopy(runtime.criterion.state_dict()),
                rng=torch.get_rng_state().clone())
            raise Stop()
    torch.manual_seed(7)
    with pytest.raises(Stop):
        fit(cfg, part_model, train, valid, torch.device('cpu'), on_epoch_end=stop)
    torch.set_rng_state(captured['rng'])
    fit(cfg, part_model, train, valid, torch.device('cpu'), progress=captured['progress'],
        optimizer_state=captured['optimizer'], scheduler_state=captured['scheduler'], criterion_state=captured['criterion'])
    for expected, actual in zip(full_model.parameters(), part_model.parameters()):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
