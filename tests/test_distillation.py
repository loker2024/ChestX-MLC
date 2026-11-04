import numpy as np
import pandas as pd
import pytest
import torch

from train.distillation import TeacherCache, distillation_loss, write_teacher_cache


def test_cache_preserves_ids_and_aligns_shuffled_batches(tmp_path):
    frame = pd.DataFrame([["001", *([0.2] * 10)], ["02", *([0.8] * 10)]], columns=["Study_id", *[f"p{i}" for i in range(1, 11)]])
    write_teacher_cache(tmp_path / "cache", frame, {"train_csv_sha256": "abc"})
    cache = TeacherCache.load(tmp_path / "cache", ["02", "001"], "abc")
    assert cache.batch(["02", "001"], torch.device("cpu"))[:, 0].tolist() == pytest.approx([0.8, 0.2])
    with pytest.raises(ValueError):
        TeacherCache.load(tmp_path / "cache", ["001"], "abc")
    with pytest.raises(ValueError):
        TeacherCache.load(tmp_path / "cache", ["02", "001"], "different")


def test_cache_rejects_corruption(tmp_path):
    frame = pd.DataFrame([["001", *([0.2] * 10)]], columns=["Study_id", *[f"p{i}" for i in range(1, 11)]])
    write_teacher_cache(tmp_path / "cache", frame, {"train_csv_sha256": "abc"})
    path = tmp_path / "cache" / "predictions.csv"
    path.write_text(path.read_text().replace("0.2", "0.9"))
    with pytest.raises(ValueError):
        TeacherCache.load(tmp_path / "cache", ["001"], "abc")


def test_soft_loss_has_expected_gradient():
    logits = torch.tensor([[0.0, 0.0]], requires_grad=True)
    target = torch.tensor([[0.8, 0.2]])
    loss = distillation_loss(logits, target)
    assert loss.item() == pytest.approx(np.log(2))
    loss.backward()
    assert logits.grad.tolist()[0] == pytest.approx([-0.15, 0.15])


def test_distillation_configuration_requires_bce_and_cache():
    from common.config import TrainConfig
    with pytest.raises(ValueError):
        TrainConfig(distill_enabled=True, distill_cache=None)
    with pytest.raises(ValueError):
        TrainConfig(distill_enabled=True, distill_cache="cache", loss="asl")


def test_distillation_changes_student_update_without_changing_teacher():
    from dataclasses import replace
    from common.config import load_config
    from train.engine import train_one_epoch
    cfg = load_config('configs/exp_D002a.yaml')
    cfg = replace(cfg, train=replace(cfg.train, amp=False, ema_enabled=False, distill_enabled=True, distill_cache='unused'))
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(10))
        def forward(self, x):
            return self.bias.expand(len(x), -1)
    class Cache:
        def batch(self, ids, device):
            return torch.full((len(ids), 10), 0.8, device=device)
    batch = {'images': torch.ones(1, 3, 2, 2), 'labels': torch.ones(1, 10), 'offsets': torch.tensor([0, 1]), 'study_ids': ['001'], 'views': torch.tensor([0])}
    model = Model()
    stats = {}
    opt = torch.optim.SGD(model.parameters(), lr=1)
    train_one_epoch(cfg, model, [batch], torch.nn.BCEWithLogitsLoss(), opt, torch.amp.GradScaler('cpu', enabled=False), torch.device('cpu'), teacher_cache=Cache(), loss_stats=stats)
    assert model.bias.detach().tolist() == pytest.approx([0.065] * 10)
    assert stats['hard'] == pytest.approx(np.log(2))
    assert stats['soft'] == pytest.approx(np.log(2))


def test_training_rejects_stale_index_and_wrong_fusion(tmp_path):
    frame = pd.DataFrame([["001", *([0.2] * 10)]], columns=["Study_id", *[f"p{i}" for i in range(1, 11)]])
    write_teacher_cache(tmp_path / 'cache', frame, {'train_csv_sha256': 'abc', 'index_sha256': 'old', 'fusion': 'wrong'})
    with pytest.raises(ValueError):
        TeacherCache.load(tmp_path / 'cache', ['001'], 'abc', index_sha256='new')


def test_fusion_aligns_by_study_id_and_clips_extreme_probabilities():
    from run_distill_teacher import fuse_teacher_frames
    columns = ['Study_id', *[f'p{i}' for i in range(1, 11)]]
    first = pd.DataFrame([['001', *([1.0] * 10)], ['02', *([0.5] * 10)]], columns=columns)
    second = pd.DataFrame([['02', *([0.5] * 10)], ['001', *([0.5] * 10)]], columns=columns)
    third = pd.DataFrame([['001', *([0.5] * 10)], ['02', *([0.5] * 10)]], columns=columns)
    out = fuse_teacher_frames([first, second, third])
    assert out['Study_id'].tolist() == ['001', '02']
    assert out.loc[0, 'p1'] == pytest.approx(1 / (1 + np.exp(-(4 / 9) * np.log((1 - 1e-6) / 1e-6))))
    assert out.loc[1, 'p1'] == pytest.approx(0.5)


@pytest.mark.parametrize('rows', [
    [['001', *([0.2] * 10)], ['001', *([0.4] * 10)]],
    [['001', *([1.1] * 10)]],
    [['001', *([float('nan')] * 10)]],
])
def test_invalid_teacher_rows_are_not_written(tmp_path, rows):
    frame = pd.DataFrame(rows, columns=['Study_id', *[f'p{i}' for i in range(1, 11)]])
    with pytest.raises(ValueError):
        write_teacher_cache(tmp_path / 'bad', frame, {'train_csv_sha256': 'abc'})
    assert not (tmp_path / 'bad').exists()
