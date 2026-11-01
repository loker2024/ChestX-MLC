from dataclasses import replace

import numpy as np
import pytest
import torch

from common.config import TrainConfig, load_config
from model.losses import build_loss


def test_power_changes_only_nondegenerate_class_weights():
    cfg = load_config('configs/exp_A001.yaml')
    labels = np.zeros((10, 10))
    labels[0, 0] = 1
    labels[:, 1] = 1
    for power, expected in [(1.0, 9.0), (0.5, 3.0), (0.0, 1.0)]:
        candidate = replace(cfg, train=replace(cfg.train, pos_weight_power=power))
        weight = build_loss(candidate, labels).pos_weight
        assert weight[0].item() == pytest.approx(expected)
        assert torch.all(weight[1:] == 1)
    assert build_loss(cfg, labels).pos_weight[0].item() == pytest.approx(9.0)


@pytest.mark.parametrize('power', [-0.1, 1.1, float('nan'), float('inf')])
def test_power_rejects_invalid_values(power):
    with pytest.raises(ValueError, match='pos_weight_power'):
        TrainConfig(pos_weight_power=power)


def test_example_config_is_single_variable_and_resume_rejects_change():
    from run_train import _validate_resume_config

    baseline = load_config('configs/exp_A001.yaml')
    candidate = load_config('configs/exp_pos_weight_sqrt.yaml')
    expected = replace(baseline, train=replace(baseline.train, pos_weight_power=0.5))
    assert candidate == expected
    with pytest.raises(ValueError, match='train.pos_weight_power'):
        _validate_resume_config(candidate, baseline)


def test_disabled_pos_weight_does_not_require_labels():
    cfg = load_config('configs/exp_pos_weight_sqrt.yaml')
    cfg = replace(cfg, train=replace(cfg.train, pos_weight=False))
    assert build_loss(cfg).pos_weight is None
