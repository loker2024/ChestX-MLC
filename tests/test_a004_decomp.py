"""A-004 单变量分解（A-004-a / A-004-b）的配置差异。"""
from pathlib import Path

from common.config import load_config


CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _baseline():
    return load_config(CONFIGS / "exp_A001.yaml").to_dict()


def test_a004a_changes_only_dropout():
    baseline = _baseline()
    candidate = load_config(CONFIGS / "exp_A004a.yaml").to_dict()
    assert baseline["model"].pop("dropout") == 0.0
    assert candidate["model"].pop("dropout") == 0.2
    assert candidate == baseline


def test_a004b_changes_only_weight_decay():
    baseline = _baseline()
    candidate = load_config(CONFIGS / "exp_A004b.yaml").to_dict()
    assert baseline["model"].pop("dropout") == 0.0
    assert candidate["model"].pop("dropout") == 0.0
    assert baseline["train"].pop("weight_decay") == 0.01
    assert candidate["train"].pop("weight_decay") == 0.05
    assert candidate == baseline
