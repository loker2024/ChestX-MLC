"""配置读取与校验：默认值、未知字段、类型/范围错误、Path 序列化。"""

from pathlib import Path

import pytest

from common.config import load_config

BASE = """data:
  train: {csv: data/train.csv, images: data/images/train}
  valid: {csv: data/valid.csv, images: data/images/valid}
  test: {csv: data/test.csv, images: data/images/test}
image:
  size: 32
model:
  pretrained: false
train:
  epochs: 1
  warmup_epochs: 0
  batch_studies: 2
  num_workers: 0
  early_stop_patience: 0
eval:
  batch_studies: 2
output:
  run_dir: outputs
seed: 42
device: cpu
"""


def _write(tmp_path: Path, text: str = BASE) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_default_config_loads():
    config = load_config(Path(__file__).resolve().parents[1] / "configs/default.yaml")

    assert config.model.backbone == "resnet18"
    assert config.image.size == 320
    assert config.eval.threshold == 0.5
    assert config.output.run_dir == Path("outputs")
    assert "checkpoint_dir" not in config.to_dict()["output"]


def test_to_dict_converts_paths_to_strings(tmp_path):
    config = load_config(_write(tmp_path))

    exported = config.to_dict()

    assert exported["data"]["train"]["csv"] == "data/train.csv"
    assert exported["output"]["run_dir"] == "outputs"
    assert isinstance(exported["seed"], int)


def test_unknown_top_level_field_raises(tmp_path):
    with pytest.raises(ValueError, match="未知字段"):
        load_config(_write(tmp_path, BASE + "\nunknown: 1\n"))


def test_unknown_nested_field_raises(tmp_path):
    text = BASE.replace(
        "model:\n  pretrained: false", "model:\n  pretrained: false\n  typo: 1"
    )
    with pytest.raises(ValueError, match="未知字段"):
        load_config(_write(tmp_path, text))


def test_wrong_type_raises(tmp_path):
    text = BASE.replace("  epochs: 1", "  epochs: many")
    with pytest.raises(TypeError, match="整数"):
        load_config(_write(tmp_path, text))


def test_warmup_must_be_less_than_epochs(tmp_path):
    text = BASE.replace("  warmup_epochs: 0", "  warmup_epochs: 1")
    with pytest.raises(ValueError, match="warmup"):
        load_config(_write(tmp_path, text))


def test_eval_threshold_range_is_checked(tmp_path):
    text = BASE.replace(
        "eval:\n  batch_studies: 2", "eval:\n  batch_studies: 2\n  threshold: 1.5"
    )
    with pytest.raises(ValueError, match="threshold"):
        load_config(_write(tmp_path, text))


def test_missing_data_section_raises(tmp_path):
    with pytest.raises(ValueError, match="必填"):
        load_config(_write(tmp_path, "seed: 1\n"))


def test_empty_config_raises(tmp_path):
    with pytest.raises(ValueError, match="为空"):
        load_config(_write(tmp_path, ""))
