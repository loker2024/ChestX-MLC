"""配置读取与校验：默认值、未知字段、类型/范围错误、Path 序列化。"""

from pathlib import Path

import pytest

from common.config import config_from_dict, load_config

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


def _write(tmp_path: Path, text: str = BASE, name: str = "config.yaml") -> Path:
    path = tmp_path / name
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


def test_config_from_dict_roundtrip(tmp_path):
    """checkpoint 里存的 to_dict() 必须能原样还原，否则续训会读错超参。"""
    config = load_config(_write(tmp_path))

    restored = config_from_dict(config.to_dict())

    assert restored == config
    assert isinstance(restored.data.train.csv, Path)


def test_config_from_dict_rejects_empty():
    with pytest.raises(ValueError, match="无法恢复"):
        config_from_dict({})


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


def test_unknown_lr_scheduler_raises(tmp_path):
    text = BASE.replace("  warmup_epochs: 0", "  warmup_epochs: 0\n  lr_scheduler: step")
    with pytest.raises(ValueError, match="lr_scheduler"):
        load_config(_write(tmp_path, text))


def test_lr_min_must_not_exceed_lr(tmp_path):
    text = BASE.replace("  warmup_epochs: 0", "  warmup_epochs: 0\n  lr_min: 1.0")
    with pytest.raises(ValueError, match="lr_min"):
        load_config(_write(tmp_path, text))


def test_cosine_experiment_config_loads():
    """RUN-003-A 配置只改调度：25 轮、warmup 2、余弦到 1e-5、num_workers 16。"""
    config = load_config(Path(__file__).resolve().parents[1] / "configs/exp_cosine.yaml")

    assert config.train.lr_scheduler == "cosine"
    assert config.train.lr_min == pytest.approx(1e-5)
    assert config.train.warmup_epochs == 2
    assert config.train.epochs == 25
    assert config.train.early_stop_patience == 5
    assert config.train.pos_weight is False
    assert config.train.num_workers == 16


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


def test_base_config_is_merged(tmp_path):
    """`_base` 指向的父配置先读入，子配置里的同名字段再覆盖上去。"""
    _write(tmp_path, BASE, name="base.yaml")
    child = tmp_path / "child.yaml"
    child.write_text("_base: base.yaml\ntrain:\n  epochs: 3\n", encoding="utf-8")

    config = load_config(child)

    assert config.train.epochs == 3          # 子配置覆盖
    assert config.train.batch_studies == 2   # 父配置延续
    assert config.image.size == 32           # 子配置没写的段整段继承


def test_base_cycle_raises(tmp_path):
    (tmp_path / "a.yaml").write_text("_base: b.yaml\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("_base: a.yaml\n", encoding="utf-8")

    with pytest.raises(ValueError, match="循环"):
        load_config(tmp_path / "a.yaml")


def test_base_must_be_a_string(tmp_path):
    child = tmp_path / "child.yaml"
    child.write_text("_base: [base.yaml]\n", encoding="utf-8")

    with pytest.raises(TypeError, match="_base"):
        load_config(child)


def test_base_file_missing_raises(tmp_path):
    child = tmp_path / "child.yaml"
    child.write_text("_base: no_such_file.yaml\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        load_config(child)


def test_experiment_configs_extend_default():
    """随代码发布的实验配置必须一直能读通，否则复现命令一跑就断。"""
    root = Path(__file__).resolve().parents[1] / "configs"
    short = load_config(root / "exp_short.yaml")
    weighted = load_config(root / "exp_short_pos_weight.yaml")

    assert short.train.epochs == 6
    assert short.train.pos_weight is False
    assert weighted.train.pos_weight is True
    assert weighted.train.epochs == 6
    assert weighted.image.size == 320
