"""训练入口：CLI 产物、seed 覆盖、预训练元数据与日志落盘。"""

import json
from pathlib import Path

import pandas as pd

import run_train
from common.checkpoint import load_checkpoint


def _write_config(tmp_path: Path, split: dict, run_root: Path) -> Path:
    text = f"""data:
  train: {{csv: {json.dumps(str(split["csv"]))}, images: {json.dumps(str(split["images"]))}}}
  valid: {{csv: {json.dumps(str(split["csv"]))}, images: {json.dumps(str(split["images"]))}}}
  test: {{csv: {json.dumps(str(split["csv"]))}, images: {json.dumps(str(split["images"]))}}}
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
  run_dir: {json.dumps(str(run_root))}
seed: 42
device: cpu
"""
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_run_train_cli_writes_artifacts(tmp_path, labeled_split):
    run_root = tmp_path / "outputs"
    config_path = _write_config(tmp_path, labeled_split, run_root)

    run_train.main(
        ["--config", str(config_path), "--seed", "7", "--tag", "smoke"]
    )

    runs = sorted(run_root.iterdir())
    assert len(runs) == 1
    run_dir = runs[0]
    for name in ("config.yaml", "train.log", "metrics.csv", "best.pt"):
        assert (run_dir / name).is_file()

    log = (run_dir / "train.log").read_text(encoding="utf-8")
    assert "预训练权重: 未加载" in log
    assert "运行目录" in log

    checkpoint = load_checkpoint(run_dir / "best.pt")
    assert checkpoint["seed"] == 7
    assert checkpoint["epoch"] == 1
    assert checkpoint["metadata"]["pretrained_loaded"] is False
    assert checkpoint["config"]["seed"] == 7

    metrics = pd.read_csv(run_dir / "metrics.csv")
    assert len(metrics) == 1
    assert {"epoch", "train_loss", "valid_macro_auc"} <= set(metrics.columns)
