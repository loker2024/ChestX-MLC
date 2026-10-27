"""断点续训：进度恢复、学习率衔接、last.pt 内容与 CLI 续训。

这里模拟“训练跑到一半被人为中断”的两种情形：
1. 引擎层：`on_epoch_end` 在某一轮结束时抛异常中断，把进度、优化器与调度器
   状态抓下来，之后接着跑，学习率必须和一次性跑完完全一致；
2. 入口层：CLI 先跑 1 个 epoch，然后 `--resume` 补跑到 3 个 epoch，
   metrics.csv 累积、train.log 追加、且不新建 run 目录。
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
import torch

import run_train
from common.checkpoint import load_checkpoint
from common.config import Config, load_config
from data.dataset import build_loader
from model.network import build_model
from train.engine import fit


class _StopTraining(Exception):
    """在 on_epoch_end 里抛出来，模拟训练中途被打断。"""


def _make_config(config_path: Path, epochs: int, warmup_epochs: int = 0) -> Config:
    """关掉预训练与早停，把轮数压到几个，CPU 上几秒跑完。"""
    config = load_config(config_path)
    return replace(
        config,
        model=replace(config.model, pretrained=False),
        train=replace(
            config.train,
            epochs=epochs,
            warmup_epochs=warmup_epochs,
            early_stop_patience=0,
        ),
    )


def _loaders(config: Config):
    return build_loader(config, "train", train=True), build_loader(config, "valid")


@pytest.fixture
def config_path(tmp_path: Path, labeled_split: dict, write_config) -> Path:
    split = (labeled_split["csv"], labeled_split["images"])
    return write_config({"train": split, "valid": split, "test": split})


def test_resume_continues_history_and_learning_rate(config_path: Path) -> None:
    """中断后续训：history 接上旧记录，学习率与一次跑完的轨迹一致。"""
    config = _make_config(config_path, epochs=4, warmup_epochs=2)
    train_loader, val_loader = _loaders(config)
    device = torch.device("cpu")

    # 基准：同一随机种子、一次性跑完 4 轮。
    torch.manual_seed(0)
    full = fit(config, build_model(config), train_loader, val_loader, device)
    expected_lrs = [record["lr"] for record in full["history"]]

    # 跑到第 2 轮结束时“被打断”，保存可续训的状态。
    torch.manual_seed(0)
    model = build_model(config)
    captured: dict[str, object] = {}

    def stop_after_second_epoch(progress, runtime) -> None:
        if progress.epoch == 2:
            captured.update(
                progress=progress,
                optimizer=runtime.optimizer.state_dict(),
                scheduler=runtime.scheduler.state_dict(),
            )
            raise _StopTraining

    with pytest.raises(_StopTraining):
        fit(config, model, train_loader, val_loader, device, on_epoch_end=stop_after_second_epoch)

    resumed = fit(
        config,
        model,
        train_loader,
        val_loader,
        device,
        progress=captured["progress"],
        optimizer_state=captured["optimizer"],
        scheduler_state=captured["scheduler"],
    )

    assert [record["epoch"] for record in resumed["history"]] == [1, 2, 3, 4]
    assert [record["lr"] for record in resumed["history"]] == pytest.approx(expected_lrs)
    assert resumed["progress"].epoch == 4


def test_resume_keeps_best_epoch_from_before_interruption(config_path: Path) -> None:
    """续训不能把之前挑出的最优权重丢掉：best_epoch 必须沿用旧值。"""
    config = _make_config(config_path, epochs=3)
    train_loader, val_loader = _loaders(config)
    device = torch.device("cpu")

    torch.manual_seed(0)
    model = build_model(config)
    captured: dict[str, object] = {}

    def stop_after_first_epoch(progress, runtime) -> None:
        captured.update(progress=progress, optimizer=runtime.optimizer.state_dict())
        raise _StopTraining

    with pytest.raises(_StopTraining):
        fit(config, model, train_loader, val_loader, device, on_epoch_end=stop_after_first_epoch)

    progress = captured["progress"]
    best_state = {name: value.clone() for name, value in progress.best_state.items()}
    assert progress.best_epoch == 1

    resumed = fit(
        config,
        model,
        train_loader,
        val_loader,
        device,
        progress=progress,
        optimizer_state=captured["optimizer"],
    )

    # 最优轮次与快照都还在，且 history 是累加的。
    assert resumed["progress"].best_epoch >= 1
    if resumed["best_epoch"] == 1:
        for name, value in resumed["best_state"].items():
            assert torch.equal(value, best_state[name])


def _write_config(
    tmp_path: Path, split: dict, run_root: Path, epochs: int, name: str = "config.yaml"
) -> Path:
    """写一份指向临时数据的训练配置；epochs 与 run 根目录可调。"""
    text = f"""data:
  train: {{csv: {json.dumps(str(split["csv"]))}, images: {json.dumps(str(split["images"]))}}}
  valid: {{csv: {json.dumps(str(split["csv"]))}, images: {json.dumps(str(split["images"]))}}}
  test: {{csv: {json.dumps(str(split["csv"]))}, images: {json.dumps(str(split["images"]))}}}
image:
  size: 32
model:
  pretrained: false
train:
  epochs: {epochs}
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
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_run_train_resume_finishes_remaining_epochs(tmp_path: Path, labeled_split: dict) -> None:
    run_root = tmp_path / "outputs"
    first_config = _write_config(tmp_path, labeled_split, run_root, epochs=1)
    run_train.main(["--config", str(first_config), "--tag", "resume"])

    run_dir = next(run_root.iterdir())
    assert (run_dir / "last.pt").is_file()

    # 把轮数提到 3，然后从断点补跑剩下两轮。
    more_config = _write_config(tmp_path, labeled_split, run_root, epochs=3, name="more.yaml")
    run_train.main(["--config", str(more_config), "--resume", str(run_dir)])

    # 续训复用原目录，不新建 run。
    assert [path.name for path in run_root.iterdir()] == [run_dir.name]

    metrics = pd.read_csv(run_dir / "metrics.csv")
    assert list(metrics["epoch"]) == [1, 2, 3]

    last = load_checkpoint(run_dir / "last.pt")
    assert last["epoch"] == 3
    assert last["optimizer"] is not None
    assert last["scheduler"] is not None
    assert last["extras"]["history"][-1]["epoch"] == 3
    assert (run_dir / "config_resume.yaml").is_file()

    best = load_checkpoint(run_dir / "best.pt")
    assert 1 <= best["epoch"] <= 3

    log = (run_dir / "train.log").read_text(encoding="utf-8")
    assert "续训" in log
    assert "从断点继续" in log


def test_resume_without_config_reuses_saved_config(tmp_path: Path, labeled_split: dict) -> None:
    run_root = tmp_path / "outputs"
    config_path = _write_config(tmp_path, labeled_split, run_root, epochs=1)
    run_train.main(["--config", str(config_path)])

    run_dir = next(run_root.iterdir())
    # 不给 --config：沿用 last.pt 里存档的配置（epochs=1，已经跑满）。
    run_train.main(["--resume", str(run_dir)])

    assert len(pd.read_csv(run_dir / "metrics.csv")) == 1
    log = (run_dir / "train.log").read_text(encoding="utf-8")
    assert "已跑满" in log
    assert (run_dir / "best.pt").is_file()


def test_resume_requires_last_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="last.pt"):
        run_train._resolve_resume(run_dir)


def test_resume_rejects_checkpoint_without_progress(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    torch.save({"model": {}, "extras": None, "config": None}, run_dir / "last.pt")
    with pytest.raises(ValueError, match="extras"):
        run_train._resolve_resume(run_dir)
