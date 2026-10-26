"""对比入口：读回 run 目录里存档的配置与权重，重算指标并写出对比表。"""

from pathlib import Path

import pandas as pd
import pytest
import yaml

import run_compare
from common.checkpoint import save_checkpoint
from common.config import load_config
from model.network import build_model


def _make_run(root: Path, name: str, raw: dict, pos_weight: bool, seed: int) -> Path:
    """造一个最小 run 目录：config.yaml（训练时存档的配置）+ best.pt。

    真实 run 目录由 run_train.py 写出；这里只需要对比入口会读的那两样东西。
    """
    run_dir = root / name
    run_dir.mkdir()

    # 每个 run 有自己的 pos_weight 和 seed —— 正是 P7 要做的那种对照。
    run_config = yaml.safe_load(yaml.safe_dump(raw))  # 深拷贝，两个 run 互不影响
    run_config.setdefault("train", {})["pos_weight"] = pos_weight
    run_config["seed"] = seed
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(run_config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    config = load_config(run_dir / "config.yaml")
    model = build_model(config)  # 测试配置里 pretrained=false，不会联网
    save_checkpoint(
        run_dir / "best.pt",
        model,
        epoch=3,
        metrics={"valid_macro_auc": 0.5},
        config=config,
        seed=seed,
        metadata={"pretrained_loaded": False, "device": "cpu"},
    )
    return run_dir


@pytest.fixture
def run_pair(tmp_path, labeled_split, write_config):
    """两个只差 pos_weight / seed 的 run，模拟 baseline 与加权组。"""
    config_path = write_config(
        {
            "train": (labeled_split["csv"], labeled_split["images"]),
            "valid": (labeled_split["csv"], labeled_split["images"]),
            "test": (labeled_split["csv"], labeled_split["images"]),
        }
    )
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    # write_config 省略了 model 段，默认会去下载 ImageNet 权重；测试里显式关掉。
    raw["model"] = {"pretrained": False}
    return (
        _make_run(tmp_path, "run_a", raw, pos_weight=False, seed=42),
        _make_run(tmp_path, "run_b", raw, pos_weight=True, seed=7),
    )


def test_compare_writes_summary_for_each_run(tmp_path, run_pair):
    run_a, run_b = run_pair
    out = tmp_path / "compare"

    run_compare.main(
        [
            "--runs",
            str(run_a),
            str(run_b),
            "--thresholds",
            "0.3,0.5",
            "--out",
            str(out),
            "--device",
            "cpu",
        ]
    )

    summary = pd.read_csv(out / "summary.csv", encoding="utf-8")
    assert summary["run"].tolist() == ["run_a", "run_b"]
    assert summary["pos_weight"].tolist() == ["否", "是"]
    assert summary["seed"].tolist() == [42, 7]
    assert summary["best_epoch"].tolist() == [3, 3]
    assert summary["valid_studies"].tolist() == [2, 2]
    assert {
        "backbone",
        "image_size",
        "pretrained",
        "train_device",
        "macro_auc",
        "f1@0.3",
        "f1@0.5",
    } <= set(summary.columns)

    # Markdown 是给人看的版本：标题 + 表格，能直接贴进 docs/experiments.md。
    markdown = (out / "summary.md").read_text(encoding="utf-8")
    assert markdown.startswith("# 训练 run 对比")
    assert "| f1@0.5 |" in markdown


def test_missing_run_artifacts_raise(tmp_path):
    """目录里没有 config.yaml / best.pt 时，应提示它来自 run_train.py。"""
    empty = tmp_path / "empty_run"
    empty.mkdir()

    with pytest.raises(FileNotFoundError, match="缺少"):
        run_compare.main(["--runs", str(empty), "--out", str(tmp_path / "out"), "--device", "cpu"])
