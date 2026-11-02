"""训练结束后的 fp32 复评：调用契约、CSV 列与最优行定位。

背景：训练期验证跑 AMP（fp16）前向，`run_eval` / 提交跑 fp32 前向，两条路径
在阈值附近会翻转少量样本，macro-F1 可差千分位。复评只做一次、只在训练结束时
做，所以这几个纯函数要能被单独验证，不必每次都跑完整训练。
"""

import numpy as np
import pandas as pd
import pytest
import torch
from pathlib import Path

import run_train
from common.config import load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_recheck_forces_fp32_and_drops_per_class(monkeypatch):
    """复评必须显式关掉 AMP，并且只把标量指标带回来。"""
    seen: dict = {}

    monkeypatch.setattr(
        run_train, "train_labels", lambda _: np.zeros((2, 10), dtype=np.float32)
    )
    monkeypatch.setattr(
        run_train, "build_loss", lambda cfg, labels, device=None: torch.nn.Identity()
    )

    def fake_evaluate(config, model, loader, criterion, device, **kwargs):
        seen.update(kwargs)
        return {"loss": 0.42, "macro_auc": 0.81, "macro_f1": 0.43, "per_class": pd.DataFrame()}

    monkeypatch.setattr(run_train, "evaluate", fake_evaluate)

    metrics = run_train._recheck_best_metrics(
        load_config(CONFIGS / "exp_example.yaml"),
        torch.nn.Identity(),
        object(),
        train_loader=object(),
        device=torch.device("cpu"),
    )

    assert seen == {"use_amp": False}
    assert metrics == {"loss": 0.42, "macro_auc": 0.81, "macro_f1": 0.43}


def test_record_fp32_metrics_only_touches_best_epoch():
    """只给最优那一轮加列，其他轮次的记录保持原样。"""
    history = [{"epoch": 1}, {"epoch": 2}, {"epoch": 3}]

    run_train._record_fp32_metrics(
        history, 2, {"loss": 0.3, "macro_auc": 0.8, "macro_f1": 0.4}
    )

    assert "valid_macro_auc_fp32" not in history[0]
    assert "valid_macro_auc_fp32" not in history[2]
    assert history[1]["valid_loss_fp32"] == 0.3
    assert history[1]["valid_macro_auc_fp32"] == 0.8
    assert history[1]["valid_macro_f1_fp32"] == 0.4
    # 原始列不动。
    assert "valid_macro_auc" not in history[1]


def test_record_fp32_metrics_ignores_unknown_epoch():
    """找不到对应轮次时安静返回，不抛错也不新建行。"""
    history = [{"epoch": 1}]

    run_train._record_fp32_metrics(
        history, 9, {"loss": 0.3, "macro_auc": 0.8, "macro_f1": 0.4}
    )

    assert history == [{"epoch": 1}]


def test_write_metrics_keeps_fp32_columns(tmp_path):
    """列取并集：只有复评那一行有值，其余行留空。"""
    history = [
        {"epoch": 1, "valid_macro_auc": 0.70},
        {"epoch": 2, "valid_macro_auc": 0.72, "valid_macro_auc_fp32": 0.7195},
    ]
    path = tmp_path / "metrics.csv"

    run_train._write_metrics(path, history)

    frame = pd.read_csv(path)
    assert list(frame.columns) == ["epoch", "valid_macro_auc", "valid_macro_auc_fp32"]
    assert frame["valid_macro_auc_fp32"].isna().tolist() == [True, False]
    assert frame.loc[1, "valid_macro_auc_fp32"] == pytest.approx(0.7195)


def test_report_fp32_delta_prints_both_values(capsys):
    """对比行要同时给出 fp32 值、AMP 值与差值，缺 AMP 值时不炸。"""
    run_train._report_fp32_delta(
        5, {"macro_auc": 0.7962, "macro_f1": 0.4230}, {"macro_auc": 0.7962, "macro_f1": 0.4218}
    )
    out = capsys.readouterr().out
    assert "fp32 复评（epoch 5）" in out
    assert "0.4218" in out and "0.4230" in out and "-0.0012" in out

    run_train._report_fp32_delta(
        5, {"macro_auc": float("nan"), "macro_f1": float("nan")},
        {"macro_auc": 0.7962, "macro_f1": 0.4218},
    )
    assert "训练期 AMP 无有效值" in capsys.readouterr().out
