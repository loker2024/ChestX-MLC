"""端到端测试：训练一步 → 保存/重载 checkpoint → 推理 → 生成并校验 submission。

这是 P6 的闭环测试：用 2 个 study / 3 张小图跑一遍完整链路，确认各模块
拼在一起时数据形状、ID 顺序和文件格式都对得上。
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import run_eval
import run_infer
from common.checkpoint import save_checkpoint
from common.config import Config, load_config
from common.constants import CLASS_IDS, NUM_CLASSES
from data.dataset import build_loader
from infer.predict import load_checkpoint_model, predict_loader
from infer.submission import (
    SUBMISSION_COLUMNS,
    build_submission_frame,
    validate_submission,
    write_submission,
)
from model.network import build_model
from train.engine import fit


@pytest.fixture
def config_path(
    tmp_path: Path, labeled_split: dict, unlabeled_split: dict, write_config
) -> Path:
    """train / valid 用带标签数据，test 用去掉标签的同一批图片。"""
    return write_config(
        {
            "train": (labeled_split["csv"], labeled_split["images"]),
            "valid": (labeled_split["csv"], labeled_split["images"]),
            "test": (unlabeled_split["csv"], unlabeled_split["images"]),
        }
    )


@pytest.fixture
def config(config_path: Path) -> Config:
    """关掉预训练和早停、只跑 1 个 epoch，CPU 上几秒完成。"""
    config = load_config(config_path)
    return replace(
        config,
        model=replace(config.model, pretrained=False),
        train=replace(
            config.train,
            epochs=1,
            warmup_epochs=0,
            early_stop_patience=0,
        ),
    )


def test_pipeline_from_training_to_submission(
    config: Config,
    config_path: Path,
    tmp_path: Path,
) -> None:
    device = torch.device("cpu")
    torch.manual_seed(0)

    # 1. 训练一个 epoch，确认 loss 有限、参数确实更新。
    train_loader = build_loader(config, "train", train=True)
    val_loader = build_loader(config, "valid")
    model = build_model(config)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    result = fit(config, model, train_loader, val_loader, device)

    assert math.isfinite(result["history"][0]["train_loss"])
    assert any(
        not torch.equal(value, before[name])
        for name, value in model.state_dict().items()
    )

    # 2. 保存 best.pt，再从文件重载，权重必须逐位一致。
    model.load_state_dict(result["best_state"])
    checkpoint_path = save_checkpoint(
        tmp_path / "best.pt",
        model,
        epoch=result["best_epoch"],
        metrics=result["best_metrics"],
        config=config,
        seed=config.seed,
    )
    reloaded = load_checkpoint_model(config, checkpoint_path, device)
    for name, value in model.state_dict().items():
        assert torch.equal(value, reloaded.state_dict()[name])

    # 3. test 推理：行数等于 study 数，概率在 [0, 1]，列名是 Study_id + p1..p10。
    test_loader = build_loader(config, "test")
    predictions = predict_loader(reloaded, test_loader, device)
    expected_columns = ["Study_id"] + [f"p{class_id}" for class_id in CLASS_IDS]
    assert list(predictions.columns) == expected_columns
    assert predictions["Study_id"].tolist() == [
        record.study_id for record in test_loader.dataset.records
    ]
    probabilities = predictions[expected_columns[1:]].to_numpy()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()

    # 4. 宽表 -> 长表 -> 写文件 -> 按赛题格式校验。
    study_subjects = {
        record.study_id: record.subject_id for record in test_loader.dataset.records
    }
    submission = build_submission_frame(predictions, study_subjects)
    assert list(submission.columns) == list(SUBMISSION_COLUMNS)
    assert len(submission) == len(predictions) * NUM_CLASSES
    submission_path = write_submission(submission, tmp_path / "submission.csv")
    validated = validate_submission(submission_path, config.data.test.csv)
    assert len(validated) == len(submission)

    # 5. 两个命令行入口也能直接跑通（main 接受 argv 就是为了这里复用）。
    run_eval.main(
        ["--config", str(config_path), "--checkpoint", str(checkpoint_path)]
    )
    cli_submission_path = tmp_path / "cli_submission.csv"
    run_infer.main(
        [
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint_path),
            "--out",
            str(cli_submission_path),
        ]
    )
    assert cli_submission_path.exists()
    validate_submission(cli_submission_path, config.data.test.csv)
