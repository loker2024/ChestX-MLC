"""多模型融合的纯函数与 CLI 回归测试。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
import yaml

from infer.ensemble import fuse_prediction_frames, normalize_weights
from infer.predict import predict_loader_logits
from common.checkpoint import save_checkpoint
from common.config import load_config
from model.network import build_model


def _probability_frame(study_ids: list[str], values: list[float]) -> pd.DataFrame:
    frame = pd.DataFrame({"Study_id": study_ids})
    for class_id in range(1, 11):
        frame[f"p{class_id}"] = values
    return frame


def _logit_frame(study_ids: list[str], values: list[float]) -> pd.DataFrame:
    frame = pd.DataFrame({"Study_id": study_ids})
    for class_id in range(1, 11):
        frame[f"z{class_id}"] = values
    return frame


def test_probability_fusion_aligns_studies_and_normalizes_weights() -> None:
    first = _probability_frame(["study_a", "study_b"], [0.2, 0.8])
    second = _probability_frame(["study_b", "study_a"], [0.4, 0.6])

    fused = fuse_prediction_frames([first, second], weights=[1.0, 3.0], method="probability")

    assert fused["Study_id"].tolist() == ["study_a", "study_b"]
    np.testing.assert_allclose(fused["p1"], [0.5, 0.5])


def test_logit_fusion_applies_sigmoid_after_weighted_logits() -> None:
    first = _logit_frame(["study_a"], [0.0])
    second = _logit_frame(["study_a"], [np.log(3.0)])

    fused = fuse_prediction_frames([first, second], weights=[1.0, 1.0], method="logit")

    np.testing.assert_allclose(fused["p1"], [1.0 / (1.0 + np.sqrt(3.0)) * np.sqrt(3.0)])


def test_default_weights_are_equal() -> None:
    weights = normalize_weights(None, 3)
    np.testing.assert_allclose(weights, [1 / 3, 1 / 3, 1 / 3])


@pytest.mark.parametrize(
    ("weights", "count", "message"),
    [
        ([1.0], 2, "数量"),
        ([-1.0, 2.0], 2, "非负"),
        ([0.0, 0.0], 2, "正"),
    ],
)
def test_invalid_weights_raise(weights: list[float], count: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_weights(weights, count)


def test_fusion_rejects_mismatched_studies() -> None:
    first = _probability_frame(["study_a"], [0.2])
    second = _probability_frame(["study_b"], [0.8])

    with pytest.raises(ValueError, match="Study_id"):
        fuse_prediction_frames([first, second], weights=None, method="probability")


def test_fusion_rejects_unknown_method_and_missing_columns() -> None:
    frame = _probability_frame(["study_a"], [0.2])

    with pytest.raises(ValueError, match="fusion method"):
        fuse_prediction_frames([frame], weights=None, method="mean")

    with pytest.raises(ValueError, match="p1"):
        fuse_prediction_frames(
            [frame.drop(columns="p1")], weights=None, method="probability"
        )


class _MeanImageModel(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        values = images.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        return values.repeat(1, 10)


def test_predict_loader_logits_aggregates_images_by_study() -> None:
    batch = {
        "images": torch.tensor(
            [
                [[[1.0, 1.0], [1.0, 1.0]]],
                [[[3.0, 3.0], [3.0, 3.0]]],
                [[[5.0, 5.0], [5.0, 5.0]]],
            ]
        ),
        "offsets": torch.tensor([0, 2, 3]),
        "study_ids": ["study_a", "study_b"],
    }

    logits = predict_loader_logits(_MeanImageModel(), [batch], torch.device("cpu"))

    assert logits["Study_id"].tolist() == ["study_a", "study_b"]
    np.testing.assert_allclose(logits["z1"], [2.0, 5.0])


def _make_run_pair(tmp_path: Path, labeled_split: dict, write_config) -> tuple[Path, Path]:
    config_path = write_config(
        {
            "train": (labeled_split["csv"], labeled_split["images"]),
            "valid": (labeled_split["csv"], labeled_split["images"]),
            "test": (labeled_split["csv"], labeled_split["images"]),
        }
    )
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["model"] = {"pretrained": False}
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    config = load_config(config_path)

    runs = []
    for index in range(2):
        torch.manual_seed(index)
        model = build_model(config)
        run_dir = tmp_path / f"run_{index}"
        checkpoint = save_checkpoint(run_dir / "best.pt", model, config=config, seed=index)
        runs.append(checkpoint)
    return runs[0], runs[1]


def test_run_ensemble_valid_writes_metrics_and_threshold_curve(
    tmp_path: Path, labeled_split: dict, write_config
) -> None:
    import run_ensemble

    checkpoint_a, checkpoint_b = _make_run_pair(tmp_path, labeled_split, write_config)
    output_dir = tmp_path / "ensemble_valid"

    run_ensemble.main(
        [
            "--checkpoint",
            str(checkpoint_a),
            str(checkpoint_b),
            "--weights",
            "1,3",
            "--fusion",
            "probability",
            "--split",
            "valid",
            "--thresholds",
            "0.3,0.5",
            "--out",
            str(output_dir),
        ]
    )

    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["fusion"] == "probability"
    np.testing.assert_allclose(metrics["weights"], [0.25, 0.75])
    assert {"macro_auc", "macro_f1"} <= set(metrics)
    assert len(pd.read_csv(output_dir / "per_class.csv")) == 10
    assert pd.read_csv(output_dir / "f1_curve.csv")["threshold"].tolist() == [0.3, 0.5]
    assert set(pd.read_csv(output_dir / "predictions.csv").columns) == {
        "Study_id",
        *(f"p{class_id}" for class_id in range(1, 11)),
    }


def test_run_ensemble_test_writes_validated_submission(
    tmp_path: Path, labeled_split: dict, write_config
) -> None:
    import run_ensemble

    checkpoint_a, checkpoint_b = _make_run_pair(tmp_path, labeled_split, write_config)
    output_dir = tmp_path / "ensemble_test"

    run_ensemble.main(
        [
            "--checkpoint",
            str(checkpoint_a),
            str(checkpoint_b),
            "--fusion",
            "logit",
            "--split",
            "test",
            "--out",
            str(output_dir),
        ]
    )

    submission = pd.read_csv(output_dir / "submission.csv")
    assert list(submission.columns) == [
        "Subject_id",
        "Study_id",
        "Predict_class",
        "Probability",
    ]
    assert submission.groupby("Study_id").size().tolist() == [10, 10]
    assert submission["Probability"].between(0.0, 1.0).all()


def test_run_ensemble_rejects_config_count_mismatch(tmp_path: Path) -> None:
    import run_ensemble

    with pytest.raises(ValueError, match="--config 数量"):
        run_ensemble.main(
            [
                "--checkpoint",
                str(tmp_path / "a.pt"),
                str(tmp_path / "b.pt"),
                "--config",
                str(tmp_path / "only.yaml"),
            ]
        )


def test_run_ensemble_rejects_subject_mismatch() -> None:
    import run_ensemble

    loader_a = SimpleNamespace(
        dataset=SimpleNamespace(
            records=[SimpleNamespace(study_id="study_a", subject_id="subject_a")]
        )
    )
    loader_b = SimpleNamespace(
        dataset=SimpleNamespace(
            records=[SimpleNamespace(study_id="study_a", subject_id="subject_b")]
        )
    )

    with pytest.raises(ValueError, match="Subject_id"):
        run_ensemble._check_subject_loaders([loader_a, loader_b])
