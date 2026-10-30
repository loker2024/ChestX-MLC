"""多模型融合的纯函数与 CLI 回归测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from infer.ensemble import fuse_prediction_frames, normalize_weights


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
