"""提交文件测试：宽表摊平、阈值过滤、UTF-8 写出与格式校验。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from common.constants import CLASS_IDS, NUM_CLASSES
from infer.submission import (
    SUBMISSION_COLUMNS,
    build_submission_frame,
    validate_submission,
    write_submission,
)

STUDY_TO_SUBJECT = {"study_00000": "subject_00000", "study_00001": "subject_00001"}
TEST_HEADER = [
    "Dicom_id",
    "Subject_id",
    "Study_id",
    "ViewPosition",
    "PatientOrientation",
    "Rows",
    "Columns",
]


def _predictions(first: list[float], second: list[float]) -> pd.DataFrame:
    """造两行预测宽表，列名与 predict_loader 的输出一致。"""
    frame = pd.DataFrame(
        [first, second], columns=[f"p{class_id}" for class_id in CLASS_IDS]
    )
    frame.insert(0, "Study_id", list(STUDY_TO_SUBJECT))
    return frame


@pytest.fixture
def predictions() -> pd.DataFrame:
    # 第一个 study 只有类别 1 过 0.5；第二个 study 全部低于 0.5。
    return _predictions(
        [0.9, 0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        [0.2] * NUM_CLASSES,
    )


def _valid_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    return build_submission_frame(predictions, STUDY_TO_SUBJECT)


def test_build_submission_frame_outputs_all_classes(predictions):
    frame = _valid_frame(predictions)

    assert list(frame.columns) == list(SUBMISSION_COLUMNS)
    assert len(frame) == len(STUDY_TO_SUBJECT) * NUM_CLASSES
    # study 顺序跟宽表一致，study 内部类别按 1~10 排列。
    assert frame["Study_id"].tolist() == ["study_00000"] * NUM_CLASSES + [
        "study_00001"
    ] * NUM_CLASSES
    assert frame["Subject_id"].tolist()[:NUM_CLASSES] == ["subject_00000"] * NUM_CLASSES
    assert frame["Predict_class"].tolist()[:NUM_CLASSES] == list(CLASS_IDS)
    assert frame["Probability"].tolist()[:NUM_CLASSES] == predictions.iloc[0, 1:].tolist()


def test_build_submission_frame_filters_by_threshold(predictions):
    frame = build_submission_frame(predictions, STUDY_TO_SUBJECT, threshold=0.5)

    # 第一个 study 只剩类别 1；第二个 study 全都没过线，保留概率最高的类别 1。
    assert len(frame) == 2
    assert frame["Study_id"].tolist() == ["study_00000", "study_00001"]
    assert frame["Predict_class"].tolist() == [1, 1]
    assert frame["Probability"].tolist() == [0.9, 0.2]


def test_write_and_validate_submission(tmp_path, predictions, unlabeled_split):
    frame = _valid_frame(predictions)
    path = write_submission(frame, tmp_path / "submission.csv")

    assert path.read_text(encoding="utf-8").startswith(
        "Subject_id,Study_id,Predict_class,Probability"
    )
    validated = validate_submission(path, unlabeled_split["csv"])
    assert len(validated) == len(frame)
    assert validated["Study_id"].tolist() == frame["Study_id"].tolist()
    assert validated["Predict_class"].tolist() == frame["Predict_class"].tolist()
    assert np.allclose(validated["Probability"], frame["Probability"])


def test_validate_submission_can_allow_filtered_file(
    tmp_path, predictions, unlabeled_split
):
    frame = build_submission_frame(predictions, STUDY_TO_SUBJECT, threshold=0.5)
    path = write_submission(frame, tmp_path / "submission.csv")

    # 默认要求每个 study 10 类齐全，过滤后的文件会被拒绝。
    with pytest.raises(ValueError, match="个类别"):
        validate_submission(path, unlabeled_split["csv"])

    validated = validate_submission(
        path, unlabeled_split["csv"], require_all_classes=False
    )
    assert len(validated) == 2


def test_validate_submission_preserves_leading_zeros(tmp_path, write_csv):
    test_csv = write_csv(
        "test_leading_zero.csv",
        [["pic_000000", "0000123", "0000001", "AP", "Erect", "48", "64"]],
        header=TEST_HEADER,
    )
    predictions = pd.DataFrame(
        {"Study_id": ["0000001"], **{f"p{class_id}": [0.1] for class_id in CLASS_IDS}}
    )

    frame = build_submission_frame(predictions, {"0000001": "0000123"})
    path = write_submission(frame, tmp_path / "submission.csv")
    validated = validate_submission(path, test_csv)

    assert validated["Study_id"].unique().tolist() == ["0000001"]
    assert validated["Subject_id"].unique().tolist() == ["0000123"]


def _write_and_validate(tmp_path, frame, test_csv):
    path = write_submission(frame, tmp_path / "bad.csv")
    return validate_submission(path, test_csv)


def test_validate_submission_rejects_wrong_column_order(
    tmp_path, predictions, unlabeled_split
):
    frame = _valid_frame(predictions)
    frame = frame[["Study_id", "Subject_id", "Predict_class", "Probability"]]

    with pytest.raises(ValueError, match="列名或顺序"):
        _write_and_validate(tmp_path, frame, unlabeled_split["csv"])


def test_validate_submission_rejects_missing_study(
    tmp_path, predictions, unlabeled_split
):
    frame = _valid_frame(predictions)
    frame = frame[frame["Study_id"] == "study_00000"]

    with pytest.raises(ValueError, match="缺少"):
        _write_and_validate(tmp_path, frame, unlabeled_split["csv"])


def test_validate_submission_rejects_unknown_study(
    tmp_path, predictions, unlabeled_split
):
    frame = _valid_frame(predictions).copy()
    extra = frame.iloc[[0]].copy()
    extra["Study_id"] = "study_99999"
    extra["Subject_id"] = "subject_99999"
    frame = pd.concat([frame, extra], ignore_index=True)

    with pytest.raises(ValueError, match="不存在"):
        _write_and_validate(tmp_path, frame, unlabeled_split["csv"])


def test_validate_submission_rejects_subject_mismatch(
    tmp_path, predictions, unlabeled_split
):
    frame = _valid_frame(predictions).copy()
    frame.loc[0, "Subject_id"] = "subject_99999"

    with pytest.raises(ValueError, match="Subject_id"):
        _write_and_validate(tmp_path, frame, unlabeled_split["csv"])


def test_validate_submission_rejects_invalid_class_id(
    tmp_path, predictions, unlabeled_split
):
    frame = _valid_frame(predictions).copy()
    frame.loc[0, "Predict_class"] = 11

    with pytest.raises(ValueError, match="Predict_class"):
        _write_and_validate(tmp_path, frame, unlabeled_split["csv"])


def test_validate_submission_rejects_invalid_probability(
    tmp_path, predictions, unlabeled_split
):
    frame = _valid_frame(predictions).copy()
    frame.loc[0, "Probability"] = 1.5

    with pytest.raises(ValueError, match="Probability"):
        _write_and_validate(tmp_path, frame, unlabeled_split["csv"])


def test_validate_submission_rejects_duplicate_rows(
    tmp_path, predictions, unlabeled_split
):
    frame = _valid_frame(predictions)
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="重复"):
        _write_and_validate(tmp_path, frame, unlabeled_split["csv"])


def test_build_submission_frame_rejects_invalid_threshold(predictions):
    for threshold in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError, match="threshold"):
            build_submission_frame(
                predictions, STUDY_TO_SUBJECT, threshold=threshold
            )


def test_build_submission_frame_rejects_unknown_study(predictions):
    with pytest.raises(ValueError, match="不在 study_subjects"):
        build_submission_frame(predictions, {"study_00000": "subject_00000"})


def test_validate_submission_rejects_missing_test_columns(tmp_path, predictions, write_csv):
    frame = _valid_frame(predictions)
    path = write_submission(frame, tmp_path / "submission.csv")
    test_csv = write_csv(
        "test_missing_columns.csv",
        [["pic_000000", "subject_00000"]],
        header=["Dicom_id", "Subject_id"],
    )

    with pytest.raises(ValueError, match="缺少必需的列"):
        validate_submission(path, test_csv)


def test_run_infer_rejects_out_of_range_threshold():
    import run_infer

    with pytest.raises(SystemExit):
        run_infer.parse_args(["--checkpoint", "unused.pt", "--threshold", "1.5"])


def test_validate_submission_rejects_conflicting_test_subjects(
    tmp_path, predictions, write_csv
):
    frame = _valid_frame(predictions)
    path = write_submission(frame, tmp_path / "submission.csv")
    test_csv = write_csv(
        "test_conflicting_subjects.csv",
        [
            ["pic_000000", "subject_00000", "study_00000", "AP", "Erect", "48", "64"],
            ["pic_000001", "subject_99999", "study_00000", "AP", "Erect", "48", "64"],
            ["pic_000002", "subject_00001", "study_00001", "AP", "Erect", "48", "64"],
        ],
        header=TEST_HEADER,
    )

    with pytest.raises(ValueError, match="多个 Subject_id"):
        validate_submission(path, test_csv)
