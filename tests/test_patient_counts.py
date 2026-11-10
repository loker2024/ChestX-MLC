"""抓住将图片行数当检查数、ID 错配和缺失计数的回归。"""
import numpy as np
import pandas as pd
import pytest

from data.patient_counts import align_patient_counts, build_patient_counts, read_patient_metadata


def test_multiple_images_count_as_one_study():
    frame = pd.DataFrame({"Subject_id": ["001", "001", "001", "002"],
                          "Study_id": ["010", "010", "011", "012"]})
    result = build_patient_counts(frame)
    assert result["Study_id"].tolist() == ["010", "011", "012"]
    assert result["patient_study_count"].tolist() == [2, 2, 1]
    predictions = pd.DataFrame({"Study_id": ["012", "010", "011"]})
    np.testing.assert_array_equal(align_patient_counts(predictions, result), [1, 2, 2])


@pytest.mark.parametrize("patients,studies", [(["001", "002"], ["010", "010"]),
                                              (["", "002"], ["010", "011"])])
def test_conflicting_or_empty_ids_rejected(patients, studies):
    with pytest.raises(ValueError):
        build_patient_counts(pd.DataFrame({"Subject_id": patients, "Study_id": studies}))


def test_csv_keeps_leading_zeros_and_ignores_labels(tmp_path):
    path = tmp_path / "metadata.csv"
    path.write_text("Subject_id,Study_id,Predict_class\n001,002,invalid\n", encoding="utf-8")
    metadata = read_patient_metadata(path)
    assert metadata.to_dict("records") == [{"Subject_id": "001", "Study_id": "002"}]


def test_missing_or_extra_study_and_patient_mismatch_fail():
    counts = build_patient_counts(pd.DataFrame({"Subject_id": ["001"], "Study_id": ["010"]}))
    for predictions in [pd.DataFrame({"Study_id": ["999"]}),
                        pd.DataFrame({"Study_id": ["010"], "Subject_id": ["002"]})]:
        with pytest.raises(ValueError):
            align_patient_counts(predictions, counts)
