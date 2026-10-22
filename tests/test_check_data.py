"""数据体检脚本：ID 格式、缺图、同 study 标签与跨 split 重叠。"""

import pandas as pd
import pytest

from check_data import (
    Checker,
    check_ids,
    check_images,
    check_labels,
    check_overlaps,
    read_split,
)


def test_read_split_keeps_ids_as_strings(labeled_split):
    """CSV 里的 ID 按字符串读入，前导零不能在 pandas 这一层丢掉。"""
    frame = read_split(labeled_split["csv"], labeled=True, full=False)

    assert frame["Subject_id"].tolist() == ["subject_00000", "subject_00000", "subject_00001"]
    assert frame["Study_id"].tolist() == ["study_00000", "study_00000", "study_00001"]
    assert frame["Dicom_id"].tolist() == ["pic_000010", "pic_000011", "pic_000012"]


def test_read_split_requires_label_column(unlabeled_split):
    """test split 默认也按「带标签」读取，缺列时要明确报错。"""
    with pytest.raises(ValueError, match="Predict_class"):
        read_split(unlabeled_split["csv"], labeled=True, full=False)


def test_check_ids_flags_truncated_number(tmp_path):
    """`subject_0` 这类被截断的编号不符合固定宽度，应该报错。"""
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text(
        "Dicom_id,Subject_id,Study_id,Predict_class\n"
        "pic_000001,subject_0,study_00001,1\n",
        encoding="utf-8",
    )
    frame = read_split(csv_path, labeled=True, full=False)
    checker = Checker()

    check_ids(frame, checker)

    assert any("Subject_id" in failure for failure in checker.failures)


def test_check_images_counts_found_and_missing(labeled_split, write_csv):
    """缺图时打印「找到数 / 总数」，而不是两个相同的总数。"""
    rows = [
        *labeled_split["rows"],
        ["pic_999999", "subject_00002", "study_00002", "AP", "Erect", "48", "64", "1"],
    ]
    frame = read_split(write_csv("missing.csv", rows), labeled=True, full=False)
    checker = Checker()

    check_images(frame, labeled_split["images"], checker)

    assert len(checker.failures) == 1
    assert "图片齐全（3 / 4）" in checker.failures[0]
    assert "缺少 1 张" in checker.failures[0]


def test_check_labels_ignores_order_but_flags_different_set(labeled_split, write_csv):
    """`7;6` 与 `6;7` 是同一组标签；换成 `7` 才是真的不一致。"""
    reordered = [list(row) for row in labeled_split["rows"]]
    reordered[1][-1] = "6;7"
    frame = read_split(write_csv("reordered.csv", reordered), labeled=True, full=False)
    checker = Checker()
    check_labels(frame, checker)
    assert checker.failures == []

    conflicting = [list(row) for row in labeled_split["rows"]]
    conflicting[1][-1] = "7"
    frame = read_split(write_csv("conflicting.csv", conflicting), labeled=True, full=False)
    checker = Checker()
    check_labels(frame, checker)
    assert any("标签一致" in failure for failure in checker.failures)


def test_check_overlaps_flags_shared_subject():
    """不同 split 出现同一个患者，说明划分泄漏。"""
    def make_frame(subject: str, study: str, dicom: str) -> pd.DataFrame:
        return pd.DataFrame(
            {"Subject_id": [subject], "Study_id": [study], "Dicom_id": [dicom]}
        )

    frames = {
        "train": make_frame("subject_00000", "study_00000", "pic_000000"),
        "valid": make_frame("subject_00000", "study_00001", "pic_000001"),
        "test": make_frame("subject_00002", "study_00002", "pic_000002"),
    }
    checker = Checker()

    check_overlaps(frames, checker, full=True)

    assert any("train/valid 的 Subject_id" in failure for failure in checker.failures)
    # 同一患者但 study 不同，Study_id 这一项本身不应误报
    assert not any("train/valid 的 Study_id" in failure for failure in checker.failures)