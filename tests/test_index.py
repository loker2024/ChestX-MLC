"""图片索引与 study 聚合：前导零保留、多图归并、缺图与标签一致性校验。"""

import pytest

from data.index import index_images, read_split


def test_index_images_maps_stem_to_path(make_image):
    """索引的键是文件名（去掉后缀），值是可读的图片路径。"""
    first = make_image("subject_00000/study_00000/pic_000000.png")
    second = make_image("subject_00001/study_00001/pic_000012.jpg")
    # 非图片文件不该进索引
    (first.parent / "notes.txt").write_text("ignore me", encoding="utf-8")

    index = index_images(first.parents[2])

    assert index == {"pic_000000": first, "pic_000012": second}


def test_index_images_accepts_uppercase_suffix(make_image):
    """后缀比较前统一转小写，所以 .JPG 也能命中。"""
    source = make_image("subject_00000/study_00000/pic_000000.png")
    uppercase = source.with_suffix(".JPG")
    source.rename(uppercase)

    assert index_images(uppercase.parents[2]) == {"pic_000000": uppercase}


def test_index_images_duplicate_stem_raises(make_image):
    """不同目录下的同名图片会让 Dicom_id 指向两个文件，必须报错。"""
    first = make_image("subject_00000/study_00000/pic_000000.png")
    make_image("subject_00001/study_00001/pic_000000.jpg")

    with pytest.raises(ValueError, match="重复"):
        index_images(first.parents[2])


def test_index_images_missing_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="不存在"):
        index_images(tmp_path / "no_such_dir")


def test_read_split_aggregates_images_per_study(labeled_split):
    """同 Study_id 的两行要合成一条记录，标签与图片顺序都要保住。"""
    records = read_split(labeled_split["csv"], labeled_split["images"])

    # 顺序 = study 在 CSV 里第一次出现的顺序
    assert [record.study_id for record in records] == ["study_00000", "study_00001"]

    first = records[0]
    assert first.subject_id == "subject_00000"
    assert first.num_images == 2
    # 图片顺序与 CSV 一致，文件名里的前导零原样保留
    assert [path.stem for path in first.image_paths] == ["pic_000010", "pic_000011"]
    # "7;6" -> 下标 5、6 为 1
    assert first.labels.tolist() == [0, 0, 0, 0, 0, 1, 1, 0, 0, 0]

    second = records[1]
    assert second.num_images == 1
    assert second.labels.tolist() == [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]


def test_read_split_missing_image_raises(labeled_split, write_csv):
    rows = [
        *labeled_split["rows"],
        ["pic_999999", "subject_00002", "study_00002", "AP", "Erect", "48", "64", "1"],
    ]
    csv_path = write_csv("broken.csv", rows)

    with pytest.raises(FileNotFoundError, match="pic_999999"):
        read_split(csv_path, labeled_split["images"])


def test_read_split_inconsistent_labels_raises(labeled_split, write_csv):
    """同一 study 的两张图标签不同，说明标注导出有问题，不能取其一了事。"""
    rows = [list(row) for row in labeled_split["rows"]]
    rows[1][-1] = "6"  # 原本是 "7;6"
    csv_path = write_csv("inconsistent.csv", rows)

    with pytest.raises(ValueError, match="标签不一致"):
        read_split(csv_path, labeled_split["images"])


def test_read_split_without_labels(unlabeled_split):
    """test split 没有 Predict_class 列，labeled=False 时记录里不带标签。"""
    records = read_split(unlabeled_split["csv"], unlabeled_split["images"], labeled=False)

    assert len(records) == 2
    assert all(record.labels is None for record in records)
    # 图片仍然要正常配上
    assert sum(record.num_images for record in records) == 3


def test_read_split_requires_label_column(unlabeled_split):
    """默认 labeled=True，缺 Predict_class 列时要明确报错。"""
    with pytest.raises(ValueError, match="Predict_class"):
        read_split(unlabeled_split["csv"], unlabeled_split["images"])


def test_read_split_duplicate_dicom_raises(labeled_split, write_csv):
    """同一 Dicom_id 出现两次会让同一张图被重复计数，必须报错。"""
    rows = [*labeled_split["rows"], list(labeled_split["rows"][0])]
    csv_path = write_csv("duplicate_dicom.csv", rows)

    with pytest.raises(ValueError, match="Dicom_id"):
        read_split(csv_path, labeled_split["images"])


def test_read_split_inconsistent_subject_raises(labeled_split, write_csv):
    """同一 Study_id 对应两个 Subject_id，说明患者映射坏了，不能静默取第一个。"""
    rows = [list(row) for row in labeled_split["rows"]]
    rows[1][1] = "subject_00002"
    csv_path = write_csv("inconsistent_subject.csv", rows)

    with pytest.raises(ValueError, match="Subject_id"):
        read_split(csv_path, labeled_split["images"])
