"""study 级数据集：张量形状、letterbox 行为、变长 collate 与 DataLoader 构建。"""

import numpy as np
import pytest
import torch
from PIL import Image

from common.config import ImageConfig, load_config
from data.dataset import StudyDataset, build_loader, collate_studies
from data.index import read_split
from data.transforms import Letterbox, build_eval_transform, build_train_transform


def _config(labeled_split, unlabeled_split, write_config):
    """把两个迷你 split 写成配置文件：train/valid 带标注，test 不带。"""
    data = {
        "train": (labeled_split["csv"], labeled_split["images"]),
        "valid": (labeled_split["csv"], labeled_split["images"]),
        "test": (unlabeled_split["csv"], unlabeled_split["images"]),
    }
    return load_config(write_config(data))


def _dataset(csv_path, images_root, labeled: bool = True) -> StudyDataset:
    records = read_split(csv_path, images_root, labeled=labeled)
    return StudyDataset(records, build_eval_transform(32))


def test_eval_transform_output_shape_and_range():
    image = Image.new("RGB", (64, 48), (255, 255, 255))

    tensor = build_eval_transform(32, normalize="none")(image)

    assert tensor.shape == (3, 32, 32)
    assert tensor.dtype == torch.float32
    assert 0.0 <= float(tensor.min()) and float(tensor.max()) <= 1.0


def test_eval_transform_is_deterministic():
    """验证/推理的预处理不能有随机性，否则同一张图两次结果不同。"""
    image = Image.new("RGB", (64, 48), (200, 100, 50))
    transform = build_eval_transform(32)

    assert torch.equal(transform(image), transform(image))


def test_letterbox_keeps_aspect_ratio():
    """64×48 缩到最长边 32 后是 32×24，上下各补 4 行黑边，不会被拉成方形。"""
    image = Image.new("RGB", (64, 48), (255, 255, 255))

    letterboxed = Letterbox(32)(image)
    assert letterboxed.size == (32, 32)

    pixels = np.asarray(letterboxed)
    filled_rows = np.where(pixels.any(axis=(1, 2)))[0]
    assert filled_rows.tolist() == list(range(4, 28))
    assert (pixels[:4] == 0).all() and (pixels[28:] == 0).all()


def test_train_transform_output_shape():
    image = Image.new("RGB", (50, 70), (128, 128, 128))

    tensor = build_train_transform(32, ImageConfig(size=32))(image)

    assert tensor.shape == (3, 32, 32)
    assert torch.isfinite(tensor).all()


def test_dataset_item_has_images_and_labels(labeled_split):
    dataset = _dataset(labeled_split["csv"], labeled_split["images"])

    item = dataset[0]

    assert item["study_id"] == "study_00000"
    # 这个 study 有 2 张图，所以第一维是 2
    assert item["images"].shape == (2, 3, 32, 32)
    assert item["labels"].shape == (10,)
    assert item["labels"].dtype == torch.float32


def test_collate_studies_packs_variable_length(labeled_split):
    """3 张图 / 2 个 study：图片首尾相接，offsets 记下每个 study 的区间。"""
    dataset = _dataset(labeled_split["csv"], labeled_split["images"])

    batch = collate_studies([dataset[0], dataset[1]])

    assert batch["images"].shape == (3, 3, 32, 32)
    assert batch["offsets"].tolist() == [0, 2, 3]
    assert batch["labels"].shape == (2, 10)
    assert batch["study_ids"] == ["study_00000", "study_00001"]
    # 拼包只是拼接，不改动任何一个像素
    torch.testing.assert_close(batch["images"][:2], dataset[0]["images"])


def test_collate_studies_without_labels(unlabeled_split):
    dataset = _dataset(unlabeled_split["csv"], unlabeled_split["images"], labeled=False)

    batch = collate_studies([dataset[0], dataset[1]])

    assert "labels" not in batch
    assert batch["offsets"].tolist() == [0, 2, 3]


def test_build_loader_train_batches_by_study(labeled_split, unlabeled_split, write_config):
    config = _config(labeled_split, unlabeled_split, write_config)

    # 训练用带增强的预处理并打乱 study 顺序，所以只断言与顺序无关的性质
    batch = next(iter(build_loader(config, "train", train=True)))
    assert batch["images"].shape[0] == 3
    assert batch["offsets"].tolist()[0] == 0
    assert batch["offsets"].tolist()[-1] == 3
    assert batch["labels"].shape == (2, 10)
    assert len(batch["study_ids"]) == 2

    # 验证按 CSV 顺序，区间是确定的
    eval_batch = next(iter(build_loader(config, "valid")))
    assert eval_batch["study_ids"] == ["study_00000", "study_00001"]
    assert eval_batch["offsets"].tolist() == [0, 2, 3]

    # test 没有标注列，batch 里也就没有 labels
    test_batch = next(iter(build_loader(config, "test")))
    assert "labels" not in test_batch
    assert test_batch["study_ids"] == ["study_00000", "study_00001"]


def test_build_loader_rejects_unknown_split(labeled_split, unlabeled_split, write_config):
    config = _config(labeled_split, unlabeled_split, write_config)

    with pytest.raises(ValueError, match="split"):
        build_loader(config, "dev")


def test_view_ids_remain_aligned_with_packed_images(labeled_split):
    dataset = _dataset(labeled_split["csv"], labeled_split["images"])
    batch = collate_studies([dataset[0], dataset[1]])
    assert dataset.records[0].views == (0, 1)
    assert batch["views"].tolist() == [0, 1, 0]
