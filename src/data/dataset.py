"""study 级数据集、变长 batch 拼接与 DataLoader 构建。

一批样本的「大小」按检查数计（batch_studies），但一次检查有 1~7 张图，
图片数因此是变长的。常见做法是在图片维度补齐成 [B, max_n, C, H, W]，
但补出来的假图要额外算进前向、还要在损失和聚合里遮掉，既浪费又容易写错。

这里改成「首尾相接」：
    3 个 study 分别有 2、1、4 张图
    images  -> [7, 3, size, size]，7 张图按 study 顺序排成一串
    offsets -> [0, 2, 3, 7]，第 i 个 study 占 images[offsets[i]:offsets[i+1]]
模型只对这批真实图片做一次 forward，再用 offsets 求每个 study 的均值，
没有一张假图，也与「同一 study 内多图 logits 取均值」的聚合口径一致。
"""

from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from common.config import Config
from data.index import StudyRecord, read_split
from data.transforms import build_eval_transform, build_train_transform

# Config.data 里的三个字段名，同时也是 build_loader 接受的 split 取值。
SPLITS = ("train", "valid", "test")


class StudyDataset(Dataset):
    """以 study 为单位的 Dataset：取一个样本 = 取一次检查的全部图片 + 标签。"""

    def __init__(
        self,
        records: list[StudyRecord],
        transform: Callable[[Image.Image], torch.Tensor],
    ) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        images = [self.transform(load_image(path)) for path in record.image_paths]

        item: dict[str, Any] = {
            "images": torch.stack(images),  # [n, 3, size, size]，n = 这次检查的图片数
            "study_id": record.study_id,
        }
        if record.labels is not None:
            item["labels"] = torch.from_numpy(record.labels)
        return item


def load_image(path: Path) -> Image.Image:
    """读一张图并统一成 RGB。

    用 with 打开是为了读完立刻释放文件句柄——全量数据几万张图、num_workers
    又有多个进程，句柄泄漏在长训练里会变成 "too many open files"。

    推理脚本（infer.predict.predict_image / predict_study）也复用这个函数，
    保证「数据集里读图」和「单张计时里读图」是同一段代码、同一种 RGB 转换。
    """
    with Image.open(path) as image:
        return image.convert("RGB")


def collate_studies(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """把若干个 study 样本拼成一个 batch。

    返回的字典：

    - `images`：`[N, 3, H, W]`，N 为本批所有 study 的图片数之和；
    - `offsets`：`[B + 1]`，第 i 个 study 的图片是 `images[offsets[i]:offsets[i+1]]`；
    - `labels`：`[B, 10]`，只有带标注的 split 才有这个键（test 没有）；
    - `study_ids`：`list[str]`，长度 B，写 submission 时按它对齐 Study_id。
    """
    images = torch.cat([item["images"] for item in batch], dim=0)

    # cumsum 得到 [n0, n0+n1, ...] 的累计张量，前面补一个 0 就变成区间边界。
    counts = torch.tensor([item["images"].shape[0] for item in batch], dtype=torch.long)
    offsets = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(dim=0)])

    collated: dict[str, Any] = {
        "images": images,
        "offsets": offsets,
        "study_ids": [item["study_id"] for item in batch],
    }
    if "labels" in batch[0]:
        collated["labels"] = torch.stack([item["labels"] for item in batch])
    return collated


def build_loader(
    cfg: Config, split: str, train: bool = False, pin_memory: bool = False
) -> DataLoader:
    """按配置构建某个 split 的 DataLoader。

    - `train=True`：用带随机增强的预处理器、打乱 study 顺序、按
      `cfg.train.batch_studies` 取批；
    - `train=False`：用确定性预处理、保持 CSV 顺序、按 `cfg.eval.batch_studies`
      取批，验证和推理共用；
    - test split 的 CSV 没有 Predict_class 列，读进来时标签就是 None，
      最终 batch 里不会出现 `labels` 键；
    - `pin_memory` 由入口按实际设备传入：只有 CUDA 训练/推理才值得开启，
      CPU 或 MPS 上保持 False，避免无意义的固定内存和弃用告警。

    打乱顺序按 study 而不是按图片进行：一个 study 的所有图必须在同一批里，
    否则图片维度上的均值聚合就跨批了。
    """
    if split not in SPLITS:
        raise ValueError(f"split 只能是 {SPLITS} 之一，当前为 {split!r}")

    split_cfg = getattr(cfg.data, split)

    if train:
        loader_transform = build_train_transform(cfg.image.size, cfg.image)
        batch_size = cfg.train.batch_studies
    else:
        loader_transform = build_eval_transform(
            cfg.image.size, cfg.image.letterbox, cfg.image.normalize
        )
        batch_size = cfg.eval.batch_studies

    records = read_split(split_cfg.csv, split_cfg.images, labeled=split != "test")
    return DataLoader(
        StudyDataset(records, loader_transform),
        batch_size=batch_size,
        shuffle=train,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_studies,
        # 固定内存只在 CPU→GPU 拷贝时有意义；由入口按实际设备显式传入。
        pin_memory=pin_memory,
    )
