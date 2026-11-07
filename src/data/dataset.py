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

from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

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
        preserve_mode: bool = False,
    ) -> None:
        self.records = records
        self.transform = transform
        self.preserve_mode = preserve_mode

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        task = None
        if isinstance(index, tuple):
            index, task = index
        record = self.records[index]
        images = [self.transform(load_image(path, convert_rgb=not self.preserve_mode)) for path in record.image_paths]

        item: dict[str, Any] = {
            "images": torch.stack(images),  # [n, 3, size, size]，n = 这次检查的图片数
            "study_id": record.study_id,
            "views": torch.tensor(record.views or (2,) * len(images), dtype=torch.long),
        }
        if record.labels is not None:
            item["labels"] = torch.from_numpy(record.labels)
        if task is not None:
            item["task_id"] = task
        return item


def load_image(path: Path, convert_rgb: bool = True) -> Image.Image:
    """读一张图；默认统一成 RGB，`convert_rgb=False` 时保留原模式（如 L）。

    用 with 打开是为了读完立刻释放文件句柄——全量数据几万张图、num_workers
    又有多个进程，句柄泄漏在长训练里会变成 "too many open files"。

    推理脚本（infer.predict.predict_image / predict_study）也复用这个函数，
    保证「数据集里读图」和「单张计时里读图」是同一段代码、同一种 RGB 转换。
    单图计时传 `convert_rgb=False`：灰度图在 L 模式下 resize 更快，最终
    `Letterbox` 会在 resize 之后转 RGB，与先转 RGB 再 resize 的结果逐像素一致。
    """
    with Image.open(path) as image:
        return image.convert("RGB") if convert_rgb else image.copy()


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
        "views": torch.cat([item["views"] for item in batch]),
        "offsets": offsets,
        "study_ids": [item["study_id"] for item in batch],
    }
    if "labels" in batch[0]:
        collated["labels"] = torch.stack([item["labels"] for item in batch])
    if "task_id" in batch[0]:
        collated["task_id"] = torch.tensor([item["task_id"] for item in batch], dtype=torch.long)
    return collated


def patient_count_weights(records: list[StudyRecord], threshold: int, target_share: float) -> torch.Tensor:
    """按患者在本训练集中的检查次数给每个 study 一个采样权重。

    组一：患者检查次数 >= threshold（“多检查”）；组二：其余。
    每组的权重总和等于目标占比，因此期望上抽到多检查组的 study 比例为 target_share，
    组内各 study 权重相同，组内顺序不变。只使用训练集记录，不读取 valid 或 test。
    """
    counts = Counter(record.subject_id for record in records)
    multi = np.array([counts[record.subject_id] >= threshold for record in records], dtype=np.float64)
    total = len(records)
    if multi.sum() == 0 or multi.sum() == total:
        raise ValueError("patient_count 采样要求训练集中两组 study 都非空")
    current = multi.sum() / total
    weights = np.where(multi > 0, target_share / current, (1.0 - target_share) / (1.0 - current))
    return torch.as_tensor(weights, dtype=torch.float64)


def build_loader(
    cfg: Config, split: str, train: bool = False, pin_memory: bool = False,
    manifest_path: str | Path | None = None,
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
            cfg.image.size,
            cfg.image.letterbox,
            cfg.image.normalize,
            cfg.image.in_channels,
            cfg.image.xrv_maxval,
        )
        batch_size = cfg.eval.batch_studies

    if cfg.data.resplit.enabled and split in {"train", "valid"}:
        from data.resplit import load_resplit_records
        records = load_resplit_records(cfg, split, manifest_path)
    else:
        records = read_split(split_cfg.csv, split_cfg.images, labeled=split != "test")
    dataset = StudyDataset(records, loader_transform, preserve_mode=cfg.image.normalize == "xrv")
    if train and cfg.train.loss == "aucm":
        import numpy as np
        from data.aucm_sampler import AUCMBatchSampler
        labels = np.stack([record.labels for record in records])
        return DataLoader(dataset, batch_sampler=AUCMBatchSampler(labels, batch_size, cfg.seed),
                          num_workers=cfg.train.num_workers, collate_fn=collate_studies, pin_memory=pin_memory)
    if train and cfg.train.study_sampler == "patient_count":
        # 有放回的加权采样：每个 epoch 仍抽 len(records) 个 study，与 uniform 的步数一致。
        sampler = WeightedRandomSampler(
            patient_count_weights(records, cfg.train.study_sampler_threshold, cfg.train.study_sampler_target_share),
            num_samples=len(records), replacement=True,
            generator=torch.Generator().manual_seed(cfg.seed),
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=cfg.train.num_workers,
            collate_fn=collate_studies,
            pin_memory=pin_memory,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_studies,
        # 固定内存只在 CPU→GPU 拷贝时有意义；由入口按实际设备显式传入。
        pin_memory=pin_memory,
    )
