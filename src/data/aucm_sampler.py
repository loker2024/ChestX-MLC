"""AUCM 按 study、按目标类别采样，确保每批同时包含正负样本。"""

import math
import numpy as np
from torch.utils.data import Sampler


class AUCMBatchSampler(Sampler):
    """每批均衡一个类别，类别按随机轮次覆盖；允许稀有 study 重复。"""

    def __init__(self, labels: np.ndarray, batch_size: int, seed: int) -> None:
        self.labels = np.asarray(labels)
        if self.labels.ndim != 2 or not np.isin(self.labels, [0, 1]).all():
            raise ValueError("AUCM 标签必须是二维 0/1 矩阵")
        if batch_size < 2 or batch_size % 2:
            raise ValueError("AUCM batch_studies 必须是至少为 2 的偶数")
        self.pools = [(np.flatnonzero(self.labels[:, i] == 1), np.flatnonzero(self.labels[:, i] == 0)) for i in range(self.labels.shape[1])]
        if any(len(pos) == 0 or len(neg) == 0 for pos, neg in self.pools):
            raise ValueError("AUCM 每个类别必须同时有正样本和负样本")
        self.batch_size, self.seed, self.epoch = batch_size, seed, 0
        self.num_batches = max(len(self.pools), math.ceil(len(self.labels) / batch_size))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch]))
        tasks = np.concatenate([rng.permutation(len(self.pools)) for _ in range(math.ceil(self.num_batches / len(self.pools)))])
        for task in tasks[:self.num_batches]:
            pos, neg = self.pools[int(task)]
            indices = np.concatenate((rng.choice(pos, self.batch_size // 2, replace=True), rng.choice(neg, self.batch_size // 2, replace=True)))
            rng.shuffle(indices)
            yield [(int(index), int(task)) for index in indices]
