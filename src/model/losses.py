"""损失函数：`BCEWithLogitsLoss`，可选按训练集类别频率加权（pos_weight）。

10 个类别是互相独立的二分类，所以用 BCE 而不是把 10 类当成互斥的 softmax：
一张胸片可以同时有结节和积液。`BCEWithLogitsLoss` 把 sigmoid 和交叉熵合在
一个算子里，数值上比「先 sigmoid 再 BCE」更稳（sigmoid 饱和时不会出现
log(0)），因此模型输出 logits，不接 sigmoid。

类别不平衡：类别 10（"无异常"）在数据里单独出现 1 万多次，别的类别占比很低。
不处理时模型容易退化成「全部预测为负」也能拿到不低的 loss。要给正样本加权，
把 `train.pos_weight` 打开即可，权重按「负样本数 / 正样本数」算。
"""

import numpy as np
import torch
from torch import nn

from common.config import Config


def build_loss(cfg: Config, train_labels: np.ndarray | None = None) -> nn.Module:
    """构建训练损失。

    - 默认返回普通的 `BCEWithLogitsLoss`；
    - `cfg.train.pos_weight=True` 时启用 `pos_weight`：第 c 类的权重取
      「训练集负样本数 / 正样本数」。稀有类的正样本因此获得更大权重，
      模型不敢轻易把稀有类全判成负；
    - 启用 pos_weight 时必须传入 `train_labels`（`[num_studies, 10]` 的 0/1
      矩阵，通常由训练集的 `StudyRecord.labels` 纵向拼成），否则没法统计频率；
    - pos_weight 只作用于训练损失，推理和指标仍用原始概率，不受影响。

    `pos_weight` 是训练时的一个超参数开关，P7 会对比开 / 关两种设置的指标。
    """
    if not cfg.train.pos_weight:
        return nn.BCEWithLogitsLoss()

    if train_labels is None:
        raise ValueError(
            "train.pos_weight 已开启，但没有传入 train_labels，无法统计每个类别的正负样本数"
        )

    labels = np.asarray(train_labels, dtype=np.float64)
    positives = labels.sum(axis=0)  # [C]，每类的正样本数
    negatives = len(labels) - positives  # [C]，每类的负样本数

    # 正样本数为 0 的类别在训练集里没出现过，权重给 1.0 即可；
    # 用 np.maximum 兜住除零，避免得到 inf 后再把 loss 变成 nan。
    pos_weight = np.where(
        positives > 0, negatives / np.maximum(positives, 1.0), 1.0
    )

    return nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32)
    )
