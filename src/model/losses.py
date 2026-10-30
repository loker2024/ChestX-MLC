"""损失函数：`BCEWithLogitsLoss`（可选按类别频率加权）与非对称损失 ASL。

10 个类别是互相独立的二分类，所以用 BCE 而不是把 10 类当成互斥的 softmax：
一张胸片可以同时有结节和积液。`BCEWithLogitsLoss` 把 sigmoid 和交叉熵合在
一个算子里，数值上比「先 sigmoid 再 BCE」更稳（sigmoid 饱和时不会出现
log(0)），因此模型输出 logits，不接 sigmoid。

损失由 `train.loss` 选择：

- `bce`（默认）：普通二分类交叉熵。类别 10（"无异常"）在数据里单独出现
  1 万多次，别的类别占比很低；要给正样本加权，把 `train.pos_weight` 打开，
  权重按「负样本数 / 正样本数」算。
- `asl`：非对称损失（Asymmetric Loss，Ben-Baruch et al., ICCV 2021）。它
  不改变正样本项、只重新加权负样本项，是处理多标签长尾的常用替代。ASL 自带
  正负样本非对称重加权，因此不能与 `pos_weight` 叠加（配置校验会报错）。
"""

import numpy as np
import torch
from torch import nn

from common.config import Config


class AsymmetricLoss(nn.Module):
    """多标签非对称损失（ASL）。

    与 BCE 的差别集中在负样本项，分两步：

    1. 概率移位（probability shifting）：负样本概率先加 `clip` 再取对数，
       等价于 `max(p - clip, 0)`，把简单负样本的概率整体往 0 推；
    2. 非对称聚焦：负样本项乘 `(1 - p_neg)^gamma_neg`，`gamma_neg` 越大，
       简单负样本被压制得越狠；正样本项默认不衰减（`gamma_pos=0`）。

    本项目类别极不平衡（类别 10 占多数、其余类正样本很少），ASL 让「海量
    容易的负样本」不再主导梯度，等效于一种只对负样本生效的重加权，通常比
    在 BCE 上叠加 pos_weight 更稳。

    超参默认取论文值：`gamma_pos=0`、`gamma_neg=4`、`clip=0.05`。
    """

    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if gamma_pos < 0 or gamma_neg < 0:
            raise ValueError(
                f"ASL 的 gamma 不能为负，当前 gamma_pos={gamma_pos}、gamma_neg={gamma_neg}"
            )
        if not 0.0 <= clip < 1.0:
            raise ValueError(f"ASL 的 clip 必须落在 [0, 1)，当前为 {clip}")
        if eps <= 0:
            raise ValueError(f"ASL 的 eps 必须 > 0，当前为 {eps}")
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """返回按元素取平均的 loss；logits / targets 形状都是 `[batch, 类别数]`。"""
        targets = targets.to(dtype=torch.float32)
        # 混合精度下先统一到 fp32：fp16 的 sigmoid 在 logits 很负时会下溢到 0，
        # 取对数后虽然被 clamp 保住，但梯度会整片消失。
        probs_pos = torch.sigmoid(logits.float())
        probs_neg = 1.0 - probs_pos
        if self.clip > 0:
            probs_neg = (probs_neg + self.clip).clamp(max=1.0)

        loss_pos = targets * torch.log(probs_pos.clamp(min=self.eps))
        loss_neg = (1.0 - targets) * torch.log(probs_neg.clamp(min=self.eps))
        loss = loss_pos + loss_neg

        if self.gamma_pos > 0 or self.gamma_neg > 0:
            pt = probs_pos * targets + probs_neg * (1.0 - targets)
            gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            loss = loss * torch.pow(1.0 - pt, gamma)

        return -loss.mean()


def build_loss(
    cfg: Config,
    train_labels: np.ndarray | None = None,
    device: str | torch.device | None = None,
) -> nn.Module:
    """构建训练损失。

    - `train.loss="asl"` 时返回 `AsymmetricLoss`，超参取 `train.asl_*`；
    - `train.loss="bce"`（默认）时返回 `BCEWithLogitsLoss`；`train.pos_weight=True`
      时启用 `pos_weight`：第 c 类的权重取「训练集负样本数 / 正样本数」。
      稀有类的正样本因此获得更大权重，模型不敢轻易把稀有类全判成负；
    - 启用 pos_weight 时必须传入 `train_labels`（`[num_studies, 10]` 的 0/1
      矩阵，通常由训练集的 `StudyRecord.labels` 纵向拼成），否则没法统计频率；
    - `device` 是损失张量（pos_weight）要放置的设备。训练/验证时必须和
      logits 使用同一设备，否则 PyTorch 会报 "Expected all tensors to be on
      the same device"；不传时留在 CPU，便于 CPU 单测；
    - pos_weight 只作用于训练损失，推理和指标仍用原始概率，不受影响。
    """
    if cfg.train.loss == "asl":
        criterion = AsymmetricLoss(
            gamma_pos=cfg.train.asl_gamma_pos,
            gamma_neg=cfg.train.asl_gamma_neg,
            clip=cfg.train.asl_clip,
        )
        return criterion.to(device) if device is not None else criterion

    if not cfg.train.pos_weight:
        return nn.BCEWithLogitsLoss().to(device) if device is not None else nn.BCEWithLogitsLoss()

    if train_labels is None:
        raise ValueError(
            "train.pos_weight 已开启，但没有传入 train_labels，无法统计每个类别的正负样本数"
        )

    labels = np.asarray(train_labels, dtype=np.float64)
    positives = labels.sum(axis=0)  # [C]，每类的正样本数
    negatives = len(labels) - positives  # [C]，每类的负样本数

    # 只有“既有正样本又有负样本”的类别才需要按频率加权：
    # - 正样本数为 0：类别在训练集里没出现，权重给 1.0；
    # - 负样本数为 0：类别全是正样本，若权重给 0 会让正样本 loss 恒为 0，
    #   模型完全学不到这一类，因此也给 1.0。
    pos_weight = np.where(
        (positives > 0) & (negatives > 0),
        negatives / np.maximum(positives, 1.0),
        1.0,
    )

    return nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device)
    )
