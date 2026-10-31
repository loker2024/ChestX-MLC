"""单头类别特定残差注意力：空间类别分数的均值与注意力加权分数之和。"""

import torch
from torch import nn
from torch.nn import functional as F


class CSRAHead(nn.Module):
    """用归一化类别权重投影特征图，输出逐图原始 logits。"""

    def __init__(self, channels: int, num_classes: int, residual_weight: float, temperature: float) -> None:
        super().__init__()
        self.classifier = nn.Conv2d(channels, num_classes, kernel_size=1, bias=False)
        self.residual_weight = residual_weight
        self.temperature = temperature

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        weights = F.normalize(self.classifier.weight, p=2, dim=1)
        scores = F.conv2d(features, weights).flatten(2)
        attention = torch.softmax(scores.float() * self.temperature, dim=-1).to(scores.dtype)
        return scores.mean(dim=-1) + self.residual_weight * (scores * attention).sum(dim=-1)


class ConvNeXtCSRAHead(nn.Module):
    """复用预训练 head 的 LayerNorm 参数，对每个空间位置归一化后执行 CSRA。"""

    def __init__(self, norm: nn.Module, channels: int, num_classes: int,
                 residual_weight: float, temperature: float) -> None:
        super().__init__()
        self.norm = norm
        self.csra = CSRAHead(channels, num_classes, residual_weight, temperature)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.csra(self.norm(features))
