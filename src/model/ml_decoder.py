"""单层类别查询交叉注意力分类头，ML-Decoder 思路的轻量变体。"""
import torch
from torch import nn


class MLDecoderHead(nn.Module):
    """10 个可训练类别查询；每类独立投影输出一个 logit。"""
    def __init__(self, channels: int, num_classes: int) -> None:
        super().__init__()
        dim = 256
        self.project = nn.Linear(channels, dim)
        self.query = nn.Parameter(torch.randn(num_classes, dim) * 0.02)
        self.attention = nn.MultiheadAttention(dim, 4, dropout=0.0, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.feedforward = nn.Sequential(nn.Linear(dim, 512), nn.ReLU(), nn.Linear(512, dim))
        self.norm2 = nn.LayerNorm(dim)
        self.class_weight = nn.Parameter(torch.empty(num_classes, dim))
        self.class_bias = nn.Parameter(torch.zeros(num_classes))
        nn.init.normal_(self.class_weight, std=0.02)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        memory = self.project(features.flatten(2).transpose(1, 2)).relu()
        queries = self.query.unsqueeze(0).expand(features.size(0), -1, -1)
        attended = self.attention(queries, memory, memory, need_weights=False)[0]
        output = self.norm1(queries + attended)
        output = self.norm2(output + self.feedforward(output))
        return (output * self.class_weight.unsqueeze(0)).sum(dim=-1) + self.class_bias


class ConvNeXtMLDecoderHead(nn.Module):
    def __init__(self, norm: nn.Module, channels: int, num_classes: int) -> None:
        super().__init__()
        self.norm = norm
        self.decoder = MLDecoderHead(channels, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.norm(features))
