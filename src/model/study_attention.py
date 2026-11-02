"""同一 study 内的视角感知 gated attention。"""
import torch
from torch import nn


class StudyAttention(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.view_embedding = nn.Embedding(3, hidden_dim)
        self.v = nn.Linear(feature_dim, hidden_dim)
        self.u = nn.Linear(feature_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor, offsets: torch.Tensor, views: torch.Tensor | None) -> torch.Tensor:
        if views is None:
            views = torch.full((features.shape[0],), 2, device=features.device, dtype=torch.long)
        if views.ndim != 1 or views.numel() != features.shape[0] or bool(((views < 0) | (views > 2)).any()):
            raise ValueError('views 必须包含每张图的 0、1 或 2')
        gate = torch.tanh(self.v(features) + self.view_embedding(views)) * torch.sigmoid(self.u(features))
        scores = self.score(gate).squeeze(-1).float()
        pooled = []
        for start, end in zip(offsets[:-1].tolist(), offsets[1:].tolist()):
            if end <= start:
                raise ValueError('每个 study 必须至少包含一张图')
            weights = torch.softmax(scores[start:end], dim=0).to(features.dtype)
            pooled.append((features[start:end] * weights[:, None]).sum(dim=0))
        return torch.stack(pooled)


class AttentionStudyModel(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.attention = StudyAttention(backbone.num_features)
        self.pretrained_loaded = backbone.pretrained_loaded

    def get_classifier(self) -> nn.Module:
        return self.backbone.get_classifier()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)

    def forward_studies(self, images: torch.Tensor, offsets: torch.Tensor, views: torch.Tensor | None) -> torch.Tensor:
        features = self.backbone.forward_head(self.backbone.forward_features(images), pre_logits=True)
        return self.get_classifier()(self.attention(features, offsets, views))
