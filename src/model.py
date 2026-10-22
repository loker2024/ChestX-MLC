"""不加载外部权重的十类别 ResNet18。"""
import torch
from torch import nn
from torchvision.models import resnet18


def build_model() -> nn.Module:
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model


def load_model(path: str, device: str = 'cpu') -> nn.Module:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint['architecture'] != 'resnet18' or checkpoint['num_classes'] != 10:
        raise ValueError('不支持的模型结构')
    model = build_model().to(device)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model.eval()
