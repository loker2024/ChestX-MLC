"""训练期模型指数移动平均及可续训状态。"""
import copy
import torch
from torch import nn


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float):
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        source_state = model.state_dict()
        for key, value in self.model.state_dict().items():
            source = source_state[key].detach()
            if value.is_floating_point():
                value.lerp_(source, 1.0 - self.decay)
            else:
                value.copy_(source)
        self.updates += 1

    def state_dict(self) -> dict:
        return {'model': self.model.state_dict(), 'updates': self.updates}

    def load_state_dict(self, state: dict) -> None:
        self.model.load_state_dict(state['model'])
        self.updates = int(state['updates'])
