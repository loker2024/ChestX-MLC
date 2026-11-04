"""逐标签多标签ASL；沿用原作者标准AsymmetricLoss的detached focusing约定。"""
import math

import torch
from torch import nn
from torch.nn import functional as F

ASL_PARAMETERS = {'gamma_pos': 1, 'gamma_neg': 4, 'clip': .05, 'eps': 1e-8,
                  'detach_focal_weight': True}


class AsymmetricLabelLoss(nn.Module):
    """返回[B,10]，不求sum；类别/图片/Study归约由原训练链路处理。"""
    def __init__(self, gamma_pos: float = 1, gamma_neg: float = 4, clip: float = .05,
                 eps: float = 1e-8, detach_focal_weight: bool = True) -> None:
        super().__init__()
        if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in (gamma_pos,gamma_neg,clip,eps)):
            raise ValueError('ASL参数必须有限')
        if min(gamma_pos,gamma_neg)<0 or not 0<=clip<1 or not 0<eps<.5 or type(detach_focal_weight) is not bool:
            raise ValueError('ASL参数范围错误')
        self.gamma_pos,self.gamma_neg,self.clip,self.eps=gamma_pos,gamma_neg,clip,eps
        self.detach_focal_weight=detach_focal_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.ndim!=2 or logits.shape!=targets.shape or logits.shape[1]!=10 or not len(logits):
            raise ValueError('ASL输入和目标须为非空且相同[B,10]')
        if not logits.is_floating_point() or not torch.isfinite(logits).all():
            raise ValueError('ASL需要有限浮点Logits')
        if not torch.isfinite(targets).all() or not ((targets==0)|(targets==1)).all():
            raise ValueError('ASL标签须明确0/1，未知标签不能补阴性')
        # 退化设置直接使用logits稳定BCE，极端值也不引入概率下限误差。
        if self.gamma_pos==self.gamma_neg==self.clip==0:
            return F.binary_cross_entropy_with_logits(logits,targets.to(logits.dtype),reduction='none')
        x=logits if logits.dtype in (torch.float32,torch.float64) else logits.float()
        y=targets.to(x.dtype)
        pos=x.sigmoid(); neg=(torch.sigmoid(-x)+self.clip).clamp(max=1)
        log_pos=F.logsigmoid(x).clamp(min=math.log(self.eps))
        log_neg=neg.clamp(min=self.eps).log()
        pt=y*pos+(1-y)*neg
        weight=(1-pt).pow(y*self.gamma_pos+(1-y)*self.gamma_neg)
        if self.detach_focal_weight:
            weight=weight.detach()
        return -(y*log_pos+(1-y)*log_neg)*weight
