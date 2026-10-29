"""按验证Study级Macro-AUC早停；不将短暂波动自动诊断为过拟合。"""
from dataclasses import dataclass
import math


@dataclass
class EarlyStopping:
    patience: int = 0
    min_delta: float = 0.0
    best_score: float | None = None
    best_epoch: int | None = None
    bad_epochs: int = 0
    last_epoch: int = 0

    def __post_init__(self) -> None:
        if type(self.patience) is not int or self.patience < 0:
            raise ValueError('early-stopping-patience必须为非负整数，0表示关闭')
        if type(self.min_delta) not in (int, float) or not math.isfinite(self.min_delta) or not 0 <= self.min_delta < 1:
            raise ValueError('early-stopping-min-delta必须在[0,1)内且为有限数值')

    def step(self, score: float, epoch: int) -> bool:
        """与上一次显著改善比较，微小增益可以累计；模型选优另按实际最高AUC。"""
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('早停要求有限、已定义的验证Macro-AUC')
        if type(epoch) is not int or epoch <= self.last_epoch:
            raise ValueError('早停轮次必须为严格递增的正整数')
        self.last_epoch = epoch
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score, self.best_epoch, self.bad_epochs = score, epoch, 0
        else:
            self.bad_epochs += 1
        return self.should_stop

    @property
    def should_stop(self) -> bool:
        return self.patience > 0 and self.bad_epochs >= self.patience

    def as_dict(self) -> dict:
        return {'enabled':self.patience > 0, 'monitor':'study_macro_auc', 'mode':'max',
                'patience':self.patience, 'min_delta':self.min_delta,
                'significant_best_auc':self.best_score,'significant_best_epoch':self.best_epoch,
                'bad_epochs':self.bad_epochs,'last_epoch':self.last_epoch,
                'should_stop':self.should_stop, 'counter_source':'replayed_history_then_current_epochs'}
