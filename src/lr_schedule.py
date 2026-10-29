"""按全局epoch确定学习率；不因首轮/续训分段重启Warmup或Cosine。"""
import math


def validate_schedule(spec: dict, total_epochs: int | None = None) -> dict:
    required = {'name', 'warmup_epochs', 'total_epochs', 'start_factor', 'min_factor'}
    if not isinstance(spec, dict) or set(spec) != required or spec['name'] != 'warmup_cosine':
        raise ValueError('调度配置必须明确指定warmup_cosine及全部参数')
    if any(type(spec[k]) is not int for k in ('warmup_epochs', 'total_epochs')):
        raise ValueError('调度轮数必须为整数')
    if not 2 <= spec['warmup_epochs'] < spec['total_epochs']:
        raise ValueError('epoch级warmup至少2轮，且须小于总轮数')
    if total_epochs is not None and spec['total_epochs'] != total_epochs:
        raise ValueError('Cosine总预算必须与实验epochs一致')
    for key in ('start_factor', 'min_factor'):
        value = spec[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError('学习率倍率必须在(0,1]内')
    return dict(spec)


def epoch_factor(spec: dict, epoch: int) -> float:
    spec = validate_schedule(spec)
    if type(epoch) is not int or not 1 <= epoch <= spec['total_epochs']:
        raise ValueError('全局epoch超出预定调度预算；不能静默延长Cosine')
    warmup = spec['warmup_epochs']
    if epoch <= warmup:
        return spec['start_factor'] + (1 - spec['start_factor']) * (epoch - 1) / (warmup - 1)
    progress = (epoch - warmup) / (spec['total_epochs'] - warmup)
    return spec['min_factor'] + (1 - spec['min_factor']) * (1 + math.cos(math.pi * progress)) / 2


def apply_epoch_lr(optimizer, settings: dict, epoch: int) -> dict:
    """轮前设置；两组乘同一倍率，基础学习率及10:1比例保持不变。"""
    spec = settings.get('lr_schedule')
    if spec is not None:
        factor = epoch_factor(spec, epoch)
        for group in optimizer.param_groups:
            if group.get('name') not in ('backbone', 'head'):
                raise ValueError('Warmup+Cosine要求显式骨干/分类头参数组')
            group['lr'] = settings[group['name'] + '_lr'] * factor
    return {str(group.get('name', index)): float(group['lr'])
            for index, group in enumerate(optimizer.param_groups)}
