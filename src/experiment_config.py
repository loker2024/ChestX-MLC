"""P0 JSON实验配置：只允许已经实现的选项，避免静默忽略优化参数。"""
import copy
import json
import math
from pathlib import Path
import re

SUPPORTED = {
    'backbone': 'resnet18', 'optimizer': 'AdamW', 'scheduler': 'none',
    'loss_function': 'BCEWithLogitsLoss', 'augmentation': 'none',
    'sampler': 'shuffle', 'study_aggregation': 'mean',
}
REQUIRED_FIELDS = {*SUPPORTED, 'experiment_id', 'image_size', 'batch_size',
                   'epochs', 'learning_rate', 'random_seed', 'data_config'}
OPTIONAL_FIELDS = {'weight_decay', 'weights', 'notes', 'main_change'}


def validate_config(config: dict) -> dict:
    """返回独立的、规范化配置；尚未实现的选择明确报错。"""
    if not isinstance(config, dict):
        raise ValueError('实验配置必须是JSON对象')
    if REQUIRED_FIELDS - config.keys() or config.keys() - REQUIRED_FIELDS - OPTIONAL_FIELDS:
        raise ValueError(f'配置缺失或多余字段: missing={sorted(REQUIRED_FIELDS-config.keys())}; '
                         f'unknown={sorted(config.keys()-REQUIRED_FIELDS-OPTIONAL_FIELDS)}')
    result = copy.deepcopy(config)
    if not isinstance(result['experiment_id'], str) or not re.fullmatch(
            r'E[0-9]{2}_[A-Za-z0-9_-]+', result['experiment_id']):
        raise ValueError('experiment_id应为E01_resnet18等安全目录名')
    for key, expected in SUPPORTED.items():
        if result[key] != expected:
            raise NotImplementedError(f'{key}={result[key]!r}尚未实现；当前只支持{expected!r}')
    for key in ('image_size', 'batch_size', 'epochs', 'random_seed'):
        if type(result[key]) is not int or result[key] < (0 if key == 'random_seed' else 1):
            raise ValueError(f'{key}必须是合法整数')
    if result['image_size'] < 32 or result['batch_size'] < 2:
        raise ValueError('ResNet18训练要求image_size>=32且batch_size>=2')
    if result['random_seed'] >= 2**32:
        raise ValueError('random_seed必须小于2**32')
    result.setdefault('weight_decay', 0.01)
    for key in ('learning_rate', 'weight_decay'):
        value = result[key]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f'{key}必须为有限数值')
    if not 0 < result['learning_rate'] < 1 or result['weight_decay'] < 0:
        raise ValueError('学习率或weight_decay范围错误')
    result.setdefault('weights', None)
    if result['weights'] is not None:
        raise NotImplementedError('P0仅支持随机初始化weights=null，不下载预训练权重')
    if not isinstance(result['data_config'], str) or not result['data_config'].strip():
        raise ValueError('data_config必须为非空路径')
    for key in ('notes', 'main_change'):
        result.setdefault(key, '')
        if not isinstance(result[key], str):
            raise ValueError(f'{key}必须为字符串')
    return result


def load_config(path: Path) -> dict:
    """路径与现有命令一致，相对于项目根目录（命令执行目录）。"""
    return validate_config(json.loads(path.read_text(encoding='utf-8-sig')))


def load_data_config(config: dict) -> dict:
    data = json.loads(Path(config['data_config']).read_text(encoding='utf-8-sig'))
    if not isinstance(data, dict):
        raise ValueError('数据配置必须为对象')
    for split in ('train', 'valid', 'test'):
        paths = data.get(split)
        if not isinstance(paths, dict) or set(paths) != {'csv', 'images'} or not all(
                isinstance(p, str) and p.strip() for p in paths.values()):
            raise ValueError(f'{split}缺少csv/images路径')
    # 实验配置控制尺寸与训练批量；旧数据配置中的size/batch_size不再控制新入口。
    return {**data, 'size': config['image_size'], 'batch_size': config['batch_size']}
