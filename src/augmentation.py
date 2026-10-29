"""训练专用轻增强：一次联合仿射，再做灰度亮度/对比度；均在Normalize前。"""
import torch
from torchvision.transforms import RandomAffine, InterpolationMode
from torchvision.transforms import functional as F

LIGHT_AUG = {
    'name': 'light_aug_v1', 'affine_probability': 0.5, 'degrees': 5.0,
    'translate': [0.02, 0.02], 'scale': [0.98, 1.02],
    'intensity_probability': 0.3, 'brightness': [0.95, 1.05],
    'contrast': [0.90, 1.10],
}


def validate_augmentation(value: dict | str) -> dict | str:
    if value == 'none':
        return value
    if not isinstance(value, dict) or value != LIGHT_AUG:
        raise ValueError('仅支持none或已定义的light_aug_v1参数，禁止静默忽略未知增强')
    return dict(value)


def apply_light_aug(image, config: dict):
    """输入为letterbox后的PIL图；同Study各图片独立采样，黑色填充。"""
    validate_augmentation(config)
    if torch.rand(()).item() < config['affine_probability']:
        angle, translate, scale, shear = RandomAffine.get_params(
            [-config['degrees'], config['degrees']], config['translate'],
            config['scale'], None, list(image.size))
        image = F.affine(image, angle, translate, scale, shear,
                         interpolation=InterpolationMode.BILINEAR, fill=0)
    if torch.rand(()).item() < config['intensity_probability']:
        factors = [float(torch.empty(()).uniform_(*config[key]))
                   for key in ('brightness', 'contrast')]
        # 只处理一个灰度通道，复制RGB后不会引入颜色差异。
        image = image.convert('L')
        image = F.adjust_brightness(image, factors[0])
        image = F.adjust_contrast(image, factors[1]).convert('RGB')
    return image
