"""图像预处理：letterbox 缩放 + 归一化，训练时再叠一层轻量增强。

所有 split 共用同一套几何流程——等比缩放到最长边等于 size，再把短边居中补黑边
成 size×size。直接拉伸成方形会让不同长宽比的胸片变形（肋骨、心脏轮廓的比例
会变），补边则保留了原始比例，训练和推理看到的输入分布也一致。

返回的 transform 输入是 PIL Image，输出是 [3, size, size] 的 float32 张量。
"""

from typing import Callable

import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from common.config import ImageConfig
from common.constants import IMAGENET_MEAN, IMAGENET_STD


class Letterbox:
    """等比缩放并补边：最长边缩到 size，再居中补黑边成 size×size。

    举例：原图 2000×1000、size=320 -> 缩成 320×160，左右各补 80 列黑边。

    补黑边（像素值 0）而不是复制边缘像素，是为了让模型看到的「无效区域」
    含义稳定：正常归一化后黑边是接近 -mean/std 的常数，网络很容易学会忽略。
    """

    def __init__(self, size: int, fill: int = 0) -> None:
        self.size = size
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        # 赛题图片按 RGB 读取；万一遇到灰度或带 alpha 的图，先统一成三通道。
        image = image.convert("RGB")
        width, height = image.size

        scale = self.size / max(width, height)
        new_width = max(1, round(width * scale))  # 至少 1 像素，否则 resize 会报错
        new_height = max(1, round(height * scale))
        resized = image.resize((new_width, new_height), Image.Resampling.BILINEAR)

        canvas = Image.new("RGB", (self.size, self.size), (self.fill,) * 3)
        canvas.paste(resized, ((self.size - new_width) // 2, (self.size - new_height) // 2))
        return canvas


def build_eval_transform(
    size: int, letterbox: bool = True, normalize: str = "imagenet"
) -> Callable[[Image.Image], torch.Tensor]:
    """验证 / 推理用的确定性预处理：缩放（letterbox 或直接拉伸）-> 归一化。

    这里不能有任何随机操作，否则同一张图两次推理结果不同，指标和提交都会抖。
    """
    geometry = [Letterbox(size)] if letterbox else [T.Resize((size, size), antialias=True)]
    return T.Compose([*geometry, *_normalize_steps(normalize)])


def build_train_transform(size: int, cfg: ImageConfig) -> Callable[[Image.Image], torch.Tensor]:
    """训练用预处理：与验证相同的几何流程 + 小角度旋转、缩放、亮度 / 对比度抖动。

    增强只做「不改变语义」的几何与光照扰动：

    - 旋转 ±7°、缩放 0.9~1.1、平移 5%：胸片拍摄时患者体位本来就有轻微差异；
    - 亮度 / 对比度 ±15%：不同设备的曝光和窗宽窗位不一样。

    是否水平翻转由 `cfg.hflip` 决定，默认关闭：胸片左右肺、心脏位置都有方向
    含义，翻转会把「左侧病变」变成「右侧病变」，属于污染标签的增强。

    注意这些随机参数是「每张图各抽一次」，同一 study 的 AP 和 LATERAL 不会被
    同步扰动——它们本来就是两次不同的曝光，各自独立增强更接近真实分布。
    """
    geometry = [Letterbox(size)] if cfg.letterbox else [T.Resize((size, size), antialias=True)]

    steps: list[Callable] = [
        *geometry,
        T.RandomAffine(
            degrees=7,
            translate=(0.05, 0.05),
            scale=(0.9, 1.1),
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        ),
        T.ColorJitter(brightness=0.15, contrast=0.15),
    ]
    if cfg.hflip:
        steps.append(T.RandomHorizontalFlip(p=0.5))
    steps.extend(_normalize_steps(cfg.normalize))
    return T.Compose(steps)


def _normalize_steps(normalize: str) -> list[Callable]:
    """把 PIL 图转成张量，并按配置决定要不要减均值除方差。

    ToTensor 自己会把 0~255 的整数缩到 0~1，所以 "none" 模式下就是「只缩放
    像素到 0~1」，与配置里写的含义一致。
    """
    if normalize == "imagenet":
        return [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return [T.ToTensor()]
