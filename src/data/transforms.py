"""图像预处理：letterbox 缩放 + 归一化，训练时再叠一层轻量增强。

所有 split 共用同一套几何流程——等比缩放到最长边等于 size，再把短边居中补黑边
成 size×size。直接拉伸成方形会让不同长宽比的胸片变形（肋骨、心脏轮廓的比例
会变），补边则保留了原始比例，训练和推理看到的输入分布也一致。

返回的 transform 输入是 PIL Image，输出是 [C, size, size] 的 float32 张量。
C 默认 3（timm 骨干）；B 组 TxRV 适配时 C=1，配合 "xrv" 归一化把像素映射到
[-1024, 1024]。单通道路径只在 image.in_channels=1 时才走。
"""

from typing import Callable, Sequence

import numpy as np
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

    `channels=3`（默认）输出 RGB；`channels=1` 时全程用 PIL 的 L 模式
    （单通道），供 B 组 TxRV DenseNet 使用——单通道插值只用约 1/3 的运算量，
    且没必要把灰度胸片复制成三份再算。
    """

    def __init__(self, size: int, fill: int = 0, channels: int = 3) -> None:
        if channels not in {1, 3}:
            raise ValueError(f"Letterbox 的 channels 只能是 1 或 3，当前为 {channels}")
        self.size = size
        self.fill = fill
        self.channels = channels

    def __call__(self, image: Image.Image) -> Image.Image:
        mode = "L" if self.channels == 1 else "RGB"
        # 灰度图保留 L 模式先 resize，之后才转 RGB：缩放是逐通道的线性运算，
        # 先 resize 后 convert 与先 convert 后 resize 逐像素等价（实测张量差为 0），
        # 但单通道插值只需约 1/3 的运算量，单张推理的读图+预处理明显更快。
        if image.mode != mode:
            # 其他模式（RGB / RGBA / P 等）先统一到目标通道数，避免 alpha 参与缩放。
            image = image.convert(mode)
        width, height = image.size

        scale = self.size / max(width, height)
        new_width = max(1, round(width * scale))  # 至少 1 像素，否则 resize 会报错
        new_height = max(1, round(height * scale))
        resized = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
        if resized.mode != mode:
            resized = resized.convert(mode)

        canvas = Image.new(mode, (self.size, self.size), self.fill)
        canvas.paste(resized, ((self.size - new_width) // 2, (self.size - new_height) // 2))
        return canvas


def build_eval_transform(
    size: int,
    letterbox: bool = True,
    normalize: str = "imagenet",
    in_channels: int = 3,
    xrv_maxval: int = 255,
) -> Callable[[Image.Image], torch.Tensor]:
    """验证 / 推理用的确定性预处理：缩放（letterbox 或直接拉伸）-> 归一化。

    这里不能有任何随机操作，否则同一张图两次推理结果不同，指标和提交都会抖。
    """
    geometry = (
        [Letterbox(size, channels=in_channels)]
        if letterbox
        else [T.Resize((size, size), antialias=True)]
    )
    transform = T.Compose([*geometry, *_normalize_steps(normalize, in_channels, xrv_maxval)])
    return _XrvBitDepthTransform(transform, size, letterbox, xrv_maxval, False) if normalize == "xrv" else transform


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
    geometry = (
        [Letterbox(size, channels=cfg.in_channels)]
        if cfg.letterbox
        else [T.Resize((size, size), antialias=True)]
    )

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
    steps.extend(_normalize_steps(cfg.normalize, cfg.in_channels, cfg.xrv_maxval))
    transform = T.Compose(steps)
    return _XrvBitDepthTransform(transform, size, cfg.letterbox, cfg.xrv_maxval, True, cfg.hflip) if cfg.normalize == "xrv" else transform


class _XrvBitDepthTransform:
    """保留 16 位输入，并在任何通道转换前检查原始像素范围。"""

    def __init__(self, eight_bit: Callable, size: int, letterbox: bool, maxval: int,
                 train: bool, hflip: bool = False) -> None:
        self.eight_bit, self.size, self.letterbox, self.maxval = eight_bit, size, letterbox, maxval
        self.train, self.hflip = train, hflip
        self.augment = T.Compose([
            T.RandomAffine(degrees=7, translate=(0.05, 0.05), scale=(0.9, 1.1),
                           interpolation=InterpolationMode.BILINEAR, fill=0),
            T.ColorJitter(brightness=0.15, contrast=0.15),
            *([T.RandomHorizontalFlip(p=0.5)] if hflip else []),
        ])

    def __call__(self, image: Image.Image) -> torch.Tensor:
        pixels = np.asarray(image)
        if pixels.size == 0 or pixels.min() < 0 or pixels.max() > self.maxval:
            raise ValueError(f"图像像素超出 xrv_maxval={self.maxval} 的范围")
        if image.mode not in {"I", "I;16", "I;16B", "I;16L"}:
            return self.eight_bit(image)
        if pixels.ndim != 2 or pixels.max() > 65535:
            raise ValueError("xrv 16 位输入必须是单通道灰度图")
        tensor = torch.from_numpy(pixels.astype(np.float32)).unsqueeze(0) / self.maxval
        height, width = tensor.shape[-2:]
        if self.letterbox:
            scale = self.size / max(width, height)
            resized_width, resized_height = max(1, round(width * scale)), max(1, round(height * scale))
            tensor = T.functional.resize(tensor, [resized_height, resized_width], antialias=True)
            left, top = (self.size - resized_width) // 2, (self.size - resized_height) // 2
            tensor = T.functional.pad(tensor, [left, top, self.size - resized_width - left, self.size - resized_height - top])
        else:
            tensor = T.functional.resize(tensor, [self.size, self.size], antialias=True)
        if self.train:
            tensor = self.augment(tensor)
        return (tensor * 2 - 1) * 1024


class _NormalizedTensor:
    """PIL 图 -> [C, size, size] float32 张量，并按 mean/std 归一化。

    等价于 `T.ToTensor()` + `T.Normalize(mean, std)`（同一套 float32 运算，
    实测逐元素相同），但少一次模块调用与中间张量；单图推理计时里这部分
    开销不可忽略。`mean=(0,0,0) / std=(1,1,1)` 时就是「只把像素缩到 0~1」。
    通道数由 mean/std 的长度决定：3 通道走 RGB，1 通道走灰度。
    """

    def __init__(self, mean: Sequence[float], std: Sequence[float]) -> None:
        self._channels = len(mean)
        self._mean = torch.tensor(mean, dtype=torch.float32).view(self._channels, 1, 1)
        self._std = torch.tensor(std, dtype=torch.float32).view(self._channels, 1, 1)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(pixels)
        if tensor.ndim == 2:  # 单通道（PIL L 模式）没有通道维，补出来再对齐
            tensor = tensor.unsqueeze(0)
        else:
            tensor = tensor.permute(2, 0, 1)
        return (tensor - self._mean) / self._std


class _XrvNormalizedTensor:
    """PIL 图 -> [1, size, size] float32 张量，按 TorchXRayVision 口径归一化。

    TxRV 的 `utils.normalize` 把 8 位像素映射到 [-1024, 1024] 的线性区间：
    `(2 * pixel / maxval - 1) * 1024`。B 组必须用同一口径，否则预训练权重看到
    的输入分布完全不同（医学预训练权重全程在 [-1024,1024] 上训练）。

    显式检查像素上限：若数据实际是 16 位而 `xrv_maxval` 仍写 255，映射后的值域
    会溢出 [-1024,1024]，训练不报错但会静默劣化；这里直接按要求报错。
    """

    def __init__(self, maxval: int = 255) -> None:
        if maxval <= 0:
            raise ValueError(f"xrv_maxval 必须是正整数，当前为 {maxval}")
        self.maxval = maxval

    def __call__(self, image: Image.Image) -> torch.Tensor:
        pixels = np.asarray(image)
        if pixels.dtype not in (np.uint8, np.uint16):
            raise ValueError(
                f"xrv 归一化只支持 8/16 位灰度像素，当前 dtype={pixels.dtype}、"
                f"mode={image.mode}；请检查读图链路"
            )
        if pixels.ndim == 3:  # 万一上游给了 RGB，取三通道均值（三通道相同时等价）
            pixels = pixels.mean(axis=2)
        pixels = pixels.astype(np.float32)
        if pixels.max() > self.maxval:
            raise ValueError(
                f"图像像素最大值 {pixels.max():.0f} 超过 xrv_maxval={self.maxval}，"
                "请确认位深与 image.xrv_maxval 配置是否一致"
            )
        normalized = (2.0 * pixels / self.maxval - 1.0) * 1024.0
        return torch.from_numpy(normalized).unsqueeze(0)


def _normalize_steps(
    normalize: str, in_channels: int = 3, xrv_maxval: int = 255
) -> list[Callable]:
    """把 PIL 图转成张量，并按配置决定要不要减均值除方差。

    `_NormalizedTensor` 会把 0~255 的整数缩到 0~1，所以 "none" 模式下就是
    「只缩放像素到 0~1」，与配置里写的含义一致。`xrv` 模式改用 TxRV 的
    [-1024,1024] 映射，且只支持单通道。
    """
    if normalize == "imagenet":
        return [_NormalizedTensor(IMAGENET_MEAN, IMAGENET_STD)]
    if normalize == "xrv":
        return [_XrvNormalizedTensor(xrv_maxval)]
    return [
        _NormalizedTensor(
            (0.0,) * in_channels, (1.0,) * in_channels
        )
    ]
