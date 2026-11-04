"""推理：把一个 DataLoader 里的 study 跑成「一行一个 study」的概率宽表。

推理和验证共用同一套口径：每张图各过一次网络得到 [N, 10] logits，
再用 offsets 把同一 study 的多张图 logits 求均值，最后 sigmoid 成概率。
这样训练时看到的分数、验证时算的指标、提交时写的概率完全一致。

这里只产出宽表（Study_id + p1..p10），不直接写 submission.csv：
宽表适合算指标、调阈值、排查单个 study；真正的长表提交格式由
infer.submission 负责转换。

本模块同时提供 `predict_image` / `predict_study` 两个「不经 DataLoader」的
推理入口，供 `src/cost.py` 计时和一致性校验复用。三者（predict_loader、
predict_image、predict_study）都经 `_forward_probabilities` 做前向、聚合与
sigmoid，保证提交概率、计时口径、校验口径不分叉。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from common.checkpoint import load_checkpoint
from common.config import Config, config_from_dict
from common.constants import CLASS_IDS
from data.dataset import load_image
from data.transforms import build_eval_transform
from model.network import forward_studies, build_model


@dataclass(frozen=True)
class StudyModelSpec:
    """多模型 study 推理中的一个成员：模型 + 预处理 + 聚合参数。

    `transform_key` 是预处理的完整签名（尺寸、letterbox、归一化、通道数等）。
    `predict_study_multi` 会把 key 相同的成员合并到一次预处理：同一张图只在
    第一个成员上做一遍 transform，后续成员复用同一个 CPU 张量。这不改变数值，
    但能把「N 个模型各做一遍预处理」压成「每种预处理做一遍」。
    """

    model: nn.Module
    transform: Callable[[Image.Image], torch.Tensor]
    transform_key: object | None = None
    uses_views: bool = False
    frontal_weight: float = 0.7
    probability_space: bool = False


def load_checkpoint_model(
    config: Config, checkpoint_path: str | Path, device: torch.device
) -> nn.Module:
    """按配置建模型、装入 checkpoint 权重，返回已经放到 device 上的 eval 模型。

    建模型时强制 pretrained=False：权重马上会被 checkpoint 覆盖，没必要再下载
    一遍 ImageNet 预训练权重（离线环境也下载不了）。模型结构仍由 config 决定，
    所以推理用的配置文件必须和训练时一致（backbone、类别数）。
    """
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    if checkpoint.get("config") is not None:
        saved = config_from_dict(checkpoint["config"])
        fields = {"image": ("size", "letterbox", "normalize", "in_channels", "xrv_maxval"),
                  "model": ("arch", "backbone", "num_classes", "dropout", "head", "csra_residual_weight", "csra_temperature", "aggregation", "frontal_weight")}
        differences = [f"{section}.{field}" for section, names in fields.items()
                       for field in names if getattr(getattr(config, section), field) != getattr(getattr(saved, section), field)]
        if differences:
            raise ValueError("推理配置与 checkpoint 不一致：" + ", ".join(differences))
    else:
        import warnings
        warnings.warn("旧 checkpoint 缺少存档配置，无法验证推理口径", RuntimeWarning)
    # 只替换 pretrained 字段，backbone 和 num_classes 保持配置里的值。
    model_config = replace(config.model, pretrained=False)
    model = build_model(replace(config, model=model_config))

    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


def validate_submission_model(config: Config, checkpoint_path: str | Path) -> None:
    """禁止将研究专用的 TxRV 模型用于提交。"""
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    saved = config_from_dict(checkpoint["config"]) if checkpoint.get("config") else config
    if config.model.arch == "xrv_densenet121" or saved.model.arch == "xrv_densenet121":
        raise ValueError("TxRV 模型仅用于研究，不能生成赛题提交或参与提交融合")


def build_eval_inference_transform(
    config: Config,
) -> Callable[[Image.Image], torch.Tensor]:
    """返回与 `build_loader(config, split, train=False)` 完全相同的推理预处理。

    cost.py 不走 DataLoader（计时要求 batch=1 串行、且要控制每张图的边界），
    但预处理必须和训练/提交时一致，所以这里复用同一个 build_eval_transform。
    """
    return build_eval_transform(
        config.image.size,
        config.image.letterbox,
        config.image.normalize,
        config.image.in_channels,
        config.image.xrv_maxval,
    )


def _model_dtype(model: nn.Module) -> torch.dtype:
    """模型参数的 dtype；fp16 推理时把输入也转成半精度。"""
    try:
        return next(model.parameters()).dtype
    except StopIteration:  # 没有参数的可测替身（比如全用缓冲区的假模型）
        return torch.float32


def _forward_probabilities(
    model: nn.Module,
    images: torch.Tensor,
    offsets: torch.Tensor,
    device: torch.device,
    views: torch.Tensor | None = None,
    frontal_weight: float = 0.7,
    probability_space: bool = False,
) -> torch.Tensor:
    """CPU 上的 images/offsets -> CPU 上的 `[B, 10]` fp32 概率。

    这是全仓库唯一的「前向 + study 聚合 + sigmoid」实现：
    搬设备、按模型 dtype 转换输入、aggregate_logits 求 study 均值、sigmoid 后
    再转回 fp32 CPU 张量。predict_loader / predict_image / predict_study 都走它。

    `probability_space=True` 时聚合在概率空间进行（G1），此时 aggregate_logits
    返回的是 `logit(mean(sigmoid))`，外层 sigmoid 正好还原成概率空间均值。
    """
    return torch.sigmoid(
        _forward_logits(model, images, offsets, device, views, frontal_weight, probability_space)
    )


def _forward_logits(
    model: nn.Module,
    images: torch.Tensor,
    offsets: torch.Tensor,
    device: torch.device,
    views: torch.Tensor | None = None,
    frontal_weight: float = 0.7,
    probability_space: bool = False,
) -> torch.Tensor:
    """CPU 上的 images/offsets -> CPU 上的 `[B, 10]` fp32 study logits。"""
    model_images = images.to(device=device, dtype=_model_dtype(model), non_blocking=True)
    model_offsets = offsets.to(device=device, non_blocking=True)
    study_logits = forward_studies(model, model_images, model_offsets,
                                    views.to(device) if views is not None else None,
                                    frontal_weight, probability_space)
    # 先转 float32 再搬 CPU：以后即使模型用半精度推理，融合也使用稳定的 fp32 logits。
    return study_logits.float().cpu()


@torch.inference_mode()
def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: Config | None = None,
) -> pd.DataFrame:
    """对一个 DataLoader 做推理，返回 `Study_id, p1..p10` 的宽表。

    每个 batch 的图片按 data.dataset.collate_studies 的约定首尾相接，
    offsets 给出每个 study 的图片区间；同一 study 多张图的 logits 求均值后
    sigmoid，得到这次检查的 10 个概率。行顺序与 DataLoader 顺序一致
    （test loader 不打乱，也就是 CSV 里 study 第一次出现的顺序）。

    概率列名沿用赛题的类别 ID：p1 对应类别 1，p10 对应类别 10。
    """
    model.eval()
    study_ids: list[str] = []
    probability_batches: list[torch.Tensor] = []

    for batch in loader:
        probabilities = _forward_probabilities(
            model, batch["images"], batch["offsets"], device,
            batch["views"] if config is not None and config.model.uses_views else None,
            config.model.frontal_weight if config is not None else 0.7,
            config.model.probability_space if config is not None else False,
        )
        study_ids.extend(batch["study_ids"])
        probability_batches.append(probabilities)

    if not probability_batches:
        raise RuntimeError("推理集为空，无法生成预测")
    probabilities = torch.cat(probability_batches).numpy()
    frame = pd.DataFrame(probabilities, columns=[f"p{class_id}" for class_id in CLASS_IDS])
    frame.insert(0, "Study_id", study_ids)
    return frame


@torch.inference_mode()
def predict_loader_logits(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: Config | None = None,
) -> pd.DataFrame:
    """对 DataLoader 做 study-level 推理，返回 `Study_id, z1..z10` logits 宽表。"""
    model.eval()
    study_ids: list[str] = []
    logit_batches: list[torch.Tensor] = []

    for batch in loader:
        logits = _forward_logits(
            model,
            batch["images"],
            batch["offsets"],
            device,
            batch["views"] if config is not None and config.model.uses_views else None,
            config.model.frontal_weight if config is not None else 0.7,
            config.model.probability_space if config is not None else False,
        )
        study_ids.extend(batch["study_ids"])
        logit_batches.append(logits)

    if not logit_batches:
        raise RuntimeError("推理集为空，无法生成 logits")
    logits = torch.cat(logit_batches).numpy()
    frame = pd.DataFrame(logits, columns=[f"z{class_id}" for class_id in CLASS_IDS])
    frame.insert(0, "Study_id", study_ids)
    return frame


@torch.inference_mode()
def predict_image(
    model: nn.Module,
    image_path: str | Path,
    transform: Callable[[Image.Image], torch.Tensor],
    device: torch.device,
) -> np.ndarray:
    """单张图片的 10 类概率（batch=1，`[10]` fp32 numpy）。

    读图 + 预处理 + 前向 + sigmoid 全在这里，cost.py 用它做端到端计时：
    计时区间包住这次调用的全部工作，测的就是「读文件到算出结果」的墙钟时间。
    单图 forward 与 DataLoader 里多图 forward 在 eval 模式下逐图独立，结果一致。
    """
    model.eval()
    # convert_rgb=False：灰度图保留 L 模式，由 Letterbox 在 resize 之后转 RGB，
    # 结果与先转 RGB 再 resize 逐像素一致，但单通道插值只需 1/3 的运算量。
    tensor = transform(load_image(Path(image_path), convert_rgb=False)).unsqueeze(0)
    offsets = torch.tensor([0, 1], dtype=torch.long)
    return _forward_probabilities(model, tensor, offsets, device)[0].numpy()


@torch.inference_mode()
def predict_study(
    model: nn.Module,
    image_paths: Sequence[str | Path],
    transform: Callable[[Image.Image], torch.Tensor],
    device: torch.device,
    views: Sequence[int] | None = None,
    frontal_weight: float = 0.7,
    probability_space: bool = False,
) -> np.ndarray:
    """一次检查（study）的 10 类概率：该 study 所有图片 logits 求均值后 sigmoid。

    与 `predict_loader` 的聚合口径完全一致（同样的 aggregate_logits + sigmoid）。
    读图后转交 `predict_study_multi`：一次检查的全部图片拼成一个 batch、只做
    一次前向，对应赛题答疑「batch=1 以 study_id 为单位，多图可同时处理」的口径。

    `pictures_100` 里一个 study 只有部分图片时，这里得到的是「这批图」的均值，
    与提交时「全部图片」的均值会有差异——赛题答疑已明确每个目录含该检查全部
    图片，正常评测不会出现部分图片。
    """
    paths = [Path(path) for path in image_paths]
    if not paths:
        raise ValueError("一个 study 至少要有一张图片才能推理")
    spec = StudyModelSpec(
        model=model,
        transform=transform,
        transform_key=("single",),
        uses_views=views is not None,
        frontal_weight=frontal_weight,
        probability_space=probability_space,
    )
    return predict_study_multi([spec], paths, device, views, max_workers=1)[0]


@torch.inference_mode()
def predict_study_images(
    model: nn.Module,
    images: Sequence[Image.Image],
    transform: Callable[[Image.Image], torch.Tensor],
    device: torch.device,
    views: Sequence[int] | None = None,
    frontal_weight: float = 0.7,
    probability_space: bool = False,
) -> np.ndarray:
    """对已经解码的 PIL 图像做 study 级推理，返回 `[10]` fp32 概率。

    与 `predict_study` 数值等价，区别只在读图由调用方完成。`cost.py` 的多模型
    计时需要「同一张图只解码一次、多个模型各自预处理与前向」，因此把解码与
    前向拆开；这样多模型融合的计时不会把同一张图的解码重复计 N 次。
    """
    if not images:
        raise ValueError("一个 study 至少要有一张图片才能推理")
    model.eval()
    tensors = torch.stack([transform(image) for image in images])
    offsets = torch.tensor([0, len(images)], dtype=torch.long)
    return _forward_probabilities(
        model, tensors, offsets, device,
        torch.tensor(views) if views is not None else None,
        frontal_weight, probability_space,
    )[0].numpy()


def _map_parallel(
    function,
    items,
    max_workers: int | None,
    executor: ThreadPoolExecutor | None = None,
):
    """按需并行 map；1 张图或 `max_workers=1` 时保持串行，避免线程池开销。

    传入 `executor` 时复用调用方持有的线程池，把「每个 study 建两次线程池」
    压成「整次评测建一次」。线程数由 executor 决定，`max_workers` 只保留
    「是否并行」的语义（等于 1 时仍串行）。
    """
    items = list(items)
    if len(items) <= 1 or max_workers == 1:
        return [function(item) for item in items]
    if executor is not None:
        return list(executor.map(function, items))
    workers = min(len(items), max_workers or 4)
    if workers <= 1:
        return [function(item) for item in items]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(function, items))


@torch.inference_mode()
def predict_study_multi(
    specs: Sequence[StudyModelSpec],
    image_paths: Sequence[str | Path],
    device: torch.device,
    views: Sequence[int] | None = None,
    max_workers: int | None = 4,
    executor: ThreadPoolExecutor | None = None,
) -> list[np.ndarray]:
    """对同一 study 的多张图片做多模型推理，返回每个模型的 `[10]` 概率。

    - 图片只解码一次；`max_workers > 1` 且图数 > 1 时并行解码与缩放；
    - `transform_key` 相同的成员共享一次预处理：同一张图只做一遍 transform，
      后续成员复用同一个 CPU 张量（不改变数值，只省掉重复的 resize/normalize）；
    - 每个成员仍各自做一次前向与 study 聚合，`uses_views` 控制是否传视角。

    这是 `cost.py` 多模型计时的入口：调用方把整个函数包在计时区间里，因此解码、
    预处理、所有模型前向、聚合都被计入；并行只改变 CPU 侧调度，不改变预测。
    """
    paths = [Path(path) for path in image_paths]
    if not paths:
        raise ValueError("一个 study 至少要有一张图片才能推理")
    if not specs:
        raise ValueError("至少需要一个模型才能推理")
    if max_workers is not None and max_workers <= 0:
        raise ValueError(f"max_workers 必须为正整数或 None，当前为 {max_workers!r}")

    images = _map_parallel(
        lambda path: load_image(path, convert_rgb=False), paths, max_workers, executor
    )
    offsets = torch.tensor([0, len(paths)], dtype=torch.long)

    # 先按 transform_key 去重，再把「所有 key × 所有图片」放进同一个线程池。
    # 这样 320 与 384 两种缩放是并行做的，而不是先做完 320 再做 384。
    # 只有可哈希且显式提供的 key 才允许共享。缺少 key 或 key 本身不可哈希时，
    # 给该模型分配一个独立哨兵，避免把两个未知预处理口径静默合并。
    unique: dict[object, Callable[[Image.Image], torch.Tensor]] = {}
    spec_keys: list[object] = []
    for spec in specs:
        key = spec.transform_key
        try:
            hash(key)
        except TypeError:
            key = None
        if key is None:
            key = object()
        spec_keys.append(key)
        unique.setdefault(key, spec.transform)

    def transform_one(task):
        key, image = task
        return key, unique[key](image)

    tasks = [(key, image) for key in unique for image in images]
    transformed = _map_parallel(transform_one, tasks, max_workers, executor)
    buckets: dict[object, list[torch.Tensor]] = {key: [] for key in unique}
    for key, tensor in transformed:
        buckets[key].append(tensor)
    cache = {key: torch.stack(values) for key, values in buckets.items()}

    results: list[np.ndarray] = []
    for spec, key in zip(specs, spec_keys):
        tensor = cache[key]
        results.append(
            _forward_probabilities(
                spec.model, tensor, offsets, device,
                torch.tensor(views) if spec.uses_views and views is not None else None,
                spec.frontal_weight, spec.probability_space,
            )[0].numpy()
        )
    return results


@torch.inference_mode()
def warmup_study_shapes(
    model: nn.Module,
    image_counts: Sequence[int],
    size: int,
    channels: int,
    device: torch.device,
    frontal_weight: float = 0.7,
    probability_space: bool = False,
) -> None:
    """用合成零张量预热计时集中出现的 batch 形状，只做算子初始化，不计入成绩。

    赛题答疑明确允许「使用合成虚拟张量完成 CUDA Graph 捕获、编译、算子初始化」。
    同一 study 的图片数在 1~5 之间变化，某个形状第一次出现时会触发 kernel
    选择/编译；不预热的话，这一个 study 的耗时会把 100 个 study 的均值拉高
    几毫秒到几十毫秒。预热只走前向，不读图像、不写预测，也不参与计时。
    """
    counts = sorted({int(count) for count in image_counts if int(count) > 0})
    if not counts:
        return
    for count in counts:
        tensor = torch.zeros(count, channels, size, size)
        offsets = torch.tensor([0, count], dtype=torch.long)
        _forward_probabilities(
            model, tensor, offsets, device, None, frontal_weight, probability_space
        )
