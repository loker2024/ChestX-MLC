"""模型：ResNet-18 骨干 + 10 类分类头，以及 study 级的 logits 聚合。

赛题按「检查（study）」评分，一次检查含 1~7 张图，这些图共享同一组标签。
所以模型的输出不是「一张图 10 个 logits」，而是「一次检查 10 个 logits」：
先让每张图各过一次骨干得到 `[n, 10]`，再把同一 study 内这几张图的 logits
取均值，得到 `[1, 10]`。训练和推理必须共用这套口径，否则训练时学的是逐图
判断、推理时却用均值，两边对不上，指标会莫名其妙地差。

为什么是均值而不是取最大：同一检查的正位与侧位只是同一处病变的不同视角，
均值相当于把各视角的证据平均起来；取最大则让单张图的噪声或伪影决定结果。

网络本体交给 timm：`timm.create_model('resnet18', num_classes=10)` 会把骨干
自带的 1000 类分类头换成 `Linear(512, 10)`，正好是赛题要的结构。选 ResNet-18
是因为部署约束（模型 ≤500MB、单图 ≤100ms）反推下来的参数量与计算量都合适。
"""

import warnings

import timm
import torch
from torch import nn

from common.config import Config


def build_model(cfg: Config) -> nn.Module:
    """按配置构建网络：ResNet-18 骨干 + 单层 `Linear(512, 10)` 分类头。

    `cfg.model.pretrained=True` 时会去下载 ImageNet 预训练权重。赛题服务器可能
    没有外网，这时 `create_model` 会抛错；捕获后改写随机初始化继续训练，
    并留下一条警告——事后看到指标偏低时，这条警告能提醒你「没加载预训练权重」。

    随机初始化的重试会把真正的配置错误（例如 backbone 名字拼错）原样抛出来，
    所以这里不是在掩盖错误，只是给「离线拿不到权重」留一条退路。
    """
    try:
        model = timm.create_model(
            cfg.model.backbone,
            pretrained=cfg.model.pretrained,
            num_classes=cfg.model.num_classes,
        )
        model.pretrained_loaded = bool(cfg.model.pretrained)
    except (OSError, RuntimeError) as error:
        warnings.warn(
            f"加载 {cfg.model.backbone} 的预训练权重失败（{error}），改为随机初始化；"
            "若这是配置错误，请检查 model.backbone 与网络连接。",
            RuntimeWarning,
        )
        model = timm.create_model(
            cfg.model.backbone,
            pretrained=False,
            num_classes=cfg.model.num_classes,
        )
        model.pretrained_loaded = False
    return model


def aggregate_logits(logits: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """把逐图 logits 聚合为 study 级 logits（同一 study 内取均值）。

    参数：

    - `logits`：`[N, C]`，本批所有真实图片的 logits，N = 各 study 图片数之和，
      C = 10 个类别；
    - `offsets`：`[B + 1]`，第 i 个 study 的图片是 `logits[offsets[i]:offsets[i+1]]`，
      与 `data.dataset.collate_studies` 的输出完全对应；
    - 返回：`[B, C]`，每个 study 一行。

    实现上没有 for 循环逐个 study 求 mean，而是先把 B 个 study 的图片展开成
    「每张图属于第几个 study」的下标 `study_index`，用一次 `index_add` 求和，
    再除以每个 study 的图片数。好处是只有一次算子调用，GPU 上不用等 B 次同步；
    而且 `index_add` 对 logits 可导，训练时梯度能正常传回骨干。
    """
    counts = offsets[1:] - offsets[:-1]  # [B]，每个 study 有几张图
    study_index = torch.repeat_interleave(
        torch.arange(len(counts), device=logits.device), counts
    )

    # 先按 study 把 logits 加起来；out-of-place 的 index_add 不会改动输入张量，
    # 反向传播时能正确地把梯度分散回每张图。
    sums = torch.zeros(
        len(counts), logits.shape[1], dtype=logits.dtype, device=logits.device
    ).index_add(0, study_index, logits)

    # counts 是 long，除法会自动提升到浮点；unsqueeze 把 [B] 变成 [B, 1] 以便广播。
    return sums / counts.unsqueeze(1)
