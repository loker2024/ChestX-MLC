"""按配置构建图像分类模型，并提供 study 级 logits 聚合。

赛题按「检查（study）」评分，一次检查含 1~7 张图，这些图共享同一组标签。
模型先对每张图输出 10 个 logits，训练和推理再将它们聚合为检查级输出：
每张图各过一次模型得到 `[n, 10]`，再把同一 study 内这几张图的 logits
取均值，得到 `[1, 10]`。训练和推理必须共用这套口径，否则训练时学的是逐图
判断、推理时却用均值，两边对不上，指标会莫名其妙地差。

为什么是均值而不是取最大：同一检查的正位与侧位只是同一处病变的不同视角，
均值相当于把各视角的证据平均起来；取最大则让单张图的噪声或伪影决定结果。

网络本体交给 timm，骨干由 `cfg.model.backbone` 指定，分类输出数由
`cfg.model.num_classes` 指定。默认使用 ResNet-18，也可通过配置选择其他兼容
的 timm 图像分类模型；模型大小和端到端耗时须在赛题指定设备上实测。
"""

import warnings

import timm
import torch
from torch import nn

from common.config import Config

# 概率空间聚合时夹住两端，避免 logit(0) / logit(1) 变成 ±inf。
# 1e-6 对应 logit 约 ±13.8，BCE 在这种概率下已经饱和，误差远小于指标波动。
PROBABILITY_EPS = 1e-6


def _freeze_backbone(model: nn.Module, backbone: str) -> None:
    """冻结模型参数，只解冻 timm 暴露的分类头。"""
    classifier_getter = getattr(model, "get_classifier", None)
    if not callable(classifier_getter):
        raise ValueError(f"骨干 {backbone!r} 没有可识别的分类头（缺少 get_classifier）")
    classifier = classifier_getter()
    if not isinstance(classifier, nn.Module):
        raise ValueError(f"骨干 {backbone!r} 的分类头不是 nn.Module")
    classifier_parameters = list(classifier.parameters())
    if not classifier_parameters:
        raise ValueError(f"骨干 {backbone!r} 的分类头没有可训练参数")

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in classifier_parameters:
        parameter.requires_grad_(True)


def set_frozen_backbone_eval(model: nn.Module) -> None:
    """冻结模式下固定骨干的 BatchNorm 等运行状态，只让分类头处于 train 模式。"""
    classifier_getter = getattr(model, "get_classifier", None)
    if not callable(classifier_getter):
        raise ValueError("冻结模型缺少可识别的分类头（get_classifier）")
    classifier = classifier_getter()
    if not isinstance(classifier, nn.Module):
        raise ValueError("冻结模型的分类头不是 nn.Module")
    model.eval()
    classifier.train()


def build_model(cfg: Config) -> nn.Module:
    """按配置构建图像分类模型，输出每张图的 10 类 logits。

    默认（`cfg.model.arch="timm"`）构建 timm 骨干；`arch="xrv_densenet121"` 时改走
    `model.xrv` 的 TorchXRayVision DenseNet-121（B 组隔离研究，惰性导入，默认链路
    完全不依赖 torchxrayvision）。两条路径都输出原始 logits。

    `cfg.model.pretrained=True` 时会尝试加载所选模型的预训练权重（必要时下载）。
    赛题服务器可能没有外网，加载失败时部分异常会被捕获并回退到随机初始化，
    并留下一条警告——事后看到指标偏低时，这条警告能提醒你「没加载预训练权重」。
    B 组例外：TxRV 权重加载失败在 `model.xrv` 里直接报错，不回退。

    随机初始化的重试会把真正的配置错误（例如 backbone 名字拼错）原样抛出来，
    所以这里不是在掩盖错误，只是给「离线拿不到权重」留一条退路。
    """
    if cfg.model.arch == "xrv_densenet121":
        # 惰性导入：只有 B 组实验才需要 torchxrayvision，默认链路不受影响。
        from model.xrv import build_xrv_densenet121

        return build_xrv_densenet121(cfg)

    try:
        model = timm.create_model(
            cfg.model.backbone,
            pretrained=cfg.model.pretrained,
            num_classes=cfg.model.num_classes,
            drop_rate=cfg.model.dropout,
        )
        model.pretrained_loaded = bool(cfg.model.pretrained)
    except (OSError, RuntimeError) as error:
        if cfg.model.head in {"csra", "ml_decoder"}:
            raise RuntimeError(f"{cfg.model.head} 实验的预训练骨干加载失败，已停止构建") from error
        warnings.warn(
            f"加载 {cfg.model.backbone} 的预训练权重失败（{error}），改为随机初始化；"
            "若这是配置错误，请检查 model.backbone 与网络连接。",
            RuntimeWarning,
        )
        model = timm.create_model(
            cfg.model.backbone,
            pretrained=False,
            num_classes=cfg.model.num_classes,
            drop_rate=cfg.model.dropout,
        )
        model.pretrained_loaded = False
    if cfg.model.head == "csra":
        from model.csra import ConvNeXtCSRAHead

        model.head = ConvNeXtCSRAHead(
            model.head.norm, model.num_features, cfg.model.num_classes,
            cfg.model.csra_residual_weight, cfg.model.csra_temperature,
        )
    elif cfg.model.head == "ml_decoder":
        from model.ml_decoder import ConvNeXtMLDecoderHead

        model.head = ConvNeXtMLDecoderHead(model.head.norm, model.num_features, cfg.model.num_classes)
    if cfg.model.freeze_backbone:
        _freeze_backbone(model, cfg.model.backbone)
    return model


def _study_mean(
    values: torch.Tensor,
    offsets: torch.Tensor,
    views: torch.Tensor | None = None,
    frontal_weight: float = 0.7,
) -> torch.Tensor:
    """把逐图 `[N, C]` 值按 study 聚合为 `[B, C]`。

    没有 for 循环逐个 study 求 mean，而是先把 B 个 study 的图片展开成
    「每张图属于第几个 study」的下标 `study_index`，用一次 `index_add` 求和，
    再除以每个 study 的图片数。好处是只有一次算子调用，GPU 上不用等 B 次同步；
    而且 `index_add` 对输入可导，训练时梯度能正常传回骨干。

    `views` 为 0 正位、1 侧位、2 未知：含未知视角或只有一种视角的 study 回退
    全图均值；两种已知视角先分别求均值，再按 `frontal_weight` 加权。
    """
    counts = offsets[1:] - offsets[:-1]  # [B]，每个 study 有几张图
    study_index = torch.repeat_interleave(
        torch.arange(len(counts), device=values.device), counts
    )

    # 先按 study 把值加起来；out-of-place 的 index_add 不会改动输入张量，
    # 反向传播时能正确地把梯度分散回每张图。
    sums = torch.zeros(
        len(counts), values.shape[1], dtype=values.dtype, device=values.device
    ).index_add(0, study_index, values)

    mean = sums / counts.unsqueeze(1)
    if views is None:
        return mean
    if views.ndim != 1 or views.numel() != values.shape[0]:
        raise ValueError("views 必须与图片数量一致")
    if not 0 < frontal_weight < 1:
        raise ValueError("frontal_weight 必须在 (0, 1) 内")
    # 未知视角的 study 使用全图均值；单一已知视角同样使用该组均值。
    group_means = []
    group_counts = []
    for view_id in (0, 1):
        mask = (views == view_id).to(values.dtype)
        counts_view = torch.zeros(len(counts), dtype=values.dtype, device=values.device).index_add(0, study_index, mask)
        sums_view = torch.zeros_like(sums).index_add(0, study_index, values * mask.unsqueeze(1))
        group_means.append(sums_view / counts_view.clamp_min(1).unsqueeze(1))
        group_counts.append(counts_view)
    both = (group_counts[0] > 0) & (group_counts[1] > 0) & ((group_counts[0] + group_counts[1]) == counts)
    weighted = frontal_weight * group_means[0] + (1 - frontal_weight) * group_means[1]
    return torch.where(both.unsqueeze(1), weighted, mean)


def aggregate_logits(
    logits: torch.Tensor, offsets: torch.Tensor, views: torch.Tensor | None = None,
    frontal_weight: float = 0.7, probability_space: bool = False,
) -> torch.Tensor:
    """把逐图 logits 聚合为 study 级 logits；提供 views 时按视角加权。

    views 使用 0=正位、1=侧位、2=未知。含未知视角或只有一种视角的
    study 回退全图均值；两种已知视角先分别求均值再加权。

    参数：

    - `logits`：`[N, C]`，本批所有真实图片的 logits，N = 各 study 图片数之和，
      C = 10 个类别；
    - `offsets`：`[B + 1]`，第 i 个 study 的图片是 `logits[offsets[i]:offsets[i+1]]`，
      与 `data.dataset.collate_studies` 的输出完全对应；
    - `probability_space`：聚合在哪个空间进行。`False`（默认）在 logits 空间
      平均（G0）；`True` 先逐图 sigmoid、在概率空间聚合（G1），最后用 logit
      变换回 logits 空间，即返回 `logit(A(sigmoid(z)))`；
    - 返回：`[B, C]`，每个 study 一行。

    概率空间下仍返回 logits，是为了让下游不必分支：`BCEWithLogitsLoss` 对
    `logit(p)` 的取值与直接用概率算 BCE 等价，推理侧 `sigmoid(logit(p)) == p`，
    拿到的正是概率空间均值。两端用 `PROBABILITY_EPS` 夹住，避免 `logit(0)` /
    `logit(1)` 变成无穷；概率先升到 fp32 再算，逐步 sigmoid 在 fp16 下会下溢。
    """
    if not probability_space:
        return _study_mean(logits, offsets, views, frontal_weight)

    probs = torch.sigmoid(logits.float())
    merged = _study_mean(probs, offsets, views, frontal_weight)
    clamped = merged.clamp(PROBABILITY_EPS, 1.0 - PROBABILITY_EPS)
    return torch.logit(clamped).to(logits.dtype)
