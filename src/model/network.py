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


def build_model(cfg: Config) -> nn.Module:
    """按配置构建 timm 图像分类模型，输出每张图的 10 类 logits。

    `cfg.model.pretrained=True` 时会尝试加载所选模型的预训练权重（必要时下载）。
    赛题服务器可能没有外网，加载失败时部分异常会被捕获并回退到随机初始化，
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
