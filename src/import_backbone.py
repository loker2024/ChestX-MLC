"""把外部框架的 ImageNet 骨干权重映射进 timm 骨干，产出可训练的仓库 checkpoint。

背景：T 系列拿到的外部权重有两种。T-001 那种是「别人已经在胸片上训练过、
带 10 类头」的完整模型，用 `import_checkpoint.py` 严格装入即可推理复评。
T-003 这种是纯 ImageNet 权重（torchvision `swin_v2_t-b137f0e2.pth`），
只有 1000 类头、没有任何胸片训练痕迹，不能直接推理算 macro-AUC，
必须先接上 10 类头重新训练——所以这里产出的是「初始化用的 checkpoint」，
下游用 `run_train.py --init-from` 消费。

本脚本负责两件事：

1. 把 torchvision 的 `features.*` 命名映射到 timm 的 `patch_embed.*` / `layers.*`
   命名。两边的 SwinV2 结构等价，但有三处必须显式处理：block 内 MLP 是 `mlp.0/3`
   对 `mlp.fc1/fc2`；qkv 在 torchvision 里是合并的 `qkv.bias`，timm 拆成
   `q_bias` / `v_bias` 且没有 k 偏置（本权重 k 段全 0，映射无损）；PatchMerging
   在 timm 里挂在**下一个** stage 的 `downsample` 上，而 `layers.0.downsample`
   是 Identity。
2. 映射完逐键校验：未映射的来源键只能有 `head.*`，timm 侧缺失键也只能有
   `head.*`，否则报错——宁可停下，也不要用半截权重去训练。

映射的正确性已用数值等价验证过：同一输入下 timm `forward_features` 与
torchvision `norm(features[:8](x))` 的逐元素最大差为 0.0。注意 torchvision
原生分类头是「先全局池化再 LayerNorm」，timm 是「先 LayerNorm 再池化」，
两者输出不等价（相对差约 0.85），但那只是池化顺序，与权重映射无关。

用法（在仓库根目录执行）：

    python src/import_backbone.py \
        --source ~/Desktop/swin_v2_t-b137f0e2.pth \
        --config configs/exp_T003.yaml \
        --out outputs/T-003_init/swinv2t_in1k_loker.pt
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch import nn

from common.checkpoint import save_checkpoint
from common.config import Config, load_config
from model.network import build_model

# 块内逐键映射：来源后缀 → timm 后缀。qkv 与 downsample 单独处理。
_BLOCK_KEY_MAP: tuple[tuple[str, str], ...] = (
    ("norm1.weight", "norm1.weight"),
    ("norm1.bias", "norm1.bias"),
    ("norm2.weight", "norm2.weight"),
    ("norm2.bias", "norm2.bias"),
    ("mlp.0.weight", "mlp.fc1.weight"),
    ("mlp.0.bias", "mlp.fc1.bias"),
    ("mlp.3.weight", "mlp.fc2.weight"),
    ("mlp.3.bias", "mlp.fc2.bias"),
    ("attn.proj.weight", "attn.proj.weight"),
    ("attn.proj.bias", "attn.proj.bias"),
    ("attn.logit_scale", "attn.logit_scale"),
    ("attn.cpb_mlp.0.weight", "attn.cpb_mlp.0.weight"),
    ("attn.cpb_mlp.0.bias", "attn.cpb_mlp.0.bias"),
    ("attn.cpb_mlp.2.weight", "attn.cpb_mlp.2.weight"),
)

# torchvision features 的下标：1/3/5/7 是四个 stage 的 block 序列。
_STAGE_INDEX: dict[int, int] = {1: 0, 3: 1, 5: 2, 7: 3}
# 2/4/6 是 PatchMerging，在 timm 里属于「下一层」的 downsample。
_MERGE_INDEX: dict[int, int] = {2: 1, 4: 2, 6: 3}

# 这两类是 torchvision 预计算好并持久化的 buffer，timm 在 forward 里用同一套公式
# 现算（cpb_mlp 权重已映射过去）。数值等价已验证：两边特征逐个元素差 0.0，
# 所以丢弃它们不损失任何信息。
_DROPPABLE_SUFFIXES: tuple[str, ...] = (
    ".attn.relative_coords_table",
    ".attn.relative_position_index",
)


def file_sha256(path: str | Path) -> str:
    """分块读取文件的 sha256，避免把上百 MB 权重整体读进内存。"""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    """读裸 state_dict；已经是本仓库格式时直接报错，避免重复包装。"""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 2.6 没有 weights_only 参数
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"{path} 不是字典结构，实际为 {type(payload).__name__}")
    if "model" in payload and "config" in payload:
        raise ValueError(f"{path} 已经是仓库 checkpoint 格式，无需转换")
    candidate = payload.get("state_dict", payload)
    if not all(isinstance(value, torch.Tensor) for value in candidate.values()):
        raise TypeError(f"{path} 里没有纯张量字典，顶层键为 {sorted(payload)[:8]}")
    return dict(candidate)


def looks_like_torchvision_swinv2(state_dict: dict[str, torch.Tensor]) -> bool:
    """靠三个特征键判断是不是 torchvision 的 SwinV2 权重，避免拿错文件。"""
    return (
        "features.0.0.weight" in state_dict
        and "features.0.2.weight" in state_dict
        and "head.weight" in state_dict
    )


def map_torchvision_swinv2(state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], list[str]]:
    """把 torchvision SwinV2 权重映射成 timm 命名，返回 (映射结果, 未映射的来源键)。

    未映射的键由调用方断言只允许是 `head.*`；k 偏置段必须全零，否则说明这份
    权重和 timm 的「无 k 偏置」实现不同源，映射会丢信息，这里直接报错。
    """
    mapped: dict[str, torch.Tensor] = {}
    consumed: set[str] = set()

    def take(source_key: str, target_key: str) -> None:
        mapped[target_key] = state_dict[source_key]
        consumed.add(source_key)

    take("features.0.0.weight", "patch_embed.proj.weight")
    take("features.0.0.bias", "patch_embed.proj.bias")
    take("features.0.2.weight", "patch_embed.norm.weight")
    take("features.0.2.bias", "patch_embed.norm.bias")

    for feature_index, layer_index in _STAGE_INDEX.items():
        block_index = 0
        while f"features.{feature_index}.{block_index}.norm1.weight" in state_dict:
            source = f"features.{feature_index}.{block_index}."
            target = f"layers.{layer_index}.blocks.{block_index}."
            for source_suffix, target_suffix in _BLOCK_KEY_MAP:
                if source + source_suffix in state_dict:
                    take(source + source_suffix, target + target_suffix)
            qkv_weight = state_dict[source + "attn.qkv.weight"]
            qkv_bias = state_dict[source + "attn.qkv.bias"]
            dim = qkv_weight.shape[0] // 3
            key_bias = qkv_bias[dim : 2 * dim]
            if key_bias.abs().max().item() != 0.0:
                raise ValueError(
                    f"{source}attn.qkv.bias 的 k 段不是全零（max abs {key_bias.abs().max().item():.3e}），"
                    "timm 实现没有 k 偏置，映射会丢信息"
                )
            take(source + "attn.qkv.weight", target + "attn.qkv.weight")
            mapped[target + "attn.q_bias"] = qkv_bias[:dim]
            mapped[target + "attn.v_bias"] = qkv_bias[2 * dim :]
            consumed.add(source + "attn.qkv.bias")
            block_index += 1
        if block_index == 0:
            raise ValueError(f"features.{feature_index} 下没有找到任何 block，权重结构与预期不符")

    for feature_index, layer_index in _MERGE_INDEX.items():
        prefix = f"features.{feature_index}."
        take(prefix + "norm.weight", f"layers.{layer_index}.downsample.norm.weight")
        take(prefix + "norm.bias", f"layers.{layer_index}.downsample.norm.bias")
        take(prefix + "reduction.weight", f"layers.{layer_index}.downsample.reduction.weight")

    take("norm.weight", "norm.weight")
    take("norm.bias", "norm.bias")

    unconsumed = sorted(set(state_dict) - consumed)
    return mapped, unconsumed


def build_target_model(config: Config) -> nn.Module:
    """按配置建 timm 模型；权重马上被覆盖，不再下载 ImageNet 预训练。"""
    return build_model(replace(config, model=replace(config.model, pretrained=False)))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="把外部 ImageNet 骨干映射成仓库初始化 checkpoint")
    parser.add_argument("--source", required=True, help="外部骨干权重（裸 state_dict）")
    parser.add_argument("--config", required=True, help="与目标模型一致的 YAML 配置")
    parser.add_argument("--out", required=True, help="输出的 checkpoint 路径（.pt）")
    parser.add_argument("--expect-sha256", default=None, help="来源文件的 sha256；给出则必须一致")
    parser.add_argument("--seed", type=int, default=None, help="写入 checkpoint 的 seed，仅记录用")
    parser.add_argument("--force", action="store_true", help="允许覆盖已存在的输出文件")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    source = Path(args.source)
    output = Path(args.out)
    if output.exists() and not args.force:
        raise FileExistsError(f"{output} 已存在；确认要覆盖请加 --force")

    config = load_config(args.config)
    actual_sha = file_sha256(source)
    if args.expect_sha256 and actual_sha != args.expect_sha256:
        raise ValueError(f"来源 sha256 不符：期望 {args.expect_sha256}，实际 {actual_sha}")

    raw = load_state_dict(source)
    if not looks_like_torchvision_swinv2(raw):
        raise ValueError(
            f"{source} 不像 torchvision SwinV2 权重（缺少 features.0.0/features.0.2/head）；"
            f"当前顶层键示例 {sorted(raw)[:6]}"
        )

    mapped, unconsumed = map_torchvision_swinv2(raw)
    dropped = [key for key in unconsumed if key.endswith(_DROPPABLE_SUFFIXES)]
    stray = [
        key
        for key in unconsumed
        if not key.startswith("head.") and not key.endswith(_DROPPABLE_SUFFIXES)
    ]
    if stray:
        raise ValueError(f"以下来源键没有映射目标，映射不完整：{stray[:8]}")

    model = build_target_model(config)
    result = model.load_state_dict(mapped, strict=False)
    unexpected = list(result.unexpected_keys)
    missing_non_head = [key for key in result.missing_keys if not key.startswith("head")]
    if unexpected:
        raise ValueError(f"timm 不接受这些映射后的键：{unexpected[:8]}")
    if missing_non_head:
        raise ValueError(f"骨干有 {len(missing_non_head)} 个张量没被权重覆盖：{missing_non_head[:8]}")

    save_checkpoint(
        output,
        model,
        config=config,
        seed=args.seed,
        metadata={
            "source_path": str(source.resolve()),
            "source_sha256": actual_sha,
            "source_format": "torchvision_swinv2",
            "mapped_tensors": len(mapped),
            "dropped_buffers": dropped,
            "random_init_tensors": sorted(result.missing_keys),
            "note": "ImageNet 骨干权重经 import_backbone.py 映射到 timm 命名，张量未做任何修改；分类头为随机初始化",
        },
    )

    parameter_count = sum(tensor.numel() for tensor in model.state_dict().values())
    print(f"来源: {source}")
    print(f"sha256: {actual_sha}")
    print(f"配置: {args.config} | 骨干 {config.model.backbone} | 尺寸 {config.image.size} | strict_img_size {config.model.strict_img_size}")
    print(f"映射张量: {len(mapped)} 个（丢弃 head.weight {tuple(raw['head.weight'].shape)} / head.bias）")
    print(f"随机初始化: {sorted(result.missing_keys)}")
    print(f"模型: {parameter_count / 1e6:.2f}M 参数")
    print(f"输出: {output}")


if __name__ == "__main__":
    main()
