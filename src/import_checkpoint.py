"""把外部产出的权重文件转成本仓库的 checkpoint 格式，供 run_eval / run_infer 直接使用。

背景：T 系列（外部权重复评）拿到的 `.pth` 由别的代码库训练，外层是
`{"architecture", "model_backend", "num_classes", "state_dict", "metadata"}`，
而本仓库的 `common.checkpoint.load_checkpoint` 约定 `checkpoint["model"]` 放权重、
`checkpoint["config"]` 放存档配置。键名不同，直接把外部文件喂给 `--checkpoint`
会在 `checkpoint["model"]` 处抛 KeyError。

本脚本不复制、不改写任何张量，只做三件事：
1. 按配置构建模型，用 `strict=True` 装入外部权重——顺带验证「骨干 + 类别数」确实匹配，
   键名对不上会立刻报错，而不是等到算指标时才发现。
2. 把权重、配置与来源信息（source 路径、sha256、外部 metadata）一起存成本仓库格式。
3. 存盘前打印摘要，存盘后回读一次确认能正常 `load_checkpoint`。

用法（在仓库根目录执行）：

    python src/import_checkpoint.py \
        --source models/R1_best_model.pth \
        --config configs/exp_T001a.yaml \
        --out outputs/ensemble/ENS02_R1_R2_384_models_v1/R1_best_loker.pt

加上 `--expect-sha256` 可以锁死来源文件，避免拿错版本；权重自带的 metadata 里
若声明了 architecture / num_classes / input_size，也会和配置逐项比对。
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch import nn

from common.checkpoint import load_checkpoint, save_checkpoint
from common.config import Config, load_config
from model.network import build_model


def file_sha256(path: str | Path) -> str:
    """分块读取文件的 sha256，避免把几十 MB 权重整体读进内存。"""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_raw(path: str | Path) -> dict[str, Any]:
    """读权重文件，兼容 torch 2.6+ 的 weights_only 默认值变化。"""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 2.6 没有 weights_only 参数
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"{path} 不是字典结构的 checkpoint，实际为 {type(payload).__name__}")
    return payload


def extract_state_dict(payload: dict[str, Any], source: str | Path) -> dict[str, torch.Tensor]:
    """从外部格式里取出 state_dict；已经是本仓库格式时直接报错，避免重复包装。

    支持两种外部写法：`{"state_dict": ...}`（TorchVision 派系的常见约定）与
    裸的 `{参数名: 张量}`。后者常见于手工 `torch.save(model.state_dict())`。
    """
    if "model" in payload and "config" in payload:
        raise ValueError(f"{source} 已经是本仓库 checkpoint 格式，无需转换")
    candidate = payload.get("state_dict", payload)
    if not all(isinstance(value, torch.Tensor) for value in candidate.values()):
        raise TypeError(
            f"{source} 里没有找到纯张量字典：期望 'state_dict' 键或裸 state_dict，"
            f"实际顶层键为 {sorted(payload)[:8]}"
        )
    return dict(candidate)


def check_metadata(payload: dict[str, Any], config: Config) -> dict[str, Any]:
    """把外部 metadata 声明的口径与配置逐项比对，返回原 metadata 供存档。

    只校验三项决定性字段：骨干、类别数、输入尺寸。骨干允许配置写成带后缀的 timm 名
    （例如 `densenet121.ra_in1k`），比对时取点号前的部分。
    """
    metadata = payload.get("metadata") or {}
    declared_arch = payload.get("architecture") or metadata.get("backbone")
    if declared_arch:
        configured = config.model.backbone.split(".")[0]
        if declared_arch != configured:
            raise ValueError(f"骨干不匹配：权重声明 {declared_arch!r}，配置为 {config.model.backbone!r}")
    declared_classes = payload.get("num_classes", metadata.get("num_classes"))
    if declared_classes is not None and int(declared_classes) != config.model.num_classes:
        raise ValueError(f"类别数不匹配：权重声明 {declared_classes}，配置为 {config.model.num_classes}")
    declared_size = metadata.get("input_size", metadata.get("size"))
    if declared_size is not None and int(declared_size) != config.image.size:
        raise ValueError(f"输入尺寸不匹配：权重声明 {declared_size}，配置为 {config.image.size}")
    return dict(metadata)


def build_and_load(config: Config, state_dict: dict[str, torch.Tensor]) -> nn.Module:
    """按配置建模型并严格装入权重。

    `pretrained=False` 是必须的：权重马上被覆盖，没必要再下载 ImageNet 预训练权重
    （离线环境也下载不了）。strict=True 才是这里的关键——它保证键名与张量形状
    完全一致，等价于「这份权重确实是这个配置的模型训出来的」。
    """
    model = build_model(replace(config, model=replace(config.model, pretrained=False)))
    model.load_state_dict(state_dict, strict=True)
    return model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="把外部权重转换成仓库 checkpoint 格式")
    parser.add_argument("--source", required=True, help="外部权重文件路径")
    parser.add_argument("--config", required=True, help="与该权重口径一致的 YAML 配置")
    parser.add_argument("--out", required=True, help="输出的 checkpoint 路径（.pt）")
    parser.add_argument("--expect-sha256", default=None, help="来源文件的 sha256；给出则必须一致")
    parser.add_argument("--seed", type=int, default=None, help="写入 checkpoint 的训练 seed，仅记录用")
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

    payload = load_raw(source)
    state_dict = extract_state_dict(payload, source)
    metadata = check_metadata(payload, config)
    model = build_and_load(config, state_dict)

    save_checkpoint(
        output,
        model,
        config=config,
        seed=args.seed,
        metadata={
            "source_path": str(source.resolve()),
            "source_sha256": actual_sha,
            "source_architecture": payload.get("architecture"),
            "source_model_backend": payload.get("model_backend"),
            "source_metadata": metadata,
            "note": "外部权重经 import_checkpoint.py 转换，张量未做任何修改",
        },
    )

    verified = load_checkpoint(output, map_location="cpu")
    parameter_count = sum(tensor.numel() for tensor in verified["model"].values())
    print(f"来源: {source}")
    print(f"sha256: {actual_sha}")
    print(f"配置: {args.config} | 骨干 {config.model.backbone} | 尺寸 {config.image.size}")
    print(f"张量: {len(verified['model'])} 个 / {parameter_count / 1e6:.2f}M 参数")
    print(f"输出: {output} | 已回读校验通过")


if __name__ == "__main__":
    main()
