"""骨干权重映射：torchvision SwinV2 命名 → timm 命名。

不建真实模型，只用手工构造的最小 state_dict 验证三件事：键怎么搬、qkv 偏置
怎么拆、结构不符时是否立刻报错。真实权重（233 个张量 / 27.6M 参数）的等价性
已在 V100 上验证过，不进单测。
"""

import pytest
import torch

from import_backbone import looks_like_torchvision_swinv2, map_torchvision_swinv2

DIMS = (4, 8, 16, 32)  # 四个 stage 的通道数，按 SwinV2-T 的 96/192/384/768 缩小
HEADS = (1, 2, 4, 8)


def _block(prefix, dim, heads, key_bias_value=0.0):
    """造一个 torchvision SwinV2 block 的键（MLP 用 0/3，qkv 合并带 bias）。"""
    keys = {
        f"{prefix}norm1.weight": torch.ones(dim),
        f"{prefix}norm1.bias": torch.zeros(dim),
        f"{prefix}norm2.weight": torch.ones(dim),
        f"{prefix}norm2.bias": torch.zeros(dim),
        f"{prefix}mlp.0.weight": torch.full((4 * dim, dim), 2.0),
        f"{prefix}mlp.0.bias": torch.zeros(4 * dim),
        f"{prefix}mlp.3.weight": torch.full((dim, 4 * dim), 3.0),
        f"{prefix}mlp.3.bias": torch.zeros(dim),
        f"{prefix}attn.qkv.weight": torch.full((3 * dim, dim), 4.0),
        f"{prefix}attn.qkv.bias": torch.cat([
            torch.full((dim,), 5.0),
            torch.full((dim,), key_bias_value),
            torch.full((dim,), 6.0),
        ]),
        f"{prefix}attn.proj.weight": torch.full((dim, dim), 7.0),
        f"{prefix}attn.proj.bias": torch.zeros(dim),
        f"{prefix}attn.logit_scale": torch.zeros(heads, 1, 1),
        f"{prefix}attn.cpb_mlp.0.weight": torch.zeros(512, 2),
        f"{prefix}attn.cpb_mlp.0.bias": torch.zeros(512),
        f"{prefix}attn.cpb_mlp.2.weight": torch.zeros(heads, 512),
        f"{prefix}attn.relative_coords_table": torch.zeros(1, 15, 15, 2),
        f"{prefix}attn.relative_position_index": torch.zeros(64, dtype=torch.long),
    }
    return keys


def _fake_swinv2(key_bias_value=0.0):
    """按 torchvision 的 features.* 结构拼一份最小权重。"""
    state = {
        "features.0.0.weight": torch.zeros(DIMS[0], 3, 4, 4),
        "features.0.0.bias": torch.zeros(DIMS[0]),
        "features.0.2.weight": torch.ones(DIMS[0]),
        "features.0.2.bias": torch.zeros(DIMS[0]),
    }
    for stage, (dim, heads) in enumerate(zip(DIMS, HEADS)):
        state.update(_block(f"features.{2 * stage + 1}.0.", dim, heads, key_bias_value))
        if stage == len(DIMS) - 1:  # 最后一个 stage 后面没有 PatchMerging
            break
        merge_dim = 4 * dim
        state[f"features.{2 * stage + 2}.norm.weight"] = torch.ones(merge_dim)
        state[f"features.{2 * stage + 2}.norm.bias"] = torch.zeros(merge_dim)
        state[f"features.{2 * stage + 2}.reduction.weight"] = torch.zeros(2 * dim, merge_dim)
    state["norm.weight"] = torch.ones(DIMS[-1])
    state["norm.bias"] = torch.zeros(DIMS[-1])
    state["head.weight"] = torch.zeros(1000, DIMS[-1])
    state["head.bias"] = torch.zeros(1000)
    return state


def test_maps_blocks_mlp_and_downsample():
    """MLP 走 mlp.0/3 → mlp.fc1/fc2；PatchMerging 挂到下一层的 downsample。"""
    mapped, unconsumed = map_torchvision_swinv2(_fake_swinv2())
    assert torch.equal(mapped["patch_embed.proj.weight"], torch.zeros(DIMS[0], 3, 4, 4))
    assert torch.equal(mapped["layers.0.blocks.0.mlp.fc1.weight"], torch.full((4 * DIMS[0], DIMS[0]), 2.0))
    assert torch.equal(mapped["layers.0.blocks.0.mlp.fc2.weight"], torch.full((DIMS[0], 4 * DIMS[0]), 3.0))
    # features.2 是 4→8 的合并，在 timm 里属于 layers.1.downsample；layers.0.downsample 是 Identity
    assert "layers.0.downsample.norm.weight" not in mapped
    assert mapped["layers.1.downsample.norm.weight"].shape == torch.Size([4 * DIMS[0]])
    assert mapped["layers.1.downsample.reduction.weight"].shape == torch.Size([2 * DIMS[0], 4 * DIMS[0]])
    assert torch.equal(mapped["norm.weight"], torch.ones(DIMS[-1]))
    assert not any(key.startswith("head.") for key in mapped)


def test_splits_qkv_bias_and_drops_k():
    """qkv 偏置拆成 q/v 两段；k 段全零时无损，head 与位置 buffer 归入未消费。"""
    mapped, unconsumed = map_torchvision_swinv2(_fake_swinv2())
    dim = DIMS[0]
    assert torch.equal(mapped["layers.0.blocks.0.attn.q_bias"], torch.full((dim,), 5.0))
    assert torch.equal(mapped["layers.0.blocks.0.attn.v_bias"], torch.full((dim,), 6.0))
    assert any(key.startswith("head.") for key in unconsumed)
    assert any(key.endswith("attn.relative_coords_table") for key in unconsumed)
    assert all("relative_position_index" in key or "head." in key or "relative_coords_table" in key
               for key in unconsumed)


def test_rejects_nonzero_key_bias():
    """k 偏置非零说明来源实现与 timm 不同源，映射会丢信息，必须报错。"""
    with pytest.raises(ValueError, match="k 段不是全零"):
        map_torchvision_swinv2(_fake_swinv2(key_bias_value=0.5))


def test_detects_foreign_format():
    """缺特征键时不做映射，提示调用方拿错了文件。"""
    assert looks_like_torchvision_swinv2(_fake_swinv2())
    assert not looks_like_torchvision_swinv2({"backbone.weight": torch.zeros(2)})
