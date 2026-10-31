"""B 组 TxRV DenseNet-121 适配测试（docs/train_plan12.md 步骤③）。

全部使用合成图与小尺寸输入（32/64/224），不碰真实影像：本机没有赛题数据，
也不应该把影像带进仓库。真实权重相关的用例在缓存缺失时自动 skip，这样没装
torchxrayvision 或没下载权重的机器上 `pytest` 仍然全绿。

覆盖要点：

1. B-001 配置把 plan12 的所有口径锁死（224、单通道、xrv 归一化、10 类、
   view_weighted、25 轮 / batch 16 / lr 3e-4 / seed 42）；
2. 单通道 transform 的形状与值域 [-1024, 1024]；
3. 配置在 xrv 口径下的互斥校验（通道数、freeze、SHA-256 格式）；
4. 未加载预训练时仍输出 10 类原始 logits，且 `op_threshs` 被清空；
5. 权重加载失败必须报错、不回退随机初始化；
6. 多图聚合后的 loss 能反传到骨干；
7. checkpoint 重建（pretrained=False）加载同一份 state_dict 后输出一致；
8. SHA-256 不匹配立即报错。
"""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from common.config import ImageConfig, ModelConfig, load_config
from data.dataset import build_loader
from data.transforms import build_eval_transform, build_train_transform
from model.network import aggregate_logits, build_model
from model.xrv import (
    XrvDenseNetClassifier,
    _import_torchxrayvision_models,
    file_sha256,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
B001_CONFIG = REPO_ROOT / "configs" / "exp_B001.yaml"
EXPECTED_SHA256 = "56524913dd16a906422e8d8b66a7a5c46be1d82eb7ac012d8103776f1aa68899"


def _b_config():
    return load_config(B001_CONFIG)


def _pretrained_false(cfg):
    """checkpoint 重建 / 测试用的随机初始化结构配置。"""
    return replace(cfg, model=replace(cfg.model, pretrained=False))


def _cached_weights_path() -> Path | None:
    """按 TxRV 默认缓存目录找 densenet121-res224-all 权重；没有则返回 None。

    不直接 `import torchxrayvision`：本仓库的顶层 `model` 包与 TxRV 1.5.7 的
    `baseline_models` 冲突，必须经过 `model.xrv` 里的兼容导入。
    """
    try:
        models = _import_torchxrayvision_models()
    except ImportError:
        return None
    url = models.model_urls["densenet121-res224-all"]["weights_url"]
    path = Path.home() / ".torchxrayvision" / "models_data" / Path(url).name
    return path if path.is_file() else None


def test_b001_config_locks_plan12_settings():
    """B-001 继承 default.yaml 后，把 plan12 的固定口径全部覆盖住。"""
    cfg = _b_config()

    assert cfg.image.size == 224
    assert cfg.image.letterbox is True
    assert cfg.image.normalize == "xrv"
    assert cfg.image.in_channels == 1
    assert cfg.image.xrv_maxval == 255
    assert cfg.image.hflip is False

    assert cfg.model.arch == "xrv_densenet121"
    assert cfg.model.pretrained is True
    assert cfg.model.num_classes == 10
    assert cfg.model.aggregation == "view_weighted"
    assert cfg.model.frontal_weight == 0.7
    assert cfg.model.xrv_weights == "densenet121-res224-all"
    assert cfg.model.xrv_sha256 == EXPECTED_SHA256
    assert cfg.model.freeze_backbone is False

    assert cfg.train.epochs == 25
    assert cfg.train.batch_studies == 16
    assert cfg.train.lr == pytest.approx(3e-4)
    assert cfg.train.weight_decay == pytest.approx(0.01)
    assert cfg.train.pos_weight is True
    assert cfg.train.warmup_epochs == 2
    assert cfg.train.lr_scheduler == "cosine"
    assert cfg.train.lr_min == pytest.approx(1e-5)
    assert cfg.train.early_stop_patience == 5
    assert cfg.train.loss == "bce"

    assert cfg.seed == 42


def test_eval_transform_single_channel_value_range():
    """xrv 归一化：单通道 [1,224,224]，像素 255 -> 1024，补边 0 -> -1024。"""
    from PIL import Image

    transform = build_eval_transform(
        224, letterbox=True, normalize="xrv", in_channels=1, xrv_maxval=255
    )

    full_white = transform(Image.new("L", (224, 224), 255))
    assert tuple(full_white.shape) == (1, 224, 224)
    assert full_white.dtype == torch.float32
    assert float(full_white.min()) == pytest.approx(1024.0)
    assert float(full_white.max()) == pytest.approx(1024.0)

    # 64x48 的非方形图：letterbox 补边是 -1024，内容保持 1024。
    letterboxed = transform(Image.new("L", (64, 48), 255))
    assert tuple(letterboxed.shape) == (1, 224, 224)
    assert float(letterboxed.min()) == pytest.approx(-1024.0)
    assert float(letterboxed.max()) == pytest.approx(1024.0)


def test_imagenet_transform_still_three_channels():
    """xrv 之外的老路径不受影响：imagenet 归一化仍是 3 通道。"""
    from PIL import Image

    transform = build_eval_transform(32, letterbox=True, normalize="imagenet")
    tensor = transform(Image.new("RGB", (32, 32), (120, 120, 120)))
    assert tuple(tensor.shape) == (3, 32, 32)


def test_train_transform_single_channel_shape():
    """训练侧增强（RandomAffine/ColorJitter）在单通道上也能跑，值域不越界。"""
    from PIL import Image

    image_cfg = ImageConfig(size=32, normalize="xrv", in_channels=1, xrv_maxval=255)
    transform = build_train_transform(32, image_cfg)
    tensor = transform(Image.new("L", (40, 30), 200))

    assert tuple(tensor.shape) == (1, 32, 32)
    assert float(tensor.min()) >= -1024.0
    assert float(tensor.max()) <= 1024.0
    # 仿射补边是 0，归一化后为 -1024；原图 200 的像素落在正值区间。
    assert float(tensor.max()) > 0.0


def test_xrv_transform_rejects_pixel_above_maxval():
    """像素超过 xrv_maxval 时必须报错，避免悄悄把 16 位数据当 8 位处理。"""
    from PIL import Image

    transform = build_eval_transform(
        32, letterbox=True, normalize="xrv", in_channels=1, xrv_maxval=100
    )
    with pytest.raises(ValueError, match="xrv_maxval"):
        transform(Image.new("L", (32, 32), 120))


def test_config_rejects_incompatible_xrv_settings():
    """xrv 口径下通道数、freeze_backbone、SHA-256 格式都必须自洽。"""
    with pytest.raises(ValueError, match="xrv"):
        ImageConfig(normalize="xrv")  # 默认 in_channels=3

    with pytest.raises(ValueError, match="imagenet"):
        ImageConfig(normalize="imagenet", in_channels=1)

    with pytest.raises(ValueError, match="in_channels"):
        ImageConfig(in_channels=2)

    with pytest.raises(ValueError, match="freeze_backbone"):
        ModelConfig(arch="xrv_densenet121", freeze_backbone=True)

    with pytest.raises(ValueError, match="xrv_sha256"):
        ModelConfig(arch="xrv_densenet121", xrv_sha256="abc")

    with pytest.raises(ValueError, match="arch"):
        ModelConfig(arch="unknown")


def test_build_model_without_pretrained_outputs_ten_logits():
    """pretrained=False 只构建结构：10 类、原始 logits、无阈值归一化。"""
    model = build_model(_pretrained_false(_b_config())).eval()

    assert isinstance(model, XrvDenseNetClassifier)
    assert model.pretrained_info()["pretrained_loaded"] is False
    assert model.backbone.op_threshs is None
    assert model.backbone.apply_sigmoid is False
    assert model.get_classifier().out_features == 10

    with torch.no_grad():
        logits = model(torch.zeros(2, 1, 64, 64))
    assert tuple(logits.shape) == (2, 10)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()


def test_pretrained_load_failure_raises_without_fallback(monkeypatch):
    """加载失败必须报 RuntimeError 并说明不回退，不能静默用随机初始化。"""

    class _FailingDenseNet:
        def __init__(self, *args, **kwargs):
            raise OSError("pretrained file download failed")

    fake_models = SimpleNamespace(DenseNet=_FailingDenseNet)
    monkeypatch.setattr("model.xrv._import_torchxrayvision_models", lambda: fake_models)

    with pytest.raises(RuntimeError, match="不会回退到随机初始化"):
        build_model(_b_config())


def test_backward_through_aggregated_logits():
    """同一 study 多图 -> aggregate_logits -> loss 的梯度能回到骨干。"""
    model = build_model(_pretrained_false(_b_config()))
    model.train()
    images = torch.zeros(2, 1, 64, 64)
    images[0, 0, 8:56, 8:56] = 512.0
    images[1, 0, 16:48, 16:48] = -512.0

    per_image = model(images)
    # offsets=[0, 2] 表示 1 个 study、2 张图；views 与图片一一对应。
    study_logits = aggregate_logits(
        per_image, torch.tensor([0, 2]), torch.tensor([0, 1]), 0.7
    )
    assert tuple(study_logits.shape) == (1, 10)

    study_logits.sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "聚合后的 loss 没有任何梯度传回模型"
    assert float(sum(g.abs().sum() for g in grads)) > 0.0


def test_checkpoint_rebuild_matches_logits():
    """真实权重：224 单通道前向 10 类 logits，pretrained=False 重建后输出一致。"""
    weights_path = _cached_weights_path()
    if weights_path is None:
        pytest.skip("本机没有 densenet121-res224-all 权重缓存")

    cfg = _b_config()
    model = build_model(cfg).eval()
    info = model.pretrained_info()
    assert info["pretrained_loaded"] is True
    assert info["weights_sha256"] == EXPECTED_SHA256
    params = sum(p.numel() for p in model.parameters())
    assert 6_000_000 < params < 8_000_000

    images = torch.zeros(1, 1, 224, 224)
    images[0, 0, 20:200, 20:200] = 1024.0
    with torch.no_grad():
        expected = model(images)
    assert tuple(expected.shape) == (1, 10)

    rebuilt = build_model(_pretrained_false(cfg)).eval()
    rebuilt.load_state_dict(model.state_dict())
    with torch.no_grad():
        actual = rebuilt(images)
    assert torch.allclose(expected, actual, atol=1e-6)


def test_real_weights_sha256_matches_config():
    """真实权重文件的哈希必须与 B-001 配置写死的一致。"""
    weights_path = _cached_weights_path()
    if weights_path is None:
        pytest.skip("本机没有 densenet121-res224-all 权重缓存")

    assert file_sha256(weights_path) == EXPECTED_SHA256
    assert _b_config().model.xrv_sha256 == EXPECTED_SHA256


def test_real_weights_sha256_mismatch_raises():
    """配置里的 SHA-256 与文件不符时必须立即报错，不能继续训练。"""
    weights_path = _cached_weights_path()
    if weights_path is None:
        pytest.skip("本机没有 densenet121-res224-all 权重缓存")

    cfg = _b_config()
    broken = replace(cfg, model=replace(cfg.model, xrv_sha256="0" * 64))
    with pytest.raises(RuntimeError, match="SHA-256 不匹配"):
        build_model(broken)


def test_loader_batch_uses_single_channel(
    write_config, labeled_split, unlabeled_split
):
    """沿用既有 loader：xrv 配置下 batch 图是 [N,1,H,W]，值域不越界。"""
    config_path = write_config(
        {
            "train": (labeled_split["csv"], labeled_split["images"]),
            "valid": (labeled_split["csv"], labeled_split["images"]),
            "test": (unlabeled_split["csv"], unlabeled_split["images"]),
        }
    )
    cfg = load_config(config_path)
    cfg = replace(
        cfg,
        image=replace(
            cfg.image, size=64, normalize="xrv", in_channels=1, xrv_maxval=255
        ),
    )

    batch = next(iter(build_loader(cfg, "valid")))
    # eval batch = 2 个 study：study_00000 有 2 张图、study_00001 有 1 张，
    # 所以第一个 batch 是 3 张图，但通道维度必须变成单通道。
    assert batch["images"].shape[0] == 3
    assert tuple(batch["images"].shape[1:]) == (1, 64, 64)
    assert batch["images"].dtype == torch.float32
    assert float(batch["images"].min()) >= -1024.0
    assert float(batch["images"].max()) <= 1024.0
