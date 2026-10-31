"""模型与损失：study 级均值聚合、参数规模、BCE 的 pos_weight 与 ASL 计算。"""

import math
from pathlib import Path
from dataclasses import replace

import numpy as np
import pytest
import torch

from common.config import Config, DataConfig, ModelConfig, SplitConfig, TrainConfig
from model.losses import AsymmetricLoss, build_loss
from infer.predict import predict_loader
from model.network import aggregate_logits, build_model


def _config(pretrained: bool = False, pos_weight: bool = False, loss: str = "bce") -> Config:
    """只关心模型与损失的配置。

    data 里的路径只是占位：build_model / build_loss 都不会去读数据，
    这样测试不需要造图片目录，跑得也快。
    """
    split = SplitConfig(csv=Path("unused.csv"), images=Path("unused"))
    return Config(
        data=DataConfig(train=split, valid=split, test=split),
        model=ModelConfig(backbone="resnet18", pretrained=pretrained),
        train=TrainConfig(pos_weight=pos_weight, loss=loss),
    )


def test_build_model_outputs_ten_classes():
    """ResNet-18 + Linear(512, 10)：输入 [N,3,64,64]，输出 [N,10] 个 logits。

    测试里用 64px 输入而不是 320px：形状规则一样（32 倍下采样），但快得多。
    pretrained=False 避免测试时联网下载权重。
    """
    model = build_model(_config()).eval()

    logits = model(torch.randn(2, 3, 64, 64))

    assert logits.shape == (2, 10)


def test_build_model_parameter_count_is_deployment_friendly():
    """参数量约 11.2M，对应部署约束「模型 ≤500MB」。"""
    model = build_model(_config())

    params = sum(parameter.numel() for parameter in model.parameters())

    assert 10_000_000 < params < 13_000_000


def test_freeze_backbone_leaves_only_classifier_trainable():
    config = replace(
        _config(),
        model=replace(_config().model, freeze_backbone=True),
    )
    model = build_model(config)
    classifier = model.get_classifier()

    assert isinstance(classifier, torch.nn.Module)
    classifier_parameters = {id(parameter) for parameter in classifier.parameters()}
    trainable_parameters = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    assert classifier_parameters
    assert trainable_parameters == classifier_parameters
    assert all(
        parameter.requires_grad
        for parameter in classifier.parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in model.parameters()
        if id(parameter) not in classifier_parameters
    )


def test_aggregate_logits_averages_within_each_study():
    """3 个 study 分别有 2、1、3 张图：每个 study 取自己那几张图的均值。"""
    logits = torch.tensor(
        [
            [1.0, 2.0],  # study 0，第 1 张
            [3.0, 4.0],  # study 0，第 2 张
            [10.0, 20.0],  # study 1，只有 1 张
            [2.0, 2.0],  # study 2，第 1 张
            [4.0, 4.0],  # study 2，第 2 张
            [6.0, 6.0],  # study 2，第 3 张
        ]
    )
    offsets = torch.tensor([0, 2, 3, 6])

    aggregated = aggregate_logits(logits, offsets)

    expected = torch.tensor([[2.0, 3.0], [10.0, 20.0], [4.0, 4.0]])
    assert torch.allclose(aggregated, expected)


def test_aggregate_logits_keeps_gradient():
    """聚合必须可导：2 张图的 study 里每张图分到 1/2 的梯度，单图 study 是 1。"""
    logits = torch.randn(3, 10, requires_grad=True)

    aggregate_logits(logits, torch.tensor([0, 2, 3])).sum().backward()

    assert torch.allclose(logits.grad[:2], torch.full((2, 10), 0.5))
    assert torch.allclose(logits.grad[2], torch.ones(10))


def test_aggregate_logits_probability_space_matches_mean_of_sigmoid():
    """probability_space=True 返回 logit(mean(σ(z)))，sigmoid 后就是概率空间均值（G1）。"""
    logits = torch.tensor([[4.0], [-4.0], [2.0]]).repeat(1, 10)
    offsets = torch.tensor([0, 2, 3])
    views = torch.tensor([0, 1, 0])

    probs = torch.sigmoid(aggregate_logits(logits, offsets, views, 0.7, True))

    expected = 0.7 * torch.sigmoid(torch.tensor(4.0)) + 0.3 * torch.sigmoid(torch.tensor(-4.0))
    assert probs[0, 0] == pytest.approx(expected.item(), abs=1e-6)
    # 单图 study 两个空间必然一致
    assert probs[1, 0] == pytest.approx(torch.sigmoid(torch.tensor(2.0)).item(), abs=1e-6)

    # 与 G0 对照：logits 空间加权得到 σ(0.7·4 + 0.3·(-4)) = σ(1.6)，两者不同。
    g0 = torch.sigmoid(aggregate_logits(logits, offsets, views, 0.7, False))
    assert g0[0, 0].item() == pytest.approx(0.8320184, abs=1e-5)
    assert probs[0, 0] < g0[0, 0]


def test_aggregate_logits_probability_space_stays_finite_at_extremes():
    """端到端饱和时用 EPS 夹住，不能出现 inf / nan。"""
    logits = torch.tensor([[80.0], [-80.0]]).repeat(1, 10)

    merged = aggregate_logits(logits, torch.tensor([0, 1, 2]), None, 0.7, True)

    assert torch.isfinite(merged).all()
    probs = torch.sigmoid(merged)
    assert probs[0, 0] == pytest.approx(1.0, abs=1e-5)
    assert probs[1, 0] == pytest.approx(0.0, abs=1e-5)


def test_aggregate_logits_probability_space_keeps_gradient():
    """概率空间聚合也要可导，否则训练第一步就崩。"""
    logits = torch.randn(3, 10, requires_grad=True)

    aggregate_logits(logits, torch.tensor([0, 2, 3]), None, 0.7, True).sum().backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert (logits.grad.abs() > 0).all()


def test_default_loss_has_no_pos_weight():
    """默认就是普通 BCEWithLogitsLoss，不传训练标签也能构建。"""
    loss = build_loss(_config())

    assert loss.pos_weight is None


def test_pos_weight_from_class_frequency():
    """pos_weight = 负样本数 / 正样本数：第 1 类 3 正 7 负 → 7/3；第 2 类 1 正 9 负 → 9。"""
    labels = np.zeros((10, 10))
    labels[:3, 0] = 1.0
    labels[:1, 1] = 1.0

    loss = build_loss(_config(pos_weight=True), labels)

    expected = np.ones(10)
    expected[0] = 7 / 3
    expected[1] = 9.0
    assert torch.allclose(loss.pos_weight, torch.tensor(expected, dtype=torch.float32))


def test_pos_weight_requires_training_labels():
    """打开 pos_weight 却没给训练标签时直接报错，避免悄悄退化成普通 BCE。"""
    with pytest.raises(ValueError, match="train_labels"):
        build_loss(_config(pos_weight=True))


def test_pos_weight_loss_is_finite():
    """加权之后的 loss 仍是有限值（没有除零产生的 inf / nan）。"""
    labels = np.zeros((10, 10))
    labels[:3, 0] = 1.0

    loss = build_loss(_config(pos_weight=True), labels)
    value = loss(torch.randn(4, 10), torch.zeros(4, 10))

    assert torch.isfinite(value)


def test_build_model_records_pretrained_state():
    """随机初始化时也要显式记录，训练日志和 checkpoint 才能区分是否真的加载了预训练。"""
    model = build_model(_config(pretrained=False))

    assert model.pretrained_loaded is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")
def test_pos_weight_on_cuda_is_finite():
    """pos_weight 必须和 logits 在同一设备上，否则 CUDA 上会直接 device mismatch。"""
    labels = np.zeros((10, 10))
    labels[:3, 0] = 1.0

    loss = build_loss(_config(pos_weight=True), labels, device="cuda")
    assert loss.pos_weight is not None
    assert loss.pos_weight.device.type == "cuda"

    value = loss(
        torch.randn(4, 10, device="cuda"),
        torch.zeros(4, 10, device="cuda"),
    )
    assert torch.isfinite(value)


def test_pos_weight_all_positive_class_is_not_zero():
    """全正类若权重取 0，正样本 loss 恒为 0，模型永远学不到该类。"""
    labels = np.ones((4, 10))
    labels[:, 1:] = 0.0

    loss = build_loss(_config(pos_weight=True), labels)

    assert loss.pos_weight[0] == 1.0


def test_build_loss_returns_asl_with_configured_hyperparameters():
    """`train.loss=asl` 时构建 ASL，超参从配置透传，且不需要训练标签统计。"""
    config = _config(loss="asl")

    loss = build_loss(config)

    assert isinstance(loss, AsymmetricLoss)
    assert (loss.gamma_pos, loss.gamma_neg, loss.clip) == (0.0, 4.0, 0.05)


def test_asl_reduces_to_bce_when_focus_and_shift_are_off():
    """gamma=0、clip=0 时 ASL 应等于普通 BCE：验证实现没有引入额外偏差。"""
    torch.manual_seed(0)
    logits = torch.randn(16, 10) * 3
    targets = (torch.rand(16, 10) > 0.7).float()

    asl = AsymmetricLoss(gamma_pos=0.0, gamma_neg=0.0, clip=0.0)
    bce = torch.nn.BCEWithLogitsLoss()

    assert torch.allclose(asl(logits, targets), bce(logits, targets), atol=1e-6)


def test_asl_matches_hand_computed_reference():
    """2×2 手算：正样本项不衰减，负样本项先移位再按 (1-p_neg)^gamma_neg 聚焦。"""
    logits = torch.zeros(1, 2)
    targets = torch.tensor([[1.0, 0.0]])

    loss = AsymmetricLoss(gamma_pos=0.0, gamma_neg=4.0, clip=0.05)(logits, targets)

    # σ(0)=0.5；负样本概率移位为 1-0.5+0.05=0.55
    expected_pos = -math.log(0.5)
    expected_neg = -math.log(0.55) * (1 - 0.55) ** 4
    assert loss.item() == pytest.approx((expected_pos + expected_neg) / 2, rel=1e-6)


def test_asl_clip_and_gamma_neg_monotonically_suppress_negative_loss():
    """clip 越大、gamma_neg 越大，简单负样本的 loss 越小（方向性检查）。"""
    torch.manual_seed(0)
    logits = torch.randn(8, 10)
    targets = torch.zeros(8, 10)

    plain = AsymmetricLoss(gamma_pos=0.0, gamma_neg=0.0, clip=0.0)(logits, targets)
    shifted = AsymmetricLoss(gamma_pos=0.0, gamma_neg=0.0, clip=0.05)(logits, targets)
    focused = AsymmetricLoss(gamma_pos=0.0, gamma_neg=4.0, clip=0.05)(logits, targets)

    assert shifted < plain
    assert focused < shifted


def test_asl_gradient_is_finite_for_extreme_logits():
    """极端 logits 也不应产生 nan / inf 梯度（混合精度下会先统一到 fp32）。"""
    logits = torch.tensor([[-30.0, 30.0]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0]])

    AsymmetricLoss()(logits, targets).backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_predict_loader_empty_raises():
    """空 loader 不能走到 torch.cat([])，否则报错信息很难定位。"""
    with pytest.raises(RuntimeError, match="推理集为空"):
        predict_loader(build_model(_config()), [], torch.device("cpu"))


def test_view_weighting_uses_group_means_and_unknown_fallback():
    logits = torch.tensor([[2.], [4.], [10.], [3.], [9.]], requires_grad=True)
    offsets = torch.tensor([0, 3, 5])
    views = torch.tensor([0, 0, 1, 0, 2])
    result = aggregate_logits(logits, offsets, views, 0.7)
    assert torch.allclose(result, torch.tensor([[5.1], [6.0]]))
    result.sum().backward()
    assert torch.allclose(logits.grad[:, 0], torch.tensor([.35, .35, .3, .5, .5]))


def test_weighted_loader_and_study_prediction_match(labeled_split):
    from dataclasses import replace
    from PIL import Image
    from data.dataset import StudyDataset, collate_studies
    from data.index import read_split
    from data.transforms import build_eval_transform
    from infer.predict import predict_study
    from torch.utils.data import DataLoader

    records = read_split(labeled_split["csv"], labeled_split["images"])
    Image.new("RGB", (64, 48), (240, 240, 240)).save(records[0].image_paths[1])
    transform = build_eval_transform(32, normalize="none")
    loader = DataLoader(StudyDataset(records, transform), batch_size=2, collate_fn=collate_studies)

    class PixelModel(torch.nn.Module):
        def forward(self, images):
            return images.mean(dim=(1, 2, 3)).unsqueeze(1).repeat(1, 10)

    config = _config()
    config = replace(config, model=replace(config.model, aggregation="view_weighted", frontal_weight=0.7))
    model = PixelModel()
    result = predict_loader(model, loader, torch.device("cpu"), config)
    direct = predict_study(model, records[0].image_paths, transform, torch.device("cpu"), records[0].views, 0.7)
    np.testing.assert_allclose(result.iloc[0, 1:].to_numpy(dtype=float), direct, atol=1e-7)
    logits = torch.stack([transform(Image.open(path)).mean() for path in records[0].image_paths])
    assert direct[0] == pytest.approx(torch.sigmoid(0.7 * logits[0] + 0.3 * logits[1]).item())


def test_predict_loader_probability_space_matches_weighted_mean_of_sigmoid(labeled_split):
    """A-008 口径：predict_loader 输出应等于 0.7·σ(正位) + 0.3·σ(侧位)。

    这条链路覆盖 config.model.probability_space → predict_loader →
    aggregate_logits(probability_space=True) → 概率空间加权，是训练与提交共用的
    同一段代码，所以只要这里对，训练时的 loss 输入也是概率空间均值。
    """
    from data.dataset import StudyDataset, collate_studies
    from data.index import read_split
    from data.transforms import build_eval_transform
    from PIL import Image
    from torch.utils.data import DataLoader

    records = read_split(labeled_split["csv"], labeled_split["images"])
    # study_00000 的第一张是正位，把第二张（侧位）改成完全不同的像素值。
    Image.new("RGB", (64, 48), (10, 200, 30)).save(records[0].image_paths[1])
    transform = build_eval_transform(32, normalize="none")
    loader = DataLoader(StudyDataset(records, transform), batch_size=2, collate_fn=collate_studies)

    class PixelModel(torch.nn.Module):
        def forward(self, images):
            return images.mean(dim=(1, 2, 3)).unsqueeze(1).repeat(1, 10)

    base = _config()
    config = replace(
        base,
        model=replace(
            base.model, aggregation="view_weighted_probability", frontal_weight=0.7
        ),
    )
    assert config.model.uses_views is True and config.model.probability_space is True

    result = predict_loader(PixelModel(), loader, torch.device("cpu"), config)

    logits = torch.stack([transform(Image.open(path)).mean() for path in records[0].image_paths])
    expected = 0.7 * torch.sigmoid(logits[0]) + 0.3 * torch.sigmoid(logits[1])
    assert result.iloc[0, 1] == pytest.approx(expected.item(), abs=1e-6)
    # 与 logits 空间的结果必须有差别，否则说明开关没生效。
    logit_space = torch.sigmoid(0.7 * logits[0] + 0.3 * logits[1])
    assert abs(result.iloc[0, 1] - logit_space.item()) > 1e-6
