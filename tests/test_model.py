"""模型与损失：study 级均值聚合、参数规模、BCE 的 pos_weight 计算。"""

from pathlib import Path

import numpy as np
import pytest
import torch

from common.config import Config, DataConfig, ModelConfig, SplitConfig, TrainConfig
from model.losses import build_loss
from infer.predict import predict_loader
from model.network import aggregate_logits, build_model


def _config(pretrained: bool = False, pos_weight: bool = False) -> Config:
    """只关心模型与损失的配置。

    data 里的路径只是占位：build_model / build_loss 都不会去读数据，
    这样测试不需要造图片目录，跑得也快。
    """
    split = SplitConfig(csv=Path("unused.csv"), images=Path("unused"))
    return Config(
        data=DataConfig(train=split, valid=split, test=split),
        model=ModelConfig(backbone="resnet18", pretrained=pretrained),
        train=TrainConfig(pos_weight=pos_weight),
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
