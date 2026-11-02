"""TensorBoard 记录测试：writer 的三种模式、逐轮标量落盘、逐类曲线。

tensorboard 是可选依赖（见 requirements.txt 的注释），所以这里分两类测试：

- 不依赖 tensorboard 的：模式校验、`false` 完全不建目录、`auto` 永不抛错、
  配置项校验。这些在任何环境都必须通过。
- 依赖 tensorboard 的：真的写出 events 文件并读回来核对 tag。没装则 skip。
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from common.config import Config, load_config
from common.tracking import (
    TENSORBOARD_DIR_NAME,
    TENSORBOARD_EMA_DIR_NAME,
    TENSORBOARD_MODES,
    build_writer,
)
from data.dataset import build_loader
from model.network import build_model
from train.engine import evaluate, fit

# epoch 级曲线：每轮必有，与 metrics.csv 的列一一对应。
EPOCH_TAGS = {
    "train/lr",
    "train/loss",
    "valid/loss",
    "valid/macro_auc",
    "valid/macro_f1",
    "valid/best_macro_auc",
    "valid/macro_precision",
    "valid/macro_recall",
}

# EMA 是独立面板：只有验证曲线，tag 与 raw 面板同名（没有 train/lr、train/loss）。
EMA_TAGS = EPOCH_TAGS - {"train/lr", "train/loss"}


def _tensorboard_available() -> bool:
    """能否导入 SummaryWriter，与 `build_writer` 的判断口径保持一致。"""
    try:
        import torch.utils.tensorboard  # noqa: F401
    except ImportError:
        return False
    return True


requires_tensorboard = pytest.mark.skipif(
    not _tensorboard_available(),
    reason="需要 tensorboard：python -m pip install tensorboard",
)
requires_no_tensorboard = pytest.mark.skipif(
    _tensorboard_available(),
    reason="本环境已装 tensorboard，无法验证降级路径",
)


def _read_scalars(log_dir: Path) -> dict[str, list[float]]:
    """把 events 文件里的标量读回来，返回 {tag: [各轮取值]}。"""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(str(log_dir))
    accumulator.Reload()
    return {
        tag: [event.value for event in accumulator.Scalars(tag)]
        for tag in accumulator.Tags()["scalars"]
    }


@pytest.fixture
def config(tmp_path: Path, labeled_split: dict, write_config) -> Config:
    """指向合成数据的配置：CPU、2 个 study、训练 2 轮且关闭早停。"""
    split = (labeled_split["csv"], labeled_split["images"])
    config_path = write_config({"train": split, "valid": split, "test": split})
    _config = load_config(config_path)
    return replace(
        _config,
        model=replace(_config.model, pretrained=False),
        train=replace(
            _config.train,
            epochs=2,
            warmup_epochs=0,
            early_stop_patience=0,
        ),
    )


# ---------- 不依赖 tensorboard ----------


def test_mode_false_creates_nothing(tmp_path: Path) -> None:
    """false 模式下连目录都不该建，避免留下空 tb/ 让人以为记过曲线。"""
    log_dir = tmp_path / TENSORBOARD_DIR_NAME
    assert build_writer(log_dir, "false") is None
    assert not log_dir.exists()


def test_unknown_mode_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="train.tensorboard"):
        build_writer(tmp_path / TENSORBOARD_DIR_NAME, "yes")


def test_auto_never_raises(tmp_path: Path) -> None:
    """auto 的含义就是「能用就用，不能用就算了」，不该让训练因为缺包而中断。"""
    writer = build_writer(tmp_path / TENSORBOARD_DIR_NAME, "auto")
    if _tensorboard_available():
        assert writer is not None
        writer.close()
    else:
        assert writer is None
        assert not (tmp_path / TENSORBOARD_DIR_NAME).exists()


@requires_no_tensorboard
def test_mode_true_fails_loudly_without_tensorboard(tmp_path: Path) -> None:
    """显式要求写曲线却缺包时，必须报错，而不是默默生成空面板。"""
    with pytest.raises(ImportError, match="tensorboard"):
        build_writer(tmp_path / TENSORBOARD_DIR_NAME, "true")


@requires_no_tensorboard
def test_auto_prints_hint_when_missing(tmp_path: Path, capsys) -> None:
    build_writer(tmp_path / TENSORBOARD_DIR_NAME, "auto")
    assert "未安装 tensorboard" in capsys.readouterr().out


def test_config_rejects_unknown_tensorboard_mode(config: Config) -> None:
    with pytest.raises(ValueError, match="train.tensorboard"):
        replace(config, train=replace(config.train, tensorboard="on"))


def test_config_default_mode_is_auto(config: Config) -> None:
    assert config.train.tensorboard == "auto"
    assert config.train.tensorboard in TENSORBOARD_MODES


def test_config_roundtrip_keeps_tensorboard() -> None:
    """to_dict / config_from_dict 必须带上这个新字段，否则续训读回默认值。"""
    from common.config import config_from_dict

    configs = Path(__file__).resolve().parents[1] / "configs"
    exported = load_config(configs / "default.yaml").to_dict()
    assert exported["train"]["tensorboard"] == "auto"
    assert config_from_dict(exported).train.tensorboard == "auto"


@requires_tensorboard
def test_mode_true_creates_writer(tmp_path: Path) -> None:
    log_dir = tmp_path / TENSORBOARD_DIR_NAME
    writer = build_writer(log_dir, "true")
    assert writer is not None
    writer.close()


# ---------- 依赖 tensorboard ----------


@requires_tensorboard
def test_fit_writes_epoch_curves(config: Config, tmp_path: Path) -> None:
    """核心用例：训练 2 轮后，events 文件里应当有全部 epoch 级 tag，各 2 个点。"""
    log_dir = tmp_path / TENSORBOARD_DIR_NAME
    writer = build_writer(log_dir, "true")
    assert writer is not None
    try:
        fit(
            config,
            build_model(config),
            build_loader(config, "train", train=True),
            build_loader(config, "valid"),
            torch.device("cpu"),
            writer=writer,
        )
    finally:
        writer.close()

    scalars = _read_scalars(log_dir)
    assert set(scalars) >= EPOCH_TAGS
    for tag in EPOCH_TAGS:
        assert len(scalars[tag]) == 2, f"{tag} 应当每个 epoch 一个点"
    # EMA 有自己的面板，raw 面板里不该出现任何 ema/ 前缀的曲线。
    assert not [tag for tag in scalars if tag.startswith("ema/")]


@requires_tensorboard
def test_fit_writes_ema_in_separate_panel(config: Config, tmp_path: Path) -> None:
    """开了 EMA 时曲线写进 tb_ema/：独立面板、tag 与 raw 同名（只少训练侧两条）。"""
    ema_config = replace(config, train=replace(config.train, ema_enabled=True))
    raw_dir = tmp_path / TENSORBOARD_DIR_NAME
    ema_dir = tmp_path / TENSORBOARD_EMA_DIR_NAME
    raw_writer = build_writer(raw_dir, "true")
    ema_writer = build_writer(ema_dir, "true")
    assert raw_writer is not None and ema_writer is not None
    try:
        fit(
            ema_config,
            build_model(ema_config),
            build_loader(ema_config, "train", train=True),
            build_loader(ema_config, "valid"),
            torch.device("cpu"),
            writer=raw_writer,
            ema_writer=ema_writer,
        )
    finally:
        raw_writer.close()
        ema_writer.close()

    raw_scalars = _read_scalars(raw_dir)
    ema_scalars = _read_scalars(ema_dir)
    # 两个面板互不混入：raw 没有 ema/ 前缀，EMA 没有训练侧曲线。
    assert not [tag for tag in raw_scalars if tag.startswith("ema/")]
    assert not [tag for tag in ema_scalars if tag.startswith("train/")]
    # EMA 面板与 raw 同构：8 条汇总曲线 + 逐类曲线，只少训练侧两条。
    assert EMA_TAGS <= set(ema_scalars)
    assert {tag for tag in ema_scalars if tag.startswith("valid/auc/class_")}
    for tag in EMA_TAGS:
        assert len(ema_scalars[tag]) == 2, f"{tag} 应当每个 epoch 一个点"
    # 另有一条 EMA 减 raw 的差值曲线，正数表示 EMA 更好。
    assert len(ema_scalars["ema_minus_raw/macro_auc"]) == 2
    # 同名 tag 在两边都有，横轴同为 epoch，才能在 TensorBoard 里同图对比 raw 与 EMA。
    assert EMA_TAGS <= set(raw_scalars)


@requires_tensorboard
def test_fit_writes_per_class_curves(config: Config, tmp_path: Path) -> None:
    """10 个类别各一条 AUC 与 F1 曲线，tag 用赛题类别 ID（1~10）。"""
    log_dir = tmp_path / TENSORBOARD_DIR_NAME
    writer = build_writer(log_dir, "true")
    assert writer is not None
    try:
        fit(
            config,
            build_model(config),
            build_loader(config, "train", train=True),
            build_loader(config, "valid"),
            torch.device("cpu"),
            writer=writer,
        )
    finally:
        writer.close()

    scalars = _read_scalars(log_dir)
    for kind in ("auc", "f1"):
        tags = {tag for tag in scalars if tag.startswith(f"valid/{kind}/class_")}
        assert tags == {f"valid/{kind}/class_{index}" for index in range(1, 11)}


@requires_tensorboard
def test_best_curve_is_non_decreasing(config: Config, tmp_path: Path) -> None:
    """best_macro_auc 是历史最优值，只能是台阶上升，不能回落。"""
    log_dir = tmp_path / TENSORBOARD_DIR_NAME
    writer = build_writer(log_dir, "true")
    assert writer is not None
    try:
        fit(
            config,
            build_model(config),
            build_loader(config, "train", train=True),
            build_loader(config, "valid"),
            torch.device("cpu"),
            writer=writer,
        )
    finally:
        writer.close()

    values = _read_scalars(log_dir)["valid/best_macro_auc"]
    assert values == sorted(values)


def test_evaluate_per_class_matches_macro(config: Config) -> None:
    """逐类明细与 macro 指标必须同源：保留两位小数后完全一致。"""
    model = build_model(config)
    loader = build_loader(config, "valid")
    from model.losses import build_loss

    plain = evaluate(config, model, loader, build_loss(config), torch.device("cpu"))
    detailed = evaluate(
        config, model, loader, build_loss(config), torch.device("cpu"), with_per_class=True
    )

    assert "per_class" not in plain
    report = detailed["per_class"]
    assert list(report["class_id"]) == list(range(1, 11))
    # pandas 的 mean 跳过 NaN，正是 macro_auc 的口径；F1 则所有类都参与。
    assert detailed["macro_auc"] == pytest.approx(report["auc"].mean())
    assert detailed["macro_f1"] == pytest.approx(report["f1"].mean())
    assert detailed["macro_auc"] == pytest.approx(plain["macro_auc"])
    assert detailed["macro_f1"] == pytest.approx(plain["macro_f1"])
