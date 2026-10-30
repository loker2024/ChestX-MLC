"""配置读取与校验：默认值、未知字段、类型/范围错误、Path 序列化。"""

from pathlib import Path

import pytest

from common.config import TrainConfig, config_from_dict, load_config

BASE = """data:
  train: {csv: data/train.csv, images: data/images/train}
  valid: {csv: data/valid.csv, images: data/images/valid}
  test: {csv: data/test.csv, images: data/images/test}
image:
  size: 32
model:
  pretrained: false
train:
  epochs: 1
  warmup_epochs: 0
  batch_studies: 2
  num_workers: 0
  early_stop_patience: 0
eval:
  batch_studies: 2
output:
  run_dir: outputs
seed: 42
device: cpu
"""


def _write(tmp_path: Path, text: str = BASE, name: str = "config.yaml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_default_config_loads():
    config = load_config(Path(__file__).resolve().parents[1] / "configs/default.yaml")

    assert config.model.backbone == "resnet18"
    assert config.model.freeze_backbone is False
    assert config.image.size == 320
    assert config.train.pos_weight is True
    assert config.train.lr_scheduler == "cosine"
    assert config.train.lr_min == pytest.approx(1e-5)
    assert config.eval.threshold == 0.5
    assert config.output.run_dir == Path("outputs")
    assert "checkpoint_dir" not in config.to_dict()["output"]


def test_to_dict_converts_paths_to_strings(tmp_path):
    config = load_config(_write(tmp_path))

    exported = config.to_dict()

    assert exported["data"]["train"]["csv"] == "data/train.csv"
    assert exported["output"]["run_dir"] == "outputs"
    assert isinstance(exported["seed"], int)


def test_config_from_dict_roundtrip(tmp_path):
    """checkpoint 里存的 to_dict() 必须能原样还原，否则续训会读错超参。"""
    config = load_config(_write(tmp_path))

    restored = config_from_dict(config.to_dict())

    assert restored == config
    assert isinstance(restored.data.train.csv, Path)


def test_config_from_dict_rejects_empty():
    with pytest.raises(ValueError, match="无法恢复"):
        config_from_dict({})


def test_unknown_top_level_field_raises(tmp_path):
    with pytest.raises(ValueError, match="未知字段"):
        load_config(_write(tmp_path, BASE + "\nunknown: 1\n"))


def test_unknown_nested_field_raises(tmp_path):
    text = BASE.replace(
        "model:\n  pretrained: false", "model:\n  pretrained: false\n  typo: 1"
    )
    with pytest.raises(ValueError, match="未知字段"):
        load_config(_write(tmp_path, text))


def test_wrong_type_raises(tmp_path):
    text = BASE.replace("  epochs: 1", "  epochs: many")
    with pytest.raises(TypeError, match="整数"):
        load_config(_write(tmp_path, text))


def test_warmup_must_be_less_than_epochs(tmp_path):
    text = BASE.replace("  warmup_epochs: 0", "  warmup_epochs: 1")
    with pytest.raises(ValueError, match="warmup"):
        load_config(_write(tmp_path, text))


def test_unknown_lr_scheduler_raises(tmp_path):
    text = BASE.replace("  warmup_epochs: 0", "  warmup_epochs: 0\n  lr_scheduler: step")
    with pytest.raises(ValueError, match="lr_scheduler"):
        load_config(_write(tmp_path, text))


def test_lr_min_must_not_exceed_lr(tmp_path):
    text = BASE.replace("  warmup_epochs: 0", "  warmup_epochs: 0\n  lr_min: 1.0")
    with pytest.raises(ValueError, match="lr_min"):
        load_config(_write(tmp_path, text))


def test_cosine_experiment_config_loads():
    """RUN-003-A 配置只改调度：25 轮、warmup 2、余弦到 1e-5、num_workers 16。"""
    config = load_config(Path(__file__).resolve().parents[1] / "configs/exp_cosine.yaml")

    assert config.train.lr_scheduler == "cosine"
    assert config.train.lr_min == pytest.approx(1e-5)
    assert config.train.warmup_epochs == 2
    assert config.train.epochs == 25
    assert config.train.early_stop_patience == 5
    assert config.train.pos_weight is False
    assert config.train.num_workers == 16


def test_run003b_experiment_config_loads():
    """RUN-003-B 固定当前默认口径：ResNet-18 + cosine + pos_weight，seed 42。"""
    config = load_config(Path(__file__).resolve().parents[1] / "configs/exp_run003b.yaml")

    assert config.model.backbone == "resnet18"
    assert config.train.lr_scheduler == "cosine"
    assert config.train.lr_min == pytest.approx(1e-5)
    assert config.train.warmup_epochs == 2
    assert config.train.pos_weight is True
    assert config.train.epochs == 25
    assert config.train.early_stop_patience == 5
    assert config.train.num_workers == 16
    assert config.seed == 42


def test_freeze_head_experiment_config_only_enables_freezing():
    configs = Path(__file__).resolve().parents[1] / "configs"
    base = load_config(configs / "default.yaml").to_dict()
    config = load_config(configs / "exp_freeze_head.yaml").to_dict()

    assert config["model"]["freeze_backbone"] is True
    base["model"]["freeze_backbone"] = True
    assert config == base


def test_run006a_experiment_config_loads():
    """RUN-006-A 只把 RUN-006 的 seed 换成 2026，其余口径完全一致。"""
    configs = Path(__file__).resolve().parents[1] / "configs"
    run006 = load_config(configs / "exp_densenet121.yaml")
    config = load_config(configs / "exp_run006a.yaml")

    assert config.model.backbone == "densenet121.ra_in1k"
    assert config.seed == 2026
    assert config.model.aggregation == "mean"

    # 与 RUN-006 逐字段比对：除 seed 外不允许有任何差异，否则就不是单变量对照。
    run006_dict = run006.to_dict()
    config_dict = config.to_dict()
    differences = {
        key: (run006_dict[key], config_dict[key])
        for key in run006_dict
        if run006_dict[key] != config_dict[key]
    }
    assert differences == {"seed": (42, 2026)}


def test_run007a_experiment_config_loads():
    """RUN-007-A 只把 RUN-007 的 seed 换成 2026，其余口径（含加权聚合）完全一致。"""
    configs = Path(__file__).resolve().parents[1] / "configs"
    run007 = load_config(configs / "exp_run007.yaml")
    config = load_config(configs / "exp_run007a.yaml")

    assert config.model.backbone == "densenet121.ra_in1k"
    assert config.model.aggregation == "view_weighted"
    assert config.model.frontal_weight == pytest.approx(0.7)
    assert config.seed == 2026

    # 与 RUN-007 逐字段比对：除 seed 外不允许有任何差异，否则就不是单变量对照。
    run007_dict = run007.to_dict()
    config_dict = config.to_dict()
    differences = {
        key: (run007_dict[key], config_dict[key])
        for key in run007_dict
        if run007_dict[key] != config_dict[key]
    }
    assert differences == {"seed": (42, 2026)}


def test_run007b_experiment_config_loads():
    """RUN-007-B 只在 RUN-007-A 上改 image.size=384、train.epochs=6 两处。"""
    configs = Path(__file__).resolve().parents[1] / "configs"
    run007a = load_config(configs / "exp_run007a.yaml")
    config = load_config(configs / "exp_run007b.yaml")

    assert config.model.backbone == "densenet121.ra_in1k"
    assert config.model.aggregation == "view_weighted"
    assert config.model.frontal_weight == pytest.approx(0.7)
    assert config.seed == 2026
    assert config.image.size == 384
    assert config.train.epochs == 6

    # 与 RUN-007-A 逐字段比对：只允许 image.size 与 train.epochs 两处差异，
    # 否则就不是「只提分辨率 + 缩短周期」的单变量探针。
    base_dict = run007a.to_dict()
    config_dict = config.to_dict()
    image_diff = {
        key: (base_dict["image"][key], config_dict["image"][key])
        for key in base_dict["image"]
        if base_dict["image"][key] != config_dict["image"][key]
    }
    train_diff = {
        key: (base_dict["train"][key], config_dict["train"][key])
        for key in base_dict["train"]
        if base_dict["train"][key] != config_dict["train"][key]
    }
    assert image_diff == {"size": (320, 384)}
    assert train_diff == {"epochs": (25, 6)}
    assert {
        key
        for key in base_dict
        if key not in {"image", "train"} and base_dict[key] != config_dict[key]
    } == set()


RUN008_BACKBONES = {
    "a": "efficientnet_b2.ra_in1k",
    "b": "efficientnetv2_rw_s.ra2_in1k",
    "c": "convnext_tiny.fb_in1k",
    "d": "convnextv2_nano.fcmae_ft_in1k",
}


@pytest.mark.parametrize(("key", "backbone"), sorted(RUN008_BACKBONES.items()))
def test_run008_experiment_configs_load(key, backbone):
    """RUN008-A/B/C/D 继承 RUN-007-A，只覆盖 model.backbone，其余字段逐项一致。"""
    configs = Path(__file__).resolve().parents[1] / "configs"
    run007a = load_config(configs / "exp_run007a.yaml").to_dict()
    config = load_config(configs / f"exp_run008{key}.yaml").to_dict()

    assert config["model"].pop("backbone") == backbone
    assert run007a["model"].pop("backbone") == "densenet121.ra_in1k"
    # 四组统一缩短为 6 轮、关闭早停（RUN-007-A 仍是 25 轮上限、耐心 5）。
    assert config["train"].pop("epochs") == 6
    assert config["train"].pop("early_stop_patience") == 0
    assert run007a["train"].pop("epochs") == 25
    assert run007a["train"].pop("early_stop_patience") == 5
    # 除骨干与训练预算外，其余配置必须与 RUN-007-A 完全相同。
    assert config == run007a

    config = load_config(configs / f"exp_run008{key}.yaml")
    assert config.model.pretrained is True
    assert config.model.num_classes == 10
    assert config.model.aggregation == "view_weighted"
    assert config.model.frontal_weight == pytest.approx(0.7)
    assert config.image.size == 320
    assert config.train.epochs == 6
    assert config.train.early_stop_patience == 0
    assert config.train.batch_studies == 16
    assert config.seed == 2026


def test_a001_experiment_config_loads():
    """A-001 是 RUN008-C 的完整展开重跑：不继承，只差训练预算与 seed 三处。"""
    configs = Path(__file__).resolve().parents[1] / "configs"
    run008c = load_config(configs / "exp_run008c.yaml").to_dict()
    config = load_config(configs / "exp_A001.yaml").to_dict()

    assert config["train"].pop("epochs") == 8
    assert config["train"].pop("early_stop_patience") == 2
    run008c["train"].pop("epochs")
    run008c["train"].pop("early_stop_patience")
    assert config["seed"] == 42
    run008c["seed"] = 42
    # 除训练预算与 seed 外，A-001 与 RUN008-C 逐字段一致。
    assert config == run008c

    text = (configs / "exp_A001.yaml").read_text(encoding="utf-8")
    # 完整展开：没有 `_base` 顶层键（注释里提到 `_base` 不算）。
    assert not any(line.strip().startswith("_base:") for line in text.splitlines())

    cfg = load_config(configs / "exp_A001.yaml")
    assert cfg.image.size == 320
    assert cfg.model.backbone == "convnext_tiny.fb_in1k"
    assert cfg.model.pretrained is True
    assert cfg.model.aggregation == "view_weighted"
    assert cfg.model.frontal_weight == pytest.approx(0.7)
    assert cfg.train.epochs == 8
    assert cfg.train.early_stop_patience == 2
    assert cfg.train.lr_scheduler == "cosine"
    assert cfg.train.pos_weight is True
    assert cfg.seed == 42

def test_a001b_experiment_config_loads():
    """A-001-B 只把 A-001 的损失从 BCE(+pos_weight) 换成 ASL，其余逐字段一致。"""
    configs = Path(__file__).resolve().parents[1] / "configs"
    a001 = load_config(configs / "exp_A001.yaml").to_dict()
    config = load_config(configs / "exp_A001b.yaml").to_dict()

    # 只覆盖损失方案相关的五个字段，其余沿用 A-001。
    assert config["train"].pop("loss") == "asl"
    assert config["train"].pop("pos_weight") is False
    assert config["train"].pop("asl_gamma_pos") == 0.0
    assert config["train"].pop("asl_gamma_neg") == 4.0
    assert config["train"].pop("asl_clip") == 0.05
    a001["train"].pop("loss")
    a001["train"].pop("pos_weight")
    a001["train"].pop("asl_gamma_pos")
    a001["train"].pop("asl_gamma_neg")
    a001["train"].pop("asl_clip")
    assert config == a001

    text = (configs / "exp_A001b.yaml").read_text(encoding="utf-8")
    assert any(line.strip() == "_base: exp_A001.yaml" for line in text.splitlines())

    cfg = load_config(configs / "exp_A001b.yaml")
    assert cfg.model.backbone == "convnext_tiny.fb_in1k"
    assert cfg.image.size == 320
    assert cfg.train.epochs == 8
    assert cfg.train.lr_scheduler == "cosine"
    assert cfg.train.pos_weight is False
    assert cfg.train.loss == "asl"
    assert cfg.seed == 42


def test_a001c_experiment_config_loads():
    """A-001-C 只把 A-001 的骨干冻结（仅训练分类头），其余逐字段一致。"""
    configs = Path(__file__).resolve().parents[1] / "configs"
    a001 = load_config(configs / "exp_A001.yaml").to_dict()
    config = load_config(configs / "exp_A001c.yaml").to_dict()

    # 唯一差异：冻结骨干；其余字段（含损失、调度、轮数、seed）沿用 A-001。
    assert config["model"].pop("freeze_backbone") is True
    assert a001["model"].pop("freeze_backbone") is False
    assert config == a001

    text = (configs / "exp_A001c.yaml").read_text(encoding="utf-8")
    assert any(line.strip() == "_base: exp_A001.yaml" for line in text.splitlines())

    cfg = load_config(configs / "exp_A001c.yaml")
    assert cfg.model.backbone == "convnext_tiny.fb_in1k"
    assert cfg.model.pretrained is True
    assert cfg.model.freeze_backbone is True
    assert cfg.image.size == 320
    assert cfg.train.epochs == 8
    assert cfg.train.early_stop_patience == 2
    assert cfg.train.lr_scheduler == "cosine"
    assert cfg.train.pos_weight is True
    assert cfg.train.loss == "bce"
    assert cfg.seed == 42


def test_a001d_experiment_config_loads():
    """A-001-D 只取消 A-001 的正侧位加权（view_weighted → mean），其余逐字段一致。"""
    configs = Path(__file__).resolve().parents[1] / "configs"
    a001 = load_config(configs / "exp_A001.yaml").to_dict()
    config = load_config(configs / "exp_A001d.yaml").to_dict()

    # 唯一差异：聚合方式改为全图等权均值；`frontal_weight` 保留原值但不参与计算。
    assert config["model"].pop("aggregation") == "mean"
    assert a001["model"].pop("aggregation") == "view_weighted"
    assert config == a001

    text = (configs / "exp_A001d.yaml").read_text(encoding="utf-8")
    assert any(line.strip() == "_base: exp_A001.yaml" for line in text.splitlines())

    cfg = load_config(configs / "exp_A001d.yaml")
    assert cfg.model.backbone == "convnext_tiny.fb_in1k"
    assert cfg.model.pretrained is True
    assert cfg.model.aggregation == "mean"
    assert cfg.model.frontal_weight == pytest.approx(0.7)
    assert cfg.image.size == 320
    assert cfg.train.epochs == 8
    assert cfg.train.early_stop_patience == 2
    assert cfg.train.lr_scheduler == "cosine"
    assert cfg.train.pos_weight is True
    assert cfg.train.loss == "bce"
    assert cfg.seed == 42


def test_a002_experiment_config_loads():
    """A-002 只换轻量骨干与训练预算：除 backbone / epochs / patience 外逐字段同 A-001。"""
    configs = Path(__file__).resolve().parents[1] / "configs"
    a001 = load_config(configs / "exp_A001.yaml").to_dict()
    config = load_config(configs / "exp_A002.yaml").to_dict()

    assert config["model"].pop("backbone") == "mobilenetv3_small_050.lamb_in1k"
    assert a001["model"].pop("backbone") == "convnext_tiny.fb_in1k"
    assert config["train"].pop("epochs") == 25
    assert a001["train"].pop("epochs") == 8
    assert config["train"].pop("early_stop_patience") == 5
    assert a001["train"].pop("early_stop_patience") == 2
    assert config == a001

    text = (configs / "exp_A002.yaml").read_text(encoding="utf-8")
    assert any(line.strip() == "_base: exp_A001.yaml" for line in text.splitlines())

    cfg = load_config(configs / "exp_A002.yaml")
    assert cfg.model.backbone == "mobilenetv3_small_050.lamb_in1k"
    assert cfg.model.pretrained is True
    assert cfg.model.aggregation == "view_weighted"
    assert cfg.model.frontal_weight == pytest.approx(0.7)
    assert cfg.image.size == 320
    assert cfg.image.hflip is False
    assert cfg.train.epochs == 25
    assert cfg.train.early_stop_patience == 5
    assert cfg.train.warmup_epochs == 2
    assert cfg.train.batch_studies == 16
    assert cfg.train.lr == pytest.approx(0.0003)
    assert cfg.train.lr_scheduler == "cosine"
    assert cfg.train.pos_weight is True
    assert cfg.seed == 42


# A-001-G1…G5 五组 ASL 超参对照：(gamma_pos, gamma_neg, clip)，与分组表一致。
_A001_ASL_GROUPS = {
    "G1": (0.0, 2.0, 0.05),
    "G2": (0.0, 6.0, 0.05),
    "G3": (0.0, 4.0, 0.0),
    "G4": (0.0, 4.0, 0.10),
    "G5": (1.0, 4.0, 0.05),
}


@pytest.mark.parametrize("group", sorted(_A001_ASL_GROUPS))
def test_a001g_experiment_config_loads(group):
    """A-001-G1…G5 只调 ASL 的 gamma / clip，其余口径与 A-001 逐字段一致。"""
    configs = Path(__file__).resolve().parents[1] / "configs"
    gamma_pos, gamma_neg, clip = _A001_ASL_GROUPS[group]
    path = configs / f"exp_A001_{group}.yaml"
    a001 = load_config(configs / "exp_A001.yaml").to_dict()
    config = load_config(path).to_dict()

    # 五组都是「A-001 换 ASL 损失」，只在 gamma_pos / gamma_neg / clip 上分组不同。
    assert config["train"].pop("loss") == "asl"
    assert config["train"].pop("pos_weight") is False
    assert config["train"].pop("asl_gamma_pos") == pytest.approx(gamma_pos)
    assert config["train"].pop("asl_gamma_neg") == pytest.approx(gamma_neg)
    assert config["train"].pop("asl_clip") == pytest.approx(clip)
    for key in ("loss", "pos_weight", "asl_gamma_pos", "asl_gamma_neg", "asl_clip"):
        a001["train"].pop(key)
    assert config == a001

    text = path.read_text(encoding="utf-8")
    assert any(line.strip() == "_base: exp_A001.yaml" for line in text.splitlines())

    cfg = load_config(path)
    assert cfg.model.backbone == "convnext_tiny.fb_in1k"
    assert cfg.model.pretrained is True
    assert cfg.model.aggregation == "view_weighted"
    assert cfg.image.size == 320
    assert cfg.train.epochs == 8
    assert cfg.train.early_stop_patience == 2
    assert cfg.train.lr_scheduler == "cosine"
    assert cfg.train.loss == "asl"
    assert cfg.train.pos_weight is False
    assert cfg.seed == 42


def test_asl_loss_rejects_pos_weight_combination():
    """ASL 自带非对称重加权，再叠加 pos_weight 会破坏口径，配置阶段直接报错。"""
    with pytest.raises(ValueError, match="pos_weight"):
        TrainConfig(loss="asl", pos_weight=True)


def test_invalid_loss_and_asl_hyperparameters_are_rejected():
    with pytest.raises(ValueError, match="train.loss"):
        TrainConfig(loss="focal")
    with pytest.raises(ValueError, match="asl_gamma"):
        TrainConfig(loss="asl", asl_gamma_neg=-1.0)
    with pytest.raises(ValueError, match="asl_clip"):
        TrainConfig(loss="asl", asl_clip=1.0)


def test_eval_threshold_range_is_checked(tmp_path):
    text = BASE.replace(
        "eval:\n  batch_studies: 2", "eval:\n  batch_studies: 2\n  threshold: 1.5"
    )
    with pytest.raises(ValueError, match="threshold"):
        load_config(_write(tmp_path, text))


def test_missing_data_section_raises(tmp_path):
    with pytest.raises(ValueError, match="必填"):
        load_config(_write(tmp_path, "seed: 1\n"))


def test_empty_config_raises(tmp_path):
    with pytest.raises(ValueError, match="为空"):
        load_config(_write(tmp_path, ""))


def test_base_config_is_merged(tmp_path):
    """`_base` 指向的父配置先读入，子配置里的同名字段再覆盖上去。"""
    _write(tmp_path, BASE, name="base.yaml")
    child = tmp_path / "child.yaml"
    child.write_text("_base: base.yaml\ntrain:\n  epochs: 3\n", encoding="utf-8")

    config = load_config(child)

    assert config.train.epochs == 3          # 子配置覆盖
    assert config.train.batch_studies == 2   # 父配置延续
    assert config.image.size == 32           # 子配置没写的段整段继承


def test_base_cycle_raises(tmp_path):
    (tmp_path / "a.yaml").write_text("_base: b.yaml\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("_base: a.yaml\n", encoding="utf-8")

    with pytest.raises(ValueError, match="循环"):
        load_config(tmp_path / "a.yaml")


def test_base_must_be_a_string(tmp_path):
    child = tmp_path / "child.yaml"
    child.write_text("_base: [base.yaml]\n", encoding="utf-8")

    with pytest.raises(TypeError, match="_base"):
        load_config(child)


def test_base_file_missing_raises(tmp_path):
    child = tmp_path / "child.yaml"
    child.write_text("_base: no_such_file.yaml\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        load_config(child)


def test_experiment_configs_extend_default():
    """随代码发布的实验配置必须一直能读通，否则复现命令一跑就断。"""
    root = Path(__file__).resolve().parents[1] / "configs"
    short = load_config(root / "exp_short.yaml")
    weighted = load_config(root / "exp_short_pos_weight.yaml")

    assert short.train.epochs == 6
    assert short.train.pos_weight is False
    assert weighted.train.pos_weight is True
    assert weighted.train.epochs == 6
    assert weighted.image.size == 320
