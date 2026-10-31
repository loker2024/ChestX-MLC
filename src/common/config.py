"""读取并校验 YAML 配置。

YAML 是嵌套字典，读进来后转成一组 dataclass（每个配置节对应一个类）：
代码里用 `cfg.train.lr` 访问，比 `cfg["train"]["lr"]` 更直观，字段写错时也更早暴露。

校验分两层：
1. 结构与类型：字段名拼错、类型不对，在 load_config 时就报错；
2. 取值范围：例如 lr 必须大于 0，由各个 dataclass 的 __post_init__ 检查。

配置里的相对路径都按“当前工作目录”理解。入口脚本请在仓库根目录执行，
这样 `data/train.csv`、`outputs` 才会落在预期位置。
"""

from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
import types
from typing import Any, Union, get_args, get_origin, get_type_hints

import yaml

from common.constants import DEFAULT_IMAGE_SIZE, NUM_CLASSES


@dataclass(frozen=True)
class SplitConfig:
    """一个数据划分（train / valid / test）的标注 CSV 与图片根目录。"""

    csv: Path
    images: Path


@dataclass(frozen=True)
class DataConfig:
    """三份数据划分。字段名与 YAML 里的 train / valid / test 一一对应。"""

    train: SplitConfig
    valid: SplitConfig
    test: SplitConfig


@dataclass(frozen=True)
class ImageConfig:
    """图像预处理参数，供 data.transforms 使用。"""

    size: int = DEFAULT_IMAGE_SIZE  # letterbox 之后的方形边长
    letterbox: bool = True  # True=等比缩放后补边；False=直接拉伸到 size×size
    normalize: str = "imagenet"  # "imagenet"=减均值除方差；"none"=只把像素缩到 0~1；"xrv"=TxRV 的 [-1024,1024]
    hflip: bool = False  # 训练是否水平翻转；胸片左右有方向含义，默认关闭
    in_channels: int = 3  # 模型输入的通道数；B 组 TxRV DenseNet 使用单通道 1
    xrv_maxval: int = 255  # normalize="xrv" 时的像素上限；8 位图 255、16 位图 65535

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError(f"image.size 必须是正整数，当前为 {self.size}")
        if self.in_channels not in {1, 3}:
            raise ValueError(f"image.in_channels 只能是 1 或 3，当前为 {self.in_channels}")
        if self.normalize not in {"imagenet", "none", "xrv"}:
            raise ValueError(
                f"image.normalize 只能是 'imagenet'、'none' 或 'xrv'，"
                f"当前为 {self.normalize!r}"
            )
        if self.xrv_maxval <= 0:
            raise ValueError(f"image.xrv_maxval 必须是正整数，当前为 {self.xrv_maxval}")
        if self.normalize == "xrv" and self.in_channels != 1:
            raise ValueError(
                f"image.normalize='xrv' 要求 image.in_channels=1（TxRV 单通道输入），"
                f"当前为 {self.in_channels}"
            )
        if self.normalize == "imagenet" and self.in_channels != 3:
            raise ValueError(
                f"image.normalize='imagenet' 要求 image.in_channels=3，"
                f"当前为 {self.in_channels}"
            )


@dataclass(frozen=True)
class ModelConfig:
    """模型结构与权重来源。"""

    backbone: str = "resnet18"  # timm 的模型名
    pretrained: bool = True  # 是否加载预训练权重（timm 为 ImageNet；B 组为 TxRV 医学预训练）
    num_classes: int = NUM_CLASSES  # 输出维度，赛题固定 10
    freeze_backbone: bool = False  # 是否冻结骨干，仅训练分类头
    aggregation: str = "mean"
    frontal_weight: float = 0.7
    arch: str = "timm"  # "timm"=timm 骨干；"xrv_densenet121"=TorchXRayVision DenseNet-121（B 组隔离研究）
    xrv_weights: str = "densenet121-res224-all"  # TxRV 预训练权重名
    xrv_cache_dir: Path | None = None  # 权重缓存目录；None=TxRV 默认 ~/.torchxrayvision/models_data
    xrv_sha256: str | None = None  # 期望的权重文件 SHA-256；None=不校验

    def __post_init__(self) -> None:
        if self.aggregation not in {"mean", "view_weighted"}:
            raise ValueError("model.aggregation 只能是 mean 或 view_weighted")
        if not 0.0 < self.frontal_weight < 1.0:
            raise ValueError("model.frontal_weight 必须在 (0, 1) 内")
        if not self.backbone:
            raise ValueError("model.backbone 不能为空")
        if self.arch not in {"timm", "xrv_densenet121"}:
            raise ValueError(
                f"model.arch 只能是 'timm' 或 'xrv_densenet121'，当前为 {self.arch!r}"
            )
        if self.arch == "xrv_densenet121":
            if self.freeze_backbone:
                raise ValueError(
                    "model.arch='xrv_densenet121' 要求全量微调，不能开启 freeze_backbone"
                )
            if not self.xrv_weights:
                raise ValueError("model.xrv_weights 不能为空")
            if self.xrv_sha256 and (
                len(self.xrv_sha256) != 64
                or any(ch not in "0123456789abcdef" for ch in self.xrv_sha256.lower())
            ):
                raise ValueError("model.xrv_sha256 必须是 64 位十六进制字符串")
        if self.num_classes != NUM_CLASSES:
            raise ValueError(
                f"model.num_classes 必须等于 {NUM_CLASSES}（赛题固定 10 类），"
                f"当前为 {self.num_classes}"
            )


@dataclass(frozen=True)
class TrainConfig:
    """训练超参数。"""

    epochs: int = 25
    batch_studies: int = 16  # 每个 batch 的检查数；图片数由各 study 实际张数决定
    lr: float = 3e-4
    weight_decay: float = 0.01
    pos_weight: bool = False  # 按训练集类别频率给 BCE 的正样本加权；类别极不平衡时打开
    warmup_epochs: int = 2  # 前几个 epoch 线性升温到 lr
    lr_scheduler: str = "warmup"  # "warmup"=预热后保持 lr；"cosine"=预热后余弦衰减到 lr_min
    lr_min: float = 1e-5  # cosine 衰减的终点学习率；"warmup" 模式下不生效
    amp: bool = True  # 混合精度；仅 CUDA 上生效
    grad_clip: float = 5.0  # 梯度范数裁剪上限，<=0 表示不裁剪
    num_workers: int = 8
    early_stop_patience: int = 5  # valid 指标连续多少个 epoch 不提升就停；0 表示关闭早停
    loss: str = "bce"  # "bce"=BCEWithLogitsLoss；"asl"=Asymmetric Loss（自带重加权）
    asl_gamma_pos: float = 0.0  # ASL 正样本聚焦指数；论文默认 0，即不衰减正样本
    asl_gamma_neg: float = 4.0  # ASL 负样本聚焦指数；越大，简单负样本被压制得越狠
    asl_clip: float = 0.05  # ASL 概率移位量 m：负样本概率先推低 m 再取对数

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError(f"train.epochs 必须 >= 1，当前为 {self.epochs}")
        if self.batch_studies < 1:
            raise ValueError(f"train.batch_studies 必须 >= 1，当前为 {self.batch_studies}")
        if self.lr <= 0:
            raise ValueError(f"train.lr 必须 > 0，当前为 {self.lr}")
        if self.weight_decay < 0:
            raise ValueError(f"train.weight_decay 必须 >= 0，当前为 {self.weight_decay}")
        if not 0 <= self.warmup_epochs < self.epochs:
            raise ValueError(
                f"train.warmup_epochs 必须满足 0 <= warmup < epochs，"
                f"当前 warmup={self.warmup_epochs}、epochs={self.epochs}"
            )
        if self.lr_scheduler not in {"warmup", "cosine"}:
            raise ValueError(
                f"train.lr_scheduler 只能是 'warmup' 或 'cosine'，当前为 {self.lr_scheduler!r}"
            )
        if self.lr_min < 0:
            raise ValueError(f"train.lr_min 必须 >= 0，当前为 {self.lr_min}")
        if self.lr_min > self.lr:
            raise ValueError(
                f"train.lr_min 必须 <= train.lr，当前 lr_min={self.lr_min}、lr={self.lr}"
            )
        if self.grad_clip < 0:
            raise ValueError(f"train.grad_clip 必须 >= 0，当前为 {self.grad_clip}")
        if self.num_workers < 0:
            raise ValueError(f"train.num_workers 必须 >= 0，当前为 {self.num_workers}")
        if self.early_stop_patience < 0:
            raise ValueError(
                f"train.early_stop_patience 必须 >= 0，当前为 {self.early_stop_patience}"
            )
        if self.loss not in {"bce", "asl"}:
            raise ValueError(f"train.loss 只能是 'bce' 或 'asl'，当前为 {self.loss!r}")
        if self.asl_gamma_pos < 0 or self.asl_gamma_neg < 0:
            raise ValueError(
                f"train.asl_gamma_pos / asl_gamma_neg 必须 >= 0，"
                f"当前为 {self.asl_gamma_pos} / {self.asl_gamma_neg}"
            )
        if not 0.0 <= self.asl_clip < 1.0:
            raise ValueError(f"train.asl_clip 必须落在 [0, 1)，当前为 {self.asl_clip}")
        if self.loss == "asl" and self.pos_weight:
            raise ValueError(
                "train.loss=asl 时不能再开启 train.pos_weight：ASL 自带正负样本非对称"
                "重加权，叠加 pos_weight 会破坏论文口径，请显式设 pos_weight: false"
            )


@dataclass(frozen=True)
class EvalConfig:
    """验证 / 推理阶段参数。"""

    batch_studies: int = 32  # 验证与推理的 batch 大小（同样按 study 计）
    threshold: float = 0.5  # 计算 macro-F1 的判定阈值；不参与训练，也不影响 AUC

    def __post_init__(self) -> None:
        if self.batch_studies < 1:
            raise ValueError(f"eval.batch_studies 必须 >= 1，当前为 {self.batch_studies}")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"eval.threshold 必须落在 [0, 1]，当前为 {self.threshold}")


@dataclass(frozen=True)
class OutputConfig:
    """训练与推理产物的输出目录。"""

    run_dir: Path = Path("outputs")  # 每次训练在其下新建 <时间戳>_<tag> 子目录


@dataclass(frozen=True)
class Config:
    """整份配置。除 data 外都有默认值，data 必须在 YAML 里写全三个 split。"""

    data: DataConfig
    image: ImageConfig = field(default_factory=ImageConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    seed: int = 42
    device: str = "auto"  # auto / cpu / cuda / cuda:0 / mps

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError(f"seed 必须 >= 0，当前为 {self.seed}")
        if not self.device:
            raise ValueError("device 不能为空，自动选择请写 'auto'")

    def to_dict(self) -> dict[str, Any]:
        """导出成普通 dict（Path 转成字符串），便于写 YAML 或存进 checkpoint。"""
        return _to_plain(asdict(self))


def load_config(path: str | Path = "configs/default.yaml") -> Config:
    """读取 YAML 配置文件并返回校验后的 Config 对象。

    读取失败时 PyYAML 会给出文件与行号；字段写错时的报错来自 _build_dataclass，
    会指出具体是哪一节、哪个字段出了问题。

    配置文件还可以用顶层的 `_base` 键继承另一份配置（详见 `_load_yaml`），
    实验配置因此只需要写「和默认配置不一样的那几行」。
    """
    config_path = Path(path)
    raw = _load_yaml(config_path)

    if not raw:
        raise ValueError(f"配置文件内容为空：{config_path}")
    return _build_dataclass(Config, raw, where="配置根节点")


def config_from_dict(raw: dict[str, Any]) -> Config:
    """把 `Config.to_dict()` 的结果还原成 Config，用于从 checkpoint 续训。

    checkpoint 里存的是 `config.to_dict()`（Path 已转成字符串），结构与 YAML
    读出的字典一致，所以直接走同一套 dataclass 校验，字段缺失或类型不对
    都会在这里报错，而不是训练到一半才发现。
    """
    if not isinstance(raw, dict) or not raw:
        raise ValueError("checkpoint 里的配置为空或不是键值映射，无法恢复训练")
    return _build_dataclass(Config, raw, where="checkpoint 配置根节点")


def _load_yaml(path: Path, seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    """读一份 YAML，并按 `_base` 继承链合并成完整字典。

    `_base` 写的是另一份配置文件的路径，相对当前文件所在的目录解析，例如
    `configs/exp_pos_weight.yaml` 里的 `_base: default.yaml` 就指向
    `configs/default.yaml`。

    合并规则：先递归读出基础配置，再用本文件的键**递归**覆盖它——嵌套的
    `data` / `train` 等小节按字段逐个合并，没有写的字段沿用基础配置；列表和
    标量（例如 `model.backbone`）则整体替换，不做拼接。

    `seen` 记录已经读过的文件，用来发现「A 继承 B、B 又继承 A」这种循环，
    否则递归会一直转下去。
    """
    path = path.resolve()
    if path in seen:
        chain = " -> ".join(str(item) for item in (*seen, path))
        raise ValueError(f"配置文件通过 _base 形成了循环继承：{chain}")

    with path.open(encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    if not isinstance(raw, dict):
        raise TypeError(f"配置文件顶层应该是键值映射，实际是 {type(raw).__name__}：{path}")

    base_name = raw.pop("_base", None)
    if base_name is None:
        return raw
    if not isinstance(base_name, str):
        raise TypeError(f"_base 应该是配置文件路径字符串，实际是 {base_name!r}：{path}")

    base = _load_yaml(path.parent / base_name, (*seen, path))
    return _merge_config(base, raw)


def _merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """把 override 合进 base：嵌套字典逐层合，其余值直接覆盖。"""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def _build_dataclass(cls: type, raw: Any, where: str) -> Any:
    """把 YAML 读出的字典按 dataclass 的字段声明转成实例。

    这里用两遍循环：

    1. 先在 `raw` 的键上检查未知字段，避免 YAML 里把 `epochs` 写成 `epoch`
       却被悄悄忽略（后面训练时才发现用的是默认值，非常难查）；
    2. 再按 dataclass 的字段顺序逐个取值：配置里给了就用配置值，没给就用
       字段默认值；两者都没有说明是必填字段，直接报错。
    """
    if not isinstance(raw, dict):
        raise TypeError(f"{where} 应该是键值映射，实际是 {type(raw).__name__}")

    known = {item.name for item in fields(cls)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"{where} 含未知字段 {unknown}，可用字段为 {sorted(known)}")

    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for item in fields(cls):
        field_where = f"{where}.{item.name}"
        if item.name in raw:
            kwargs[item.name] = _convert(raw[item.name], hints[item.name], field_where)
        elif item.default is not MISSING:
            kwargs[item.name] = item.default
        elif item.default_factory is not MISSING:
            kwargs[item.name] = item.default_factory()
        else:
            raise ValueError(f"{field_where} 是必填字段，但配置里没有写")
    return cls(**kwargs)


def _convert(value: Any, hint: Any, where: str) -> Any:
    """把 YAML 值转换成字段声明的类型；类型不对时给出带字段路径的报错。"""
    # 先处理 Optional[...]：B 组的 xrv_cache_dir / xrv_sha256 允许不写（None），
    # 其余字段仍然走下面的严格类型校验。
    origin = get_origin(hint)
    if origin is Union or origin is types.UnionType:
        options = [arg for arg in get_args(hint) if arg is not type(None)]
        if value is None:
            return None
        if len(options) == 1:
            return _convert(value, options[0], where)
        raise TypeError(f"{where} 不支持的联合类型 {hint}")
    if is_dataclass(hint):
        return _build_dataclass(hint, value, where)
    if hint is Path:
        if not isinstance(value, str) or not value:
            raise TypeError(f"{where} 应该是非空路径字符串，实际是 {value!r}")
        return Path(value)
    if hint is bool:
        if not isinstance(value, bool):
            raise TypeError(f"{where} 应该是 true/false，实际是 {value!r}")
        return value
    if hint is int:
        # bool 是 int 的子类，先排除掉，避免 `epochs: true` 这种写法混过去。
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{where} 应该是整数，实际是 {value!r}")
        return value
    if hint is float:
        # 允许整数写法（如 lr: 1），统一转成 float 便于后续计算。
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{where} 应该是数字，实际是 {value!r}")
        return float(value)
    if hint is str:
        if not isinstance(value, str) or not value:
            raise TypeError(f"{where} 应该是非空字符串，实际是 {value!r}")
        return value
    return value


def _to_plain(value: Any) -> Any:
    """递归把 Path 转成 str，其余类型原样返回（供 Config.to_dict 使用）。"""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    return value
