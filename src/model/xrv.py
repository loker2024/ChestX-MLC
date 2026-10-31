"""B 组隔离适配：TorchXRayVision DenseNet-121 医学预训练骨干。

本模块只在 `model.arch="xrv_densenet121"` 时被 `model.network.build_model` 惰性
导入，默认的 timm 链路（含赛题提交）完全不经过这里。B 组仅作研究对照，不进入
提交、融合或候选选择——TxRV 权重在外部医学影像数据集上预训练，按赛题规则第 1
条不能用于提交方案。

与 timm 骨干的差异都收敛在本模块：

1. 输入是单通道 224×224、值域 [-1024, 1024]（TxRV 口径），由
   `data.transforms` 的 "xrv" 归一化负责；
2. 官方权重固定输出 18 类病理，加载后把分类头替换成赛题的 10 类；
3. TxRV 的 `forward` 在 `op_threshs` 非空时会做 sigmoid + 阈值归一化，这里改为
   `features2 -> classifier` 直接拿原始 logits，并显式清空 `op_threshs`，
   保证「先聚合 logits 再算 BCE」的口径不被改写；
4. 预训练权重加载失败必须直接报错，不回退随机初始化：B 组的结论依赖「权重确实
   加载成功」这一前提，静默回退会让对照实验不可解释。只有配置显式写
   `pretrained: false` 时才构建随机初始化结构（续训重建 / 测试用）。

权重文件的 SHA-256 在校验后记录到模型属性上，由 `run_train` 打印进训练日志；
若配置了 `model.xrv_sha256`，实际文件哈希不匹配会直接报错。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import ModuleType

import torch
from torch import nn

from common.config import Config

# TxRV 官方覆盖 5 个数据集的权重名；B 组固定用它（对照 RUN-006 的 ImageNet 权重）。
DEFAULT_WEIGHTS = "densenet121-res224-all"
# TxRV 所有 DenseNet 权重的原生输入分辨率，与 docs/train_plan12.md 的 224 一致。
XRV_INPUT_RESOLUTION = 224


def file_sha256(path: str | Path) -> str:
    """分块计算文件 SHA-256；权重文件约 28MB，没必要一次读进内存。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class XrvDenseNetClassifier(nn.Module):
    """TxRV DenseNet-121 骨干 + 10 类分类头，输出每张图的原始 logits。

    不直接继承 `xrv.models.DenseNet`，而是把已加载权重的 backbone 组合进来：
    TxRV 的 `DenseNet.forward` 带有 `op_threshs` 分支，组合式封装能明确绕开它，
    也能把权重路径 / SHA-256 / 是否加载成功等研究元数据挂在模型上，交给
    `run_train` 写进日志和 checkpoint metadata。
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        *,
        weights_name: str | None = None,
        weights_path: Path | None = None,
        weights_sha256: str | None = None,
        pretrained_loaded: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.weights_name = weights_name
        self.weights_path = weights_path
        self.weights_sha256 = weights_sha256
        self.pretrained_loaded = pretrained_loaded
        # 供 run_train / run_eval 打印，形状与 TxRV 加载后的 input_resolution 一致。
        self.input_resolution = XRV_INPUT_RESOLUTION

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """[N, 1, H, W] 值域 [-1024,1024] -> [N, 10] 原始 logits。

        `features2` 内部按 TxRV 的 input_resolution 做保比例插值（输入已是 224 时
        是 no-op），随后接替换过的 10 类分类头；不做 sigmoid。
        """
        return self.backbone.classifier(self.backbone.features2(images))

    def get_classifier(self) -> nn.Module:
        """与 timm 模型同名接口，便于通用代码取分类头。"""
        return self.backbone.classifier

    def pretrained_info(self) -> dict[str, str | bool]:
        """权重来源与校验信息，供入口脚本打印 / 写日志。"""
        return {
            "arch": "xrv_densenet121",
            "weights_name": self.weights_name or "",
            "weights_path": str(self.weights_path) if self.weights_path else "",
            "weights_sha256": self.weights_sha256 or "",
            "pretrained_loaded": self.pretrained_loaded,
        }


def _import_torchxrayvision_models() -> ModuleType:
    """导入 `torchxrayvision.models`，绕开 TxRV 1.5.7 的顶层包名冲突。

    TxRV 1.5.7 的 `torchxrayvision/__init__.py` 会导入 `baseline_models`，其中
    jfhealthcare 的 VGG 代码写死 `from model.utils import get_norm`；本仓库的顶层
    包名恰好也叫 `model`（`src/model/`），于是 `import torchxrayvision` 会抛
    `ModuleNotFoundError: No module named 'model.utils'`。B 组只需要
    `torchxrayvision.models.DenseNet`，baseline_models 与本实验无关，因此先注册
    一个空占位模块跳过它的导入。副作用仅限于 `torchxrayvision.baseline_models`
    这个 B 组不使用的子模块。
    """
    import importlib.util
    import sys
    import types

    stub_name = "torchxrayvision.baseline_models"
    if stub_name not in sys.modules and importlib.util.find_spec("torchxrayvision") is not None:
        sys.modules[stub_name] = types.ModuleType(stub_name)
    from torchxrayvision import models as xrv_models

    return xrv_models


def build_xrv_densenet121(cfg: Config) -> XrvDenseNetClassifier:
    """按配置构建 TxRV DenseNet-121，加载 18 类权重后换成 10 类头。

    `cfg.model.pretrained=True`（B 组训练配置）时，权重下载 / 读取 / 哈希校验任何
    一步失败都抛 `RuntimeError`，绝不回退随机初始化；`pretrained=False` 只用于
    checkpoint 重建（推理入口会强制关掉 pretrained）和合成数据测试。
    """
    try:
        xrv_models = _import_torchxrayvision_models()
    except ImportError as error:
        raise RuntimeError(
            "B 组需要 torchxrayvision==1.5.7（pip install torchxrayvision==1.5.7）；"
            f"当前环境导入失败：{error}"
        ) from error

    model_cfg = cfg.model
    weights_name = model_cfg.xrv_weights
    cache_dir = str(model_cfg.xrv_cache_dir) if model_cfg.xrv_cache_dir else None
    if not model_cfg.pretrained:
        backbone = xrv_models.DenseNet(
            weights=None, num_classes=model_cfg.num_classes, in_channels=1
        )
        return XrvDenseNetClassifier(backbone, model_cfg.num_classes)

    try:
        # TxRV 会在缓存缺失时自行下载；这里把任何异常（网络、文件损坏、torch.load
        # 版本问题）统一包成 RuntimeError，并保留原始异常链便于排查。
        backbone = xrv_models.DenseNet(weights=weights_name, cache_dir=cache_dir)
    except Exception as error:  # noqa: BLE001 - 需要把各类加载异常统一成明确报错
        raise RuntimeError(
            f"TxRV 预训练权重 {weights_name!r} 加载失败：{error}。"
            "B 组要求权重确实加载成功，这里不会回退到随机初始化；"
            "请检查权重缓存目录（model.xrv_cache_dir）、网络连接或权重文件完整性。"
        ) from error

    weights_path = Path(backbone.weights_filename_local)
    weights_sha256 = file_sha256(weights_path)
    expected_sha256 = (model_cfg.xrv_sha256 or "").lower()
    if expected_sha256 and weights_sha256 != expected_sha256:
        raise RuntimeError(
            f"TxRV 权重 SHA-256 不匹配：期望 {expected_sha256}，"
            f"实际 {weights_sha256}（{weights_path}）。"
            "请删除损坏的缓存文件后重新下载，或修正 model.xrv_sha256。"
        )

    # 官方权重是 18 类头，换成赛题 10 类；新头随机初始化，随后全量微调。
    num_features = backbone.classifier.in_features
    backbone.classifier = nn.Linear(num_features, model_cfg.num_classes)
    # 清空 TxRV 的阈值归一化，确保 forward 走到「原始 logits」分支。
    backbone.op_threshs = None
    backbone.apply_sigmoid = False

    return XrvDenseNetClassifier(
        backbone,
        model_cfg.num_classes,
        weights_name=weights_name,
        weights_path=weights_path,
        weights_sha256=weights_sha256,
        pretrained_loaded=True,
    )
