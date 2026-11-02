"""外部权重导入：格式转换、张量不变性、口径校验与覆盖保护。

用 ResNet-18 而不是 DenseNet-121 做测试模型：结构规则相同（都是 timm 分类模型），
`pretrained=False` 不联网，单测耗时可控。真实 T 系列的 densenet121 在远端执行。
"""

import hashlib

import pytest
import torch

from common.checkpoint import load_checkpoint
from common.config import load_config
from import_checkpoint import main
from model.network import build_model

# 配置里只有 data 是根节点必填字段，其余走 dataclass 默认值；
# 这里显式写出 model/image，是为了让 resnet18 + 384 的口径与外部权重声明一致。
CONFIG_YAML = """\
data:
  train: {csv: unused.csv, images: unused}
  valid: {csv: unused.csv, images: unused}
  test: {csv: unused.csv, images: unused}
image:
  size: 384
model:
  backbone: resnet18
  pretrained: false
  num_classes: 10
"""


def _config_path(tmp_path):
    """写一份只含模型与尺寸的最小配置；其余字段走 dataclass 默认值。"""
    path = tmp_path / "config.yaml"
    if not path.exists():
        path.write_text(CONFIG_YAML, encoding="utf-8")
    return path


def _state_dict(tmp_path):
    """造一份 resnet18 / 10 类的权重，模拟外部训练产出的张量。"""
    return build_model(load_config(_config_path(tmp_path))).state_dict()


def test_import_converts_external_checkpoint(tmp_path):
    """外部 `state_dict` 结构 -> 仓库格式：张量逐个相等、配置与来源信息都留下。"""
    source = tmp_path / "R1_best_model.pth"
    state = _state_dict(tmp_path)
    torch.save(
        {
            "architecture": "resnet18",
            "model_backend": "torchvision",
            "num_classes": 10,
            "state_dict": state,
            "metadata": {"experiment_id": "fake", "input_size": 384},
        },
        source,
    )
    output = tmp_path / "R1_best_loker.pt"

    main([
        "--source", str(source),
        "--config", str(_config_path(tmp_path)),
        "--out", str(output),
        "--seed", "42",
    ])

    converted = load_checkpoint(output, map_location="cpu")
    assert converted["config"] is not None
    assert converted["seed"] == 42
    expected_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    assert converted["metadata"]["source_sha256"] == expected_sha
    assert converted["metadata"]["source_architecture"] == "resnet18"
    assert set(converted["model"]) == set(state)
    for name, tensor in state.items():
        assert torch.equal(converted["model"][name], tensor), name


def test_import_rejects_backbone_mismatch(tmp_path):
    """权重声明的骨干与配置不符时必须报错，不能靠 load_state_dict 静默兜底。"""
    source = tmp_path / "wrong.pth"
    torch.save(
        {"architecture": "densenet121", "state_dict": _state_dict(tmp_path), "metadata": {}},
        source,
    )

    with pytest.raises(ValueError, match="骨干不匹配"):
        main([
            "--source", str(source),
            "--config", str(_config_path(tmp_path)),
            "--out", str(tmp_path / "out.pt"),
        ])


def test_import_rejects_wrong_sha256(tmp_path):
    """给了 --expect-sha256 就必须一致，防止拿错权重版本。"""
    source = tmp_path / "R1.pth"
    torch.save({"architecture": "resnet18", "state_dict": _state_dict(tmp_path)}, source)

    with pytest.raises(ValueError, match="sha256 不符"):
        main([
            "--source", str(source),
            "--config", str(_config_path(tmp_path)),
            "--out", str(tmp_path / "out.pt"),
            "--expect-sha256", "0" * 64,
        ])


def test_import_refuses_to_overwrite_without_force(tmp_path):
    """输出已存在时默认拒绝，避免误覆盖已归档的转换产物。"""
    source = tmp_path / "R1.pth"
    torch.save({"architecture": "resnet18", "state_dict": _state_dict(tmp_path)}, source)
    output = tmp_path / "out.pt"
    output.write_bytes(b"placeholder")

    with pytest.raises(FileExistsError):
        main([
            "--source", str(source),
            "--config", str(_config_path(tmp_path)),
            "--out", str(output),
        ])
