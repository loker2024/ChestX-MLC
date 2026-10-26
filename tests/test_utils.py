"""通用工具：种子、设备选择、运行目录与 tag 清洗。"""

import random

import numpy as np
import pytest
import torch

from common.utils import make_run_dir, resolve_device, seed_everything


def test_seed_everything_is_reproducible():
    seed_everything(1234)
    first = (random.random(), float(np.random.rand()), torch.rand(3))
    seed_everything(1234)
    second = (random.random(), float(np.random.rand()), torch.rand(3))

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])


def test_resolve_device_cpu():
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_without_cuda_raises():
    if torch.cuda.is_available():
        pytest.skip("当前环境有 CUDA")
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_device("cuda")


def test_resolve_device_rejects_out_of_range_cuda_index():
    if not torch.cuda.is_available():
        pytest.skip("需要 CUDA")
    with pytest.raises(RuntimeError, match="CUDA 设备"):
        resolve_device("cuda:999")


def test_make_run_dir_sanitizes_tag(tmp_path):
    root = tmp_path / "outputs"

    run_dir = make_run_dir(root, "resnet18/../../evil tag")

    assert run_dir.parent == root
    assert run_dir.is_dir()
    assert "/" not in run_dir.name
    assert " " not in run_dir.name


def test_make_run_dir_empty_tag_becomes_run(tmp_path):
    run_dir = make_run_dir(tmp_path / "outputs", "///")

    assert run_dir.name.endswith("_run")
