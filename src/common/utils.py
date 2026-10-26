"""通用工具：随机种子、设备选择、运行目录。

这三个函数都不依赖赛题业务逻辑，训练和推理入口都会用到，所以放在 common 里。
"""

import random
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """固定 Python / NumPy / PyTorch 的随机种子，让同一次实验尽量可复现。

    需要说明的是：固定种子只能保证“随机数序列一致”。GPU 上有些算子会并行归约，
    累加顺序不同仍可能带来极小的数值差异，因此不要把“逐位一致”当成硬性目标；
    对比实验时把 seed 记进日志和 checkpoint 即可。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # CUDA 上要额外给所有 GPU 设种子（torch.manual_seed 只覆盖默认的一处）。
    torch.cuda.manual_seed_all(seed)

    # benchmark=True 会让 cuDNN 按速度自动挑卷积算法，速度略快但每次结果可能不同；
    # 为了让实验可复现，这里关掉自动挑选并开启 cuDNN 确定性模式；
    # 它不能保证所有算子的结果逐位一致。
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(spec: str = "auto") -> torch.device:
    """把配置里的设备字符串翻译成 torch.device。

    - "auto"：有 CUDA 就用 CUDA，其次 Apple MPS，最后 CPU；
    - 其他值原样交给 torch.device，例如 "cuda"、"cuda:1"、"cpu"、"mps"。

    显式指定了当前机器不存在的设备时会直接报错，而不是悄悄退回 CPU：
    训练半小时后才发现其实一直跑在 CPU 上，代价太大。
    """
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(spec)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"配置要求使用 {spec}，但当前环境没有可用的 CUDA")
        if device.index is not None and not 0 <= device.index < torch.cuda.device_count():
            raise RuntimeError(
                f"配置要求使用 {spec}，但当前只有 {torch.cuda.device_count()} 张 CUDA 设备"
            )
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(f"配置要求使用 {spec}，但当前环境没有可用的 MPS")
    return device


def make_run_dir(root: str | Path = "outputs", tag: str = "run") -> Path:
    """在 root 下新建本次运行的产物目录并返回路径。

    目录名形如 outputs/20261022-140312_run/，时间戳用于区分不同次运行，
    tag 可以写实验名（如 "resnet18_s320"）方便事后辨认。

    这里用 exist_ok=False：同一秒内重复调用会直接报错，
    避免两次训练稀里糊涂地写进同一个目录、互相覆盖 best.pt。
    """
    safe_tag = re.sub(r"[^0-9A-Za-z._-]+", "_", str(tag)).strip("._-")
    if not safe_tag:
        safe_tag = "run"
    run_dir = Path(root) / f"{datetime.now():%Y%m%d-%H%M%S}_{safe_tag}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir
