"""训练曲线的可选记录器：装了 tensorboard 就写，没装就安静跳过。

训练本身不依赖 tensorboard —— 赛题服务器只保证 PyTorch，不保证这个包。
所以「能不能用」的判断集中在 `build_writer` 一处，其余模块只关心拿到的是
不是一个可写的 writer（`None` 表示本次不记录）。

三种模式（对应配置项 `train.tensorboard`）：

- `auto`（默认）：能导入 `torch.utils.tensorboard` 就写曲线，导入不了就跳过，
  并在训练日志里留一行说明，免得对着空面板找原因；
- `true`：必须写。导入失败直接抛错——明确要求了就别装作记上了；
- `false`：不写。

曲线落在 `run_dir/tb/` 下，和 `config.yaml`、`metrics.csv`、`best.pt` 同属
一个 run 目录，复盘时整个目录搬走即可。把 tensorboard 的 `--logdir` 指向
`outputs/`（run 目录的上级）时，每个 run 的 `tb/` 会被识别成一个独立面板，
面板名就是 run 目录名加 `/tb` 后缀。

开了 EMA 时另起 `run_dir/tb_ema/` 一个独立面板（面板名 `<run>/tb_ema`），
tag 与 raw 面板同名（`valid/loss` 等），选中两个 run 就能在同一张图里对比
raw 与 EMA，而不是在原实验面板里塞一组 `ema/` 前缀曲线。

`SummaryWriter` 的 flush 间隔从默认 120 秒收紧到 30 秒，代价可以忽略，
换来的是进程被 kill 时最多只丢 30 秒的曲线。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# 允许的取值，与 TrainConfig.__post_init__ 共用一份，避免两处各写一套。
TENSORBOARD_MODES = frozenset({"auto", "true", "false"})

# 曲线子目录名，相对于 run 目录。
TENSORBOARD_DIR_NAME = "tb"

# EMA 影子权重单独成面板，与 raw 面板并列；目录名同时决定 TensorBoard 里的面板名。
TENSORBOARD_EMA_DIR_NAME = "tb_ema"

# SummaryWriter 的自动 flush 间隔（秒）。
FLUSH_SECS = 30

_MISSING_HINT = (
    "未安装 tensorboard，本次不写训练曲线（train.tensorboard=auto 的预期行为）。"
    "需要曲线时执行：python -m pip install tensorboard"
)


def _import_summary_writer() -> Any | None:
    """尝试拿到 SummaryWriter 类，拿不到就返回 None。

    只捕获 ImportError：包缺失或 `torch.utils.tensorboard` 依赖的东西不全，
    都属于「没装好」。其他异常（例如磁盘、权限）不应该被这里吞掉。
    """
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        return None
    return SummaryWriter


def build_writer(log_dir: str | Path, mode: str = "auto") -> Any | None:
    """按模式创建 SummaryWriter。

    参数：

    - `log_dir`：曲线目录，调用方传 `run_dir / "tb"`；
    - `mode`：`auto` / `true` / `false`，见模块文档。

    返回 `SummaryWriter` 或 `None`。`None` 意味着调用方不需要再管它，
    训练照常进行。
    """
    if mode not in TENSORBOARD_MODES:
        raise ValueError(
            f"train.tensorboard 只能是 {'、'.join(sorted(TENSORBOARD_MODES))}，当前为 {mode!r}"
        )
    if mode == "false":
        return None

    writer_class = _import_summary_writer()
    if writer_class is None:
        if mode == "true":
            raise ImportError(
                "train.tensorboard=true 要求安装 tensorboard，但导入失败；"
                "请执行 python -m pip install tensorboard，或把该项改回 auto"
            )
        print(_MISSING_HINT)
        return None

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    return writer_class(log_dir=str(log_path), flush_secs=FLUSH_SECS)
