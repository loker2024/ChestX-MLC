"""EMA 回填脚本的测试：曲线换算、无 EMA 列时的跳过、独立面板、幂等与旧版清理。

回填本身要写 events 文件，依赖 tensorboard，所以写入用例按是否装了该包做 skip；
`best_curve`、`read_epochs`、`find_runs` 是纯函数，任何环境都跑。
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from backfill_ema_tb import (
    BACKFILL_PATTERN,
    BACKFILL_SUFFIX,
    backfill_run,
    best_curve,
    find_runs,
    read_epochs,
)
from common.tracking import TENSORBOARD_DIR_NAME, TENSORBOARD_EMA_DIR_NAME


def _tensorboard_available() -> bool:
    try:
        import torch.utils.tensorboard  # noqa: F401
    except ImportError:
        return False
    return True


requires_tensorboard = pytest.mark.skipif(
    not _tensorboard_available(),
    reason="需要 tensorboard：python -m pip install tensorboard",
)


def _write_run(root: Path, *, rows: int = 4, with_ema: bool = True) -> Path:
    """造一个最小 run 目录：只有 metrics.csv，tb_ema/ 由回填脚本自己建。"""
    run_dir = root / "20260101-000000_demo"
    run_dir.mkdir()
    data = {
        "epoch": list(range(1, rows + 1)),
        "valid_macro_auc": [0.70, 0.72, 0.71, 0.75][:rows],
    }
    if with_ema:
        data.update(
            {
                "ema_valid_loss": [0.95, 0.93, 0.94, 0.90][:rows],
                "ema_valid_macro_auc": [0.71, 0.73, 0.72, 0.76][:rows],
                "ema_valid_macro_f1": [0.38, 0.39, 0.39, 0.41][:rows],
            }
        )
    pd.DataFrame(data).to_csv(run_dir / "metrics.csv", index=False, encoding="utf-8")
    return run_dir


def _read_scalars(log_dir: Path) -> dict[str, list[float]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(str(log_dir))
    accumulator.Reload()
    return {
        tag: [event.value for event in accumulator.Scalars(tag)]
        for tag in accumulator.Tags()["scalars"]
    }


def test_best_curve_is_non_decreasing() -> None:
    """历史最优曲线只能持平或上升，NaN 轮次沿用前值。"""
    curve = best_curve([0.70, float("nan"), 0.68, 0.75])
    assert curve[:2] == [0.70, 0.70]
    assert curve == sorted(curve)
    assert curve[-1] == 0.75


def test_best_curve_starts_with_nan() -> None:
    """首轮就是 NaN 时没有最优值可沿用，保持 NaN 而不是伪造 0。"""
    assert math.isnan(best_curve([float("nan"), 0.6])[0])
    assert best_curve([float("nan"), 0.6])[1] == 0.6


def test_read_epochs_returns_none_without_ema(tmp_path: Path) -> None:
    """没开 EMA 的 run 没有 ema_valid_* 列，必须返回 None 而不是空表。"""
    run_dir = _write_run(tmp_path, with_ema=False)
    assert read_epochs(run_dir) is None


def test_find_runs_accepts_parent_and_run_dir(tmp_path: Path) -> None:
    """既可以直接传 run 目录，也可以传 outputs 这种上级目录。"""
    run_dir = _write_run(tmp_path)
    assert find_runs([run_dir]) == [run_dir]
    assert find_runs([tmp_path]) == [run_dir]


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)
    message = backfill_run(run_dir, dry_run=True)
    assert "将写入 4 个 epoch" in message
    assert not (run_dir / TENSORBOARD_EMA_DIR_NAME).exists()


@requires_tensorboard
def test_backfill_writes_separate_ema_panel(tmp_path: Path) -> None:
    """回填写进 tb_ema/，tag 与 raw 同名；原 tb/ 不建，两者不混。"""
    run_dir = _write_run(tmp_path)
    message = backfill_run(run_dir)
    assert "已写入 4 个 epoch" in message
    assert not (run_dir / TENSORBOARD_DIR_NAME).exists()

    scalars = _read_scalars(run_dir / TENSORBOARD_EMA_DIR_NAME)
    for tag in ("valid/loss", "valid/macro_auc", "valid/macro_f1", "valid/best_macro_auc"):
        assert tag in scalars, f"缺 {tag}"
        assert len(scalars[tag]) == 4
    assert not [tag for tag in scalars if tag.startswith("ema/")]
    # events 里存的是 float32，逐项比较要用 approx。
    assert scalars["valid/best_macro_auc"] == pytest.approx([0.71, 0.73, 0.73, 0.76])


@requires_tensorboard
def test_backfill_purges_legacy_files_in_tb(tmp_path: Path) -> None:
    """旧版写在 tb/ 下的补写文件必须被清掉，原实验面板不能残留 ema 曲线。"""
    run_dir = _write_run(tmp_path)
    tb_dir = run_dir / TENSORBOARD_DIR_NAME
    tb_dir.mkdir()
    legacy = tb_dir / f"events.out.tfevents.1.host.1.{BACKFILL_SUFFIX}"
    legacy.write_bytes(b"")

    message = backfill_run(run_dir)
    assert "清理旧版混入 tb/ 的 1 个文件" in message
    assert not legacy.exists()
    assert list(tb_dir.glob(BACKFILL_PATTERN)) == []
    assert (run_dir / TENSORBOARD_EMA_DIR_NAME).exists()


@requires_tensorboard
def test_backfill_is_idempotent(tmp_path: Path) -> None:
    """重复执行不该叠出双份点；--force 重写后点数仍然只有一份。"""
    run_dir = _write_run(tmp_path)
    ema_dir = run_dir / TENSORBOARD_EMA_DIR_NAME

    backfill_run(run_dir)
    assert "跳过" in backfill_run(run_dir)
    assert len(_read_scalars(ema_dir)["valid/macro_auc"]) == 4

    assert "已写入" in backfill_run(run_dir, force=True)
    assert len(_read_scalars(ema_dir)["valid/macro_auc"]) == 4
    assert len(list(ema_dir.glob(BACKFILL_PATTERN))) == 1
