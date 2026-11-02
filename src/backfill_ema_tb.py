"""把 `metrics.csv` 里的 EMA 列补写成 TensorBoard 曲线，供训练结束后补图。

用法（在仓库根目录下执行）：

    python src/backfill_ema_tb.py outputs/20261102-145203_D-002-a
    python src/backfill_ema_tb.py outputs --dry-run

背景：`train/engine.py` 原先只把 raw 模型的验证指标写进 TensorBoard，EMA 的
三项指标只落在 `metrics.csv` 的 `ema_valid_*` 列里。2026-11-02 之后的训练已经
直接写 `ema/` 曲线，但此前的 run 需要按本脚本从 CSV 回填。

写入方式是在 run 的 `tb/` 下新建一个带 `.ema_backfill` 后缀的 events 文件，
不碰已有事件文件；tag 与训练时一致（`ema/loss`、`ema/macro_auc`、
`ema/macro_f1`、`ema/best_macro_auc`），`step` 用真实 epoch 号，所以补出来的
曲线与 `valid/` 那组在横轴上对齐。

脚本是幂等的：目录里已经存在 `ema/macro_auc` 就跳过。确实要重写时加 `--force`，
它会先删掉此前由本脚本生成的 `.ema_backfill` 文件，不会重复叠点。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import pandas as pd

from common.constants import CSV_ENCODING
from common.tracking import TENSORBOARD_DIR_NAME

# 本脚本生成的 events 文件后缀，用来识别并清理上一次的回填结果。
BACKFILL_SUFFIX = "ema_backfill"
BACKFILL_PATTERN = f"events.out.tfevents.*{BACKFILL_SUFFIX}"

# CSV 列 -> TensorBoard tag，顺序即写入顺序。
EMA_COLUMNS = {
    "ema_valid_loss": "ema/loss",
    "ema_valid_macro_auc": "ema/macro_auc",
    "ema_valid_macro_f1": "ema/macro_f1",
}


def read_epochs(run_dir: Path) -> pd.DataFrame | None:
    """读一个 run 的 `metrics.csv`；没有 EMA 列时返回 `None`。"""
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        return None
    frame = pd.read_csv(metrics_path, encoding=CSV_ENCODING)
    if not set(EMA_COLUMNS) <= set(frame.columns):
        return None
    return frame


def existing_tags(tb_dir: Path) -> set[str]:
    """读出 tb 目录里已记录的 tag；目录不存在或没有事件文件时返回空集。"""
    if not tb_dir.exists():
        return set()
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(str(tb_dir))
    accumulator.Reload()
    return set(accumulator.Tags().get("scalars", []))


def best_curve(values: list[float]) -> list[float]:
    """把逐轮 AUC 变成历史最优曲线（台阶状、不回落），忽略 NaN。"""
    best = -math.inf
    curve: list[float] = []
    for value in values:
        if not math.isnan(value) and value > best:
            best = value
        curve.append(best if math.isfinite(best) else float("nan"))
    return curve


def backfill_run(run_dir: Path, *, force: bool = False, dry_run: bool = False) -> str:
    """给一个 run 补 EMA 曲线，返回一句人类可读的结果描述。"""
    frame = read_epochs(run_dir)
    if frame is None:
        return f"{run_dir.name}: 跳过（没有 metrics.csv 或没有 ema_valid_* 列）"

    tb_dir = run_dir / TENSORBOARD_DIR_NAME
    if dry_run:
        steps = ", ".join(str(int(epoch)) for epoch in frame["epoch"])
        return f"{run_dir.name}: 将写入 {len(frame)} 个 epoch（step {steps}）→ {tb_dir}"

    if not force and "ema/macro_auc" in existing_tags(tb_dir):
        return f"{run_dir.name}: 跳过（已有 EMA 曲线，重写请加 --force）"

    from torch.utils.tensorboard import SummaryWriter

    if force:
        for stale in tb_dir.glob(BACKFILL_PATTERN):
            stale.unlink()

    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(tb_dir), filename_suffix=BACKFILL_SUFFIX)
    try:
        for column, tag in EMA_COLUMNS.items():
            for epoch, value in zip(frame["epoch"], frame[column]):
                if pd.isna(value):
                    continue
                writer.add_scalar(tag, float(value), int(epoch))
        auc_values = [float(value) for value in frame["ema_valid_macro_auc"]]
        for epoch, value in zip(frame["epoch"], best_curve(auc_values)):
            if math.isnan(value):
                continue
            writer.add_scalar("ema/best_macro_auc", value, int(epoch))
    finally:
        writer.close()

    return f"{run_dir.name}: 已写入 {len(frame)} 个 epoch 的 EMA 曲线"


def find_runs(paths: list[Path]) -> list[Path]:
    """把命令行给的路径展开成 run 目录列表：目录本身有 metrics.csv 就算一个 run，
    否则向下找一层子目录（方便直接传 outputs）。"""
    runs: list[Path] = []
    for path in paths:
        if not path.exists():
            print(f"{path}: 路径不存在，已跳过")
            continue
        if (path / "metrics.csv").exists():
            runs.append(path)
            continue
        runs.extend(sorted(child for child in path.iterdir() if (child / "metrics.csv").exists()))
    return runs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 metrics.csv 回填 EMA 的 TensorBoard 曲线")
    parser.add_argument(
        "runs",
        nargs="+",
        type=Path,
        help="run 目录，或包含多个 run 的上级目录（如 outputs）",
    )
    parser.add_argument("--force", action="store_true", help="已有 EMA 曲线时重写（先删旧的补写文件）")
    parser.add_argument("--dry-run", action="store_true", help="只打印会写什么，不落盘")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runs = find_runs(args.runs)
    if not runs:
        print("没有找到含 metrics.csv 的 run 目录")
        return 1

    failed = 0
    for run_dir in runs:
        try:
            print(backfill_run(run_dir, force=args.force, dry_run=args.dry_run))
        except Exception as error:  # noqa: BLE001 - 单个 run 失败不该中断整批
            failed += 1
            print(f"{run_dir.name}: 失败 —— {type(error).__name__}: {error}")
    print(f"共 {len(runs)} 个 run，失败 {failed} 个")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
