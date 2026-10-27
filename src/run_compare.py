"""对比入口：把多个训练 run 放到同一份 valid 上重新算指标，输出一张对比表。

用法（在仓库根目录下执行）：

    python src/run_compare.py \
        --runs outputs/20261026-200000_short_baseline outputs/20261026-210000_short_pos_weight \
        --thresholds 0.3,0.4,0.5,0.6,0.7 \
        --out outputs/compare

每个 run 目录都必须是 run_train.py 的产物，也就是同时有：

    config.yaml   训练时的完整配置（决定图像尺寸、归一化、数据路径）
    best.pt       valid macro-AUC 最高的权重

脚本对每个 run 做三件事：

1. 优先用 best.pt 内保存的配置重建 valid DataLoader，旧权重未含配置时才使用
   run 目录的 config.yaml；
2. 加载 best.pt 重新推理，在内存里重算 macro-AUC 和多阈值下的 macro-F1；
3. 把配置要点（骨干、尺寸、seed、pos_weight…）和指标拼成一张表，同时写出
   summary.csv、per_class.csv 和 summary.md（含逐类结果）。

为什么重算而不是抄训练日志里的 metrics.csv：不同 run 可能在不同时间、用不同
代码训练过，日志里的数字口径未必一致。这里使用同一套指标函数重算；使用者须
确认各 run 的 valid 是同一份数据，才能据此比较模型。
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from common.checkpoint import load_checkpoint
from common.config import config_from_dict, load_config
from common.constants import CLASS_IDS, CSV_ENCODING
from common.metrics import macro_auc, macro_f1_curve, parse_thresholds, per_class_report
from common.utils import resolve_device
from data.dataset import build_loader
from infer.predict import load_checkpoint_model, predict_loader


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数；argv 显式传入方便测试，平时直接用默认的系统参数。"""
    parser = argparse.ArgumentParser(description="把多个训练 run 放到同一份 valid 上对比")
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="run 目录列表（每个目录含 config.yaml 与 best.pt）",
    )
    parser.add_argument(
        "--thresholds",
        default="0.3,0.4,0.5,0.6,0.7",
        help="逗号分隔的阈值网格，用来算各阈值下的 macro-F1（默认 0.3,0.4,0.5,0.6,0.7）",
    )
    parser.add_argument(
        "--out", default="outputs/compare", help="对比表输出目录（默认 outputs/compare）"
    )
    parser.add_argument("--device", default="auto", help="评估设备，默认 auto")
    return parser.parse_args(argv)


def evaluate_run(
    run_dir: Path, thresholds: list[float], device: torch.device
) -> tuple[dict[str, Any], pd.DataFrame]:
    """在一个 run 上重算指标，返回汇总行与逐类报告。"""
    missing = [name for name in ("config.yaml", "best.pt") if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"run 目录 {run_dir} 缺少 {missing}；它应该是 run_train.py 的输出目录"
        )

    # 续训可能使用新配置；应与 best.pt 所保存的配置对齐。旧文件缺配置才回退。
    checkpoint = load_checkpoint(run_dir / "best.pt", map_location="cpu")
    raw_config = checkpoint.get("config")
    config = (
        load_config(run_dir / "config.yaml")
        if raw_config is None
        else config_from_dict(raw_config)
    )
    loader = build_loader(config, "valid", pin_memory=device.type == "cuda")
    model = load_checkpoint_model(config, run_dir / "best.pt", device)
    predictions = predict_loader(model, loader, device)

    probability_columns = [f"p{class_id}" for class_id in CLASS_IDS]
    y_score = predictions[probability_columns].to_numpy(dtype=np.float64)
    y_true = np.stack([record.labels for record in loader.dataset.records])

    # checkpoint 里存了训练时的 seed、设备与最优 epoch，直接引用，省得从目录名
    # 或命令行回忆；旧 checkpoint 没有这些字段时退回配置里的值 / 显示「未知」。
    metadata = checkpoint.get("metadata") or {}

    row: dict[str, Any] = {
        "run": run_dir.name,
        "config_source": "config.yaml（旧权重回退）" if raw_config is None else "best.pt",
        "valid_csv": str(config.data.valid.csv),
        "backbone": config.model.backbone,
        "image_size": config.image.size,
        "pos_weight": _yes_no(config.train.pos_weight),
        "seed": checkpoint.get("seed", config.seed),
        "epochs": config.train.epochs,
        "best_epoch": checkpoint.get("epoch"),
        "pretrained": _yes_no(metadata.get("pretrained_loaded")),
        "train_device": metadata.get("device", "未知"),
        "valid_studies": len(y_true),
        "macro_auc": macro_auc(y_true, y_score),
    }
    # 阈值是浮点数，列名写成 f1@0.3 这种形式，贴进 Markdown 一眼能看懂。
    curve = macro_f1_curve(y_true, y_score, thresholds)
    for threshold, f1 in zip(curve["threshold"], curve["macro_f1"]):
        row[f"f1@{threshold:g}"] = f1
    class_report = per_class_report(y_true, y_score, config.eval.threshold)
    class_report.insert(0, "run", run_dir.name)
    class_report.insert(2, "threshold", config.eval.threshold)
    return row, class_report


def _yes_no(value: Any) -> str:
    """把 True/False/None 翻译成表格里更好读的「是/否/未知」。"""
    if value is None:
        return "未知"
    return "是" if value else "否"


def _to_markdown(table: pd.DataFrame) -> str:
    """手写一个小 Markdown 表格，避免为了 to_markdown 再装 tabulate。"""
    columns = list(table.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in table.iterrows():
        cells = [f"{value:.4f}" if isinstance(value, float) else str(value) for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_summary(
    table: pd.DataFrame,
    per_class: pd.DataFrame,
    out_dir: Path,
    thresholds: list[float],
    evaluate_device: torch.device,
) -> tuple[Path, Path]:
    """写出汇总及逐类报告，返回汇总 CSV 与 Markdown 路径。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "summary.csv"
    md_path = out_dir / "summary.md"
    table.to_csv(csv_path, index=False, encoding=CSV_ENCODING)
    per_class.to_csv(out_dir / "per_class.csv", index=False, encoding=CSV_ENCODING)

    lines = [
        "# 训练 run 对比",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"- 评估设备：{evaluate_device}",
        "- 评估数据：各 run 配置里的 valid；请核对路径和数据内容一致后再比较（study 数见表格）",
        "- 指标：macro-AUC 与各阈值下的 macro-F1，均在本地重算",
        f"- 阈值网格：{', '.join(f'{value:g}' for value in thresholds)}",
        "",
        _to_markdown(table),
        "",
        "## 逐类指标",
        "",
        "F1、precision、recall 使用各 run 配置的 threshold；AUC 与阈值无关。",
        "",
        _to_markdown(per_class),
        "",
    ]
    md_path.write_text("\n".join(lines), encoding=CSV_ENCODING)
    return csv_path, md_path


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    thresholds = parse_thresholds(args.thresholds)
    device = resolve_device(args.device)

    rows: list[dict[str, Any]] = []
    reports: list[pd.DataFrame] = []
    for run in args.runs:
        run_dir = Path(run)
        print(f"评估 {run_dir} ...")
        row, class_report = evaluate_run(run_dir, thresholds, device)
        rows.append(row)
        reports.append(class_report)
        # 每个 run 算完先打印一行，跑多个 run 时能立刻看出哪个有问题。
        f1_text = ", ".join(
            f"{column}={value:.4f}" for column, value in row.items() if column.startswith("f1@")
        )
        print(f"  macro-AUC {row['macro_auc']:.4f} | {f1_text}")

    table = pd.DataFrame(rows)
    csv_path, md_path = write_summary(
        table, pd.concat(reports, ignore_index=True), Path(args.out), thresholds, device
    )
    print()
    print(_to_markdown(table))
    print()
    print(f"已写出：{csv_path}")
    print(f"已写出：{Path(args.out) / 'per_class.csv'}")
    print(f"已写出：{md_path}")


if __name__ == "__main__":
    main()
