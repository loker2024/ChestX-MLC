"""验证入口：加载 checkpoint，在带标注的 split 上输出 macro-AUC / macro-F1 与逐类明细。

用法（在仓库根目录下执行）：

    python src/run_eval.py --config configs/default.yaml --checkpoint outputs/<run>/best.pt

P7 挑判定阈值时，可以在同一条命令上多给一个阈值网格，一次看完整个 macro-F1 曲线：

    python src/run_eval.py --config configs/default.yaml --checkpoint outputs/<run>/best.pt \
        --thresholds 0.3,0.4,0.5,0.6,0.7

脚本只做「读配置 → 建 loader → 加载权重 → 推理 → 算指标」，
训练逻辑在 train.engine，提交逻辑在 infer。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common.config import load_config
from common.constants import CLASS_IDS
from common.metrics import macro_f1_curve, parse_thresholds, per_class_report
from common.utils import resolve_device
from data.dataset import build_loader
from infer.predict import load_checkpoint_model, predict_loader


def save_predictions(path: str | Path, predictions: pd.DataFrame, y_true: np.ndarray) -> Path:
    """把逐 study 的真实标签与 10 类概率写成 UTF-8 CSV。

    B 组评估要求保留「逐 Study ID / 真实标签 / 10 类概率」，供事后复算指标与
    对比分析；Study_id 按字符串写出，保留前导零。这里只写宽表，不生成赛题
    submission.csv（B 组不用于提交）。
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"Study_id": predictions["Study_id"].astype(str)})
    for index, class_id in enumerate(CLASS_IDS):
        frame[f"label_{class_id}"] = y_true[:, index].astype(int)
    for class_id in CLASS_IDS:
        frame[f"p{class_id}"] = predictions[f"p{class_id}"].to_numpy(dtype=np.float64)
    frame.to_csv(output_path, index=False, encoding="utf-8")
    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。argv 显式传入是为了测试时复用，平时直接传 None 即可。"""
    parser = argparse.ArgumentParser(description="在 valid 上评估 ChestX-MLC checkpoint")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML 配置文件路径")
    parser.add_argument("--checkpoint", required=True, help="训练产出的 best.pt")
    parser.add_argument(
        "--split",
        default="valid",
        choices=("train", "valid"),
        help="在哪个带标注的 split 上评估，默认 valid",
    )
    parser.add_argument(
        "--thresholds",
        default=None,
        help="逗号分隔的阈值网格，例如 0.3,0.4,0.5；给了就额外打印各阈值下的 macro-F1",
    )
    parser.add_argument(
        "--pred-out",
        default=None,
        help="逐 study 预测 CSV 的输出路径，包含 Study_id、真实标签与 10 类概率",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    device = resolve_device(config.device)

    loader = build_loader(config, args.split, pin_memory=device.type == "cuda")
    model = load_checkpoint_model(config, args.checkpoint, device)
    predictions = predict_loader(model, loader, device, config)

    probability_columns = [f"p{class_id}" for class_id in CLASS_IDS]
    y_score = predictions[probability_columns].to_numpy(dtype=np.float64)
    y_true = np.stack([record.labels for record in loader.dataset.records])

    # 与训练时 valid 指标使用同一个函数，保证两边数字可比。
    report = per_class_report(y_true, y_score, config.eval.threshold)
    print(f"checkpoint: {args.checkpoint}")
    print(f"设备: {device} | split: {args.split} | study 数: {len(y_true)}")
    print(f"macro-AUC: {report['auc'].mean():.4f}")
    print(f"macro-F1（阈值 {config.eval.threshold:g}）: {report['f1'].mean():.4f}")

    # 阈值网格单独一张小表：AUC 不受阈值影响，这里只看 F1 随阈值怎么变。
    if args.thresholds:
        curve = macro_f1_curve(y_true, y_score, parse_thresholds(args.thresholds))
        print("不同阈值下的 macro-F1:")
        print(curve.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    print("逐类明细:")
    print(report.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    # 逐 study 宽表放最后写：即使前面只关心汇总，也不影响原有输出格式。
    if args.pred_out:
        output_path = save_predictions(args.pred_out, predictions, y_true)
        print(f"逐 study 预测已写入: {output_path}")


if __name__ == "__main__":
    main()
