"""多模型融合入口：按 checkpoint 配置推理后融合并评估或生成提交。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from common.checkpoint import load_checkpoint
from common.config import Config, config_from_dict, load_config
from common.constants import CLASS_IDS
from common.metrics import macro_auc, macro_f1, macro_f1_curve, parse_thresholds, per_class_report
from common.utils import resolve_device
from data.dataset import build_loader
from infer.ensemble import fuse_prediction_frames, normalize_weights
from infer.predict import load_checkpoint_model, predict_loader, predict_loader_logits
from infer.submission import build_submission_frame, validate_submission, write_submission


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="融合多个 ChestX-MLC checkpoint")
    parser.add_argument("--checkpoint", nargs="+", required=True, help="一个或多个 best.pt")
    parser.add_argument(
        "--config",
        nargs="+",
        default=None,
        help="可选配置路径列表；数量必须与 checkpoint 相同，仅用于旧 checkpoint 回退",
    )
    parser.add_argument("--weights", default=None, help="逗号分隔的模型权重，默认等权")
    parser.add_argument(
        "--fusion",
        choices=("probability", "logit"),
        default="probability",
        help="融合概率或 study-level logits",
    )
    parser.add_argument(
        "--split",
        choices=("train", "valid", "test"),
        default="test",
        help="推理 split；train/valid 会额外计算指标",
    )
    parser.add_argument(
        "--thresholds",
        default=None,
        help="train/valid 的 F1 阈值网格，例如 0.3,0.5,0.7",
    )
    parser.add_argument("--out", default="outputs/ensemble", help="输出目录")
    return parser.parse_args(argv)


def _load_model_config(checkpoint_path: Path, config_path: str | None) -> Config:
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    raw_config = checkpoint.get("config")
    if raw_config is not None:
        return config_from_dict(raw_config)
    if config_path is None:
        raise ValueError(
            f"checkpoint {checkpoint_path} 未保存 config，必须通过 --config 提供对应配置"
        )
    return load_config(config_path)


def _parse_weights(text: str | None, count: int) -> np.ndarray:
    if text is None:
        return normalize_weights(None, count)
    pieces = [piece.strip() for piece in text.split(",") if piece.strip()]
    if not pieces:
        raise ValueError("--weights 不能为空")
    try:
        values = [float(piece) for piece in pieces]
    except ValueError as error:
        raise ValueError("--weights 必须是逗号分隔的数字") from error
    return normalize_weights(values, count)


def _check_config_paths(config_paths: list[str] | None, count: int) -> list[str | None]:
    if config_paths is None:
        return [None] * count
    if len(config_paths) != count:
        raise ValueError(f"--config 数量必须与 checkpoint 数量一致，当前为 {len(config_paths)} 与 {count}")
    return list(config_paths)


def _check_labeled_loaders(loaders: list[Any]) -> dict[str, np.ndarray]:
    first_records = loaders[0].dataset.records
    labels = {record.study_id: np.asarray(record.labels, dtype=np.float64) for record in first_records}
    for index, loader in enumerate(loaders[1:], start=2):
        records = loader.dataset.records
        current = {record.study_id: np.asarray(record.labels, dtype=np.float64) for record in records}
        if set(current) != set(labels):
            raise ValueError(f"第 {index} 个模型的 Study_id 与第 1 个模型不一致")
        for study_id, label in labels.items():
            if not np.array_equal(current[study_id], label):
                raise ValueError(f"第 {index} 个模型的 study {study_id} 标签与第 1 个模型不一致")
    return labels


def _write_labeled_outputs(
    output_dir: Path,
    predictions,
    labels: dict[str, np.ndarray],
    configs: list[Config],
    weights: np.ndarray,
    fusion: str,
    split: str,
    threshold_text: str | None,
) -> None:
    y_true = np.stack([labels[study_id] for study_id in predictions["Study_id"]])
    y_score = predictions[[f"p{class_id}" for class_id in CLASS_IDS]].to_numpy(dtype=np.float64)
    threshold = configs[0].eval.threshold
    report = per_class_report(y_true, y_score, threshold)
    threshold_values = (
        parse_thresholds(threshold_text) if threshold_text is not None else [threshold]
    )
    curve = macro_f1_curve(y_true, y_score, threshold_values)
    metrics = {
        "split": split,
        "fusion": fusion,
        "weights": weights.tolist(),
        "threshold": threshold,
        "study_count": int(len(y_true)),
        "macro_auc": macro_auc(y_true, y_score),
        "macro_f1": macro_f1(y_true, y_score, threshold),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    report.to_csv(output_dir / "per_class.csv", index=False, encoding="utf-8")
    curve.to_csv(output_dir / "f1_curve.csv", index=False, encoding="utf-8")
    predictions.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8")
    print(f"macro-AUC: {metrics['macro_auc']:.4f}")
    print(f"macro-F1（阈值 {threshold:g}）: {metrics['macro_f1']:.4f}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    checkpoint_paths = [Path(path) for path in args.checkpoint]
    config_paths = _check_config_paths(args.config, len(checkpoint_paths))
    configs = [
        _load_model_config(checkpoint_path, config_path)
        for checkpoint_path, config_path in zip(checkpoint_paths, config_paths)
    ]
    weights = _parse_weights(args.weights, len(checkpoint_paths))
    device = resolve_device(configs[0].device)

    prediction_frames = []
    loaders = []
    for checkpoint_path, config in zip(checkpoint_paths, configs):
        loader = build_loader(config, args.split, pin_memory=device.type == "cuda")
        loaders.append(loader)
        model = load_checkpoint_model(config, checkpoint_path, device)
        if args.fusion == "logit":
            prediction_frames.append(predict_loader_logits(model, loader, device, config))
        else:
            prediction_frames.append(predict_loader(model, loader, device, config))

    predictions = fuse_prediction_frames(prediction_frames, weights, args.fusion)
    output_dir = Path(args.out)
    if args.split in {"train", "valid"}:
        labels = _check_labeled_loaders(loaders)
        _write_labeled_outputs(
            output_dir,
            predictions,
            labels,
            configs,
            weights,
            args.fusion,
            args.split,
            args.thresholds,
        )
        return

    study_subjects = {
        record.study_id: record.subject_id for record in loaders[0].dataset.records
    }
    frame = build_submission_frame(predictions, study_subjects)
    submission_path = write_submission(frame, output_dir / "submission.csv")
    validate_submission(submission_path, configs[0].data.test.csv, require_all_classes=True)
    print(f"提交文件: {submission_path} | test study 数: {len(predictions)}")


if __name__ == "__main__":
    main()
