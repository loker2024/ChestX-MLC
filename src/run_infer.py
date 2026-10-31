"""推理入口：加载 checkpoint，在 test split 上推理并生成提交文件。

用法（在仓库根目录下执行）：

    python src/run_infer.py --config configs/default.yaml \
        --checkpoint outputs/<run>/best.pt --out outputs/submission.csv

默认输出 test 集每个 study 的全部 10 类概率（macro-AUC 需要完整概率）；
加 --threshold 0.5 则只输出概率过线的类别，与赛题示例的“每行一个预测类别”
一致。无论哪种模式，每个 study 至少保留一行，提交文件覆盖全部 test study。
"""

from __future__ import annotations

import argparse
import math

from common.config import load_config
from common.utils import resolve_device
from data.dataset import build_loader
from infer.predict import load_checkpoint_model, predict_loader, validate_submission_model
from infer.submission import build_submission_frame, validate_submission, write_submission


def _probability(value: str) -> float:
    """argparse 类型：把 --threshold 限制在 [0, 1] 的有限数。"""
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"阈值必须是数字，实际为 {value!r}") from error
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError(f"阈值必须落在 [0, 1]，实际为 {value!r}")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。argv 显式传入是为了测试时复用，平时直接传 None 即可。"""
    parser = argparse.ArgumentParser(description="用 checkpoint 生成 ChestX-MLC 提交文件")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML 配置文件路径")
    parser.add_argument("--checkpoint", required=True, help="训练产出的 best.pt")
    parser.add_argument(
        "--out", default="outputs/submission.csv", help="提交文件输出路径"
    )
    parser.add_argument(
        "--threshold",
        type=_probability,
        default=None,
        help="只输出概率 >= 该阈值的类别；默认输出全部 10 类概率",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    validate_submission_model(config, args.checkpoint)
    device = resolve_device(config.device)

    loader = build_loader(config, "test", pin_memory=device.type == "cuda")
    model = load_checkpoint_model(config, args.checkpoint, device)
    predictions = predict_loader(model, loader, device, config)

    # Subject_id 不在推理宽表里，从 test split 的索引记录补上。
    study_subjects = {
        record.study_id: record.subject_id for record in loader.dataset.records
    }
    frame = build_submission_frame(predictions, study_subjects, args.threshold)
    path = write_submission(frame, args.out)
    validate_submission(
        path, config.data.test.csv, require_all_classes=args.threshold is None
    )

    strategy = "全部 10 类概率" if args.threshold is None else f"概率 >= {args.threshold:g}"
    print(f"checkpoint: {args.checkpoint}")
    print(f"设备: {device} | test study 数: {len(predictions)}")
    print(f"提交文件: {path} | {strategy} | 共 {len(frame)} 行")


if __name__ == "__main__":
    main()
