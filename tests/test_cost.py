"""cost.py 与 infer.profiling 的测试：计时口径、submission 比对、CLI 端到端。

规则要求 cost.py「与 predict.py 用同一套推理函数」，所以这里不测 100ms 上限
本身（那是硬件相关的事），而是测三件可以稳定复现的事：

1. 目录/图片分组与 submission 解析的边界条件；
2. `predict_image` / `predict_study` 与 `predict_loader` 的概率口径一致；
3. `cost.main` 端到端跑通，结论行包含规则要求的字段，超限或不一致时退出码为 1。

测试全部用合成的小图与随机初始化的 resnet18（image.size=32，CPU），
不下载预训练权重、不碰真实影像。
"""

from __future__ import annotations

import csv
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
# cost.py 在仓库根目录，不在 src/ 里；测试直接 import 它来测 CLI 与内部函数。
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cost  # noqa: E402

from common.checkpoint import save_checkpoint  # noqa: E402
from common.config import load_config  # noqa: E402
from data.dataset import build_loader  # noqa: E402
from infer.predict import (  # noqa: E402
    build_eval_inference_transform,
    load_checkpoint_model,
    predict_image,
    predict_loader,
    predict_study,
)
from infer.profiling import (  # noqa: E402
    COST_LIMIT_MS,
    SubmissionTable,
    compare_study_predictions,
    format_conclusion,
    group_images_by_study,
    list_images,
    parse_image_identity,
    read_submission,
    select_prediction_rows,
    summarize_timings,
)
from infer.submission import build_submission_frame, write_submission  # noqa: E402
from model.network import build_model  # noqa: E402

# 测试目录里图片的摆放：2 个 study、3 张图，study_00000 有两张（验证多图聚合）。
IMAGE_RELATIVES = (
    "subject_00000/study_00000/pic_000010.png",
    "subject_00000/study_00000/pic_000011.png",
    "subject_00001/study_00001/pic_000012.png",
)
# test split 的 CSV 没有标签列。
TEST_HEADER = ["Dicom_id", "Subject_id", "Study_id", "ViewPosition"]
TEST_ROWS = [
    ["pic_000010", "subject_00000", "study_00000", "AP"],
    ["pic_000011", "subject_00000", "study_00000", "LATERAL"],
    ["pic_000012", "subject_00001", "study_00001", "AP"],
]


def _write_image(path: Path, size: tuple[int, int] = (64, 48)) -> None:
    """在 path 处生成一张纯色合成图。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (120, 120, 120)).save(path)


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    """按显式 UTF-8 写一份 CSV（Windows 默认 GBK，不指定会乱码）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """一次性搭好 cost.py 端到端需要的全部文件（模型只建一次，测试更快）。

    包含：test 图片目录（模拟 pictures_100）、5 张预热图目录、config.yaml、
    checkpoint（随机初始化 resnet18）、以及用 predict_loader 生成的 submission.csv。
    """
    base = tmp_path_factory.mktemp("cost")
    images_root = base / "images"
    for relative in IMAGE_RELATIVES:
        _write_image(images_root / relative)
    test_csv = base / "test.csv"
    _write_csv(test_csv, TEST_HEADER, TEST_ROWS)

    warmup_root = base / "warmup"
    for index in range(5):
        _write_image(warmup_root / "subject_w0" / f"study_w{index}" / f"warm_{index:02d}.png")

    config_path = base / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "data:",
                "  train:",
                f"    csv: {test_csv}",
                f"    images: {images_root}",
                "  valid:",
                f"    csv: {test_csv}",
                f"    images: {images_root}",
                "  test:",
                f"    csv: {test_csv}",
                f"    images: {images_root}",
                "image:",
                "  size: 32",
                "train:",
                "  batch_studies: 2",
                "  num_workers: 0",
                "eval:",
                "  batch_studies: 2",
                "device: cpu",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = load_config(config_path)
    config = replace(config, model=replace(config.model, pretrained=False))
    torch.manual_seed(0)
    model = build_model(config)
    checkpoint_path = save_checkpoint(
        base / "best.pt", model, epoch=1, metrics={"macro_auc": 0.5}, config=config, seed=0
    )

    device = torch.device("cpu")
    loader = build_loader(config, "test")
    predictions = predict_loader(model, loader, device)
    study_subjects = {record.study_id: record.subject_id for record in loader.dataset.records}
    submission_frame = build_submission_frame(predictions, study_subjects)
    submission_path = write_submission(submission_frame, base / "submission.csv")

    return {
        "base": base,
        "config": config,
        "config_path": config_path,
        "checkpoint": checkpoint_path,
        "images": images_root,
        "warmup": warmup_root,
        "submission": submission_path,
        "model": model,
        "device": device,
        "loaders": loader,
        "predictions": predictions,
        "transform": build_eval_inference_transform(config),
    }


# --------------------------------------------------------------------------
# 图片发现与分组
# --------------------------------------------------------------------------


def test_list_images_recursive_sorted(env: dict) -> None:
    images = list_images(env["images"])
    assert [path.relative_to(env["images"]).as_posix() for path in images] == sorted(
        IMAGE_RELATIVES
    )


def test_list_images_empty_raises(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="没有图片"):
        list_images(empty)


def test_list_images_missing_root_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list_images(tmp_path / "no_such_dir")


def test_parse_image_identity(tmp_path: Path) -> None:
    identity = parse_image_identity(tmp_path / "subject_1" / "study_2" / "pic_3.jpg")
    assert identity.subject_id == "subject_1"
    assert identity.study_id == "study_2"
    assert identity.path.name == "pic_3.jpg"


def test_parse_image_identity_rejects_shallow_path() -> None:
    # 只有文件名、没有 <subject_id>/<study_id>/ 两级目录。
    with pytest.raises(ValueError, match="层级不足"):
        parse_image_identity(Path("pic_3.jpg"))


def test_group_images_by_study_keeps_order(tmp_path: Path) -> None:
    paths = [
        tmp_path / "subject_1" / "study_b" / "pic_1.png",
        tmp_path / "subject_1" / "study_a" / "pic_2.png",
        tmp_path / "subject_1" / "study_b" / "pic_3.png",
    ]
    groups = group_images_by_study(paths)
    assert list(groups) == ["study_b", "study_a"]
    assert [path.name for path in groups["study_b"]] == ["pic_1.png", "pic_3.png"]


def test_group_images_rejects_conflicting_subject(tmp_path: Path) -> None:
    paths = [
        tmp_path / "subject_1" / "study_x" / "pic_1.png",
        tmp_path / "subject_2" / "study_x" / "pic_2.png",
    ]
    with pytest.raises(ValueError, match="多个 Subject_id"):
        group_images_by_study(paths)


# --------------------------------------------------------------------------
# submission 解析
# --------------------------------------------------------------------------


def test_read_submission_roundtrip(env: dict) -> None:
    table = read_submission(env["submission"])
    assert set(table.study_ids) == {"study_00000", "study_00001"}
    assert not table.filtered
    assert all(len(entry) == 10 for entry in table.probabilities.values())
    assert all(0.0 <= value <= 1.0 for entry in table.probabilities.values() for value in entry.values())


def _write_submission(path: Path, rows: list[list[str]], header: list[str]) -> Path:
    _write_csv(path, header, rows)
    return path


def test_read_submission_rejects_wrong_header(tmp_path: Path) -> None:
    path = _write_submission(
        tmp_path / "bad.csv",
        [["subject_1", "study_1", "0.5", "1"]],
        ["Subject_id", "Study_id", "Probability", "Predict_class"],
    )
    with pytest.raises(ValueError, match="列名或顺序"):
        read_submission(path)


@pytest.mark.parametrize(
    ("rows", "match"),
    [
        pytest.param(
            [["subject_1", "study_1", "11", "0.5"]], "Predict_class", id="class-out-of-range"
        ),
        pytest.param(
            [["subject_1", "study_1", "1", "1.5"]], "Probability", id="probability-out-of-range"
        ),
        pytest.param(
            [["subject_1", "study_1", "1", "nan"]], "Probability", id="probability-not-finite"
        ),
        pytest.param(
            [
                ["subject_1", "study_1", "1", "0.5"],
                ["subject_1", "study_1", "1", "0.6"],
            ],
            "重复提交",
            id="duplicated-class",
        ),
        pytest.param([["", "study_1", "1", "0.5"]], "为空", id="empty-subject"),
    ],
)
def test_read_submission_rejects_invalid_rows(
    tmp_path: Path, rows: list[list[str]], match: str
) -> None:
    path = _write_submission(
        tmp_path / "bad.csv",
        rows,
        ["Subject_id", "Study_id", "Predict_class", "Probability"],
    )
    with pytest.raises(ValueError, match=match):
        read_submission(path)


def test_read_submission_rejects_empty_file(tmp_path: Path) -> None:
    path = _write_submission(
        tmp_path / "empty.csv", [], ["Subject_id", "Study_id", "Predict_class", "Probability"]
    )
    with pytest.raises(ValueError, match="没有任何预测行"):
        read_submission(path)


# --------------------------------------------------------------------------
# 后处理与比对
# --------------------------------------------------------------------------


def test_select_prediction_rows_threshold() -> None:
    probabilities = np.linspace(0.0, 0.9, 10)  # 类别 1..10 概率 0.0, 0.1, ..., 0.9
    selected = select_prediction_rows(probabilities, threshold=0.55)
    assert [class_id for class_id, _ in selected] == [7, 8, 9, 10]


def test_select_prediction_rows_falls_back_to_top1() -> None:
    probabilities = np.full(10, 0.1)
    probabilities[2] = 0.4
    selected = select_prediction_rows(probabilities, threshold=0.5)
    assert selected == [(3, pytest.approx(0.4))]


def test_select_prediction_rows_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="长度必须是 10"):
        select_prediction_rows(np.zeros(9), threshold=0.5)


def _probabilities(*values: float) -> np.ndarray:
    assert len(values) == 10
    return np.asarray(values, dtype=float)


def test_compare_study_predictions_consistent() -> None:
    probabilities = _probabilities(0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.05)
    table = SubmissionTable(
        probabilities={"study_a": {index: float(value) for index, value in enumerate(probabilities, 1)}}
    )
    report = compare_study_predictions(
        {"study_a": probabilities},
        table,
        threshold=0.5,
        image_counts={"study_a": 2},
    )
    assert report.consistent
    assert report.checked_studies == 1
    assert report.checked_images == 2
    assert report.max_abs_diff == pytest.approx(0.0)


def test_compare_study_predictions_detects_probability_mismatch() -> None:
    probabilities = _probabilities(0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.05)
    table = SubmissionTable(
        probabilities={
            "study_a": {index: float(value) for index, value in enumerate(probabilities, 1)}
        }
    )
    # 改掉一个概率，超过默认容差。
    table.probabilities["study_a"][3] = 0.95
    report = compare_study_predictions({"study_a": probabilities}, table, threshold=0.5)
    assert not report.consistent
    assert report.inconsistent_studies == 1
    assert "最大绝对差" in report.inconsistent[0].reason


def test_compare_study_predictions_detects_missing_study() -> None:
    probabilities = _probabilities(0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.05)
    table = SubmissionTable(probabilities={"study_other": {1: 0.5}})
    report = compare_study_predictions(
        {"study_a": probabilities},
        table,
        threshold=0.5,
        image_counts={"study_a": 3},
    )
    assert report.inconsistent_images == 3
    assert "缺少" in report.inconsistent[0].reason


def test_compare_study_predictions_filtered_submission() -> None:
    probabilities = _probabilities(0.9, 0.1, 0.7, 0.2, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05)
    table = SubmissionTable(probabilities={"study_a": {1: 0.9, 3: 0.7}})
    report = compare_study_predictions(
        {"study_a": probabilities}, table, threshold=0.5, image_counts={"study_a": 1}
    )
    assert report.consistent
    assert report.filtered_submission

    mismatched = SubmissionTable(probabilities={"study_a": {1: 0.9, 4: 0.7}})
    report = compare_study_predictions(
        {"study_a": probabilities}, mismatched, threshold=0.5, image_counts={"study_a": 1}
    )
    assert not report.consistent
    assert "类别集合不一致" in report.inconsistent[0].reason


# --------------------------------------------------------------------------
# 计时汇总与结论
# --------------------------------------------------------------------------


def test_summarize_timings_math() -> None:
    rounds = [[10.0, 20.0, 30.0], [12.0, 22.0, 32.0], [14.0, 24.0, 34.0]]
    summary = summarize_timings(rounds)
    assert summary.images == 3
    assert summary.rounds == 3
    assert summary.per_image_mean_ms == pytest.approx((12.0, 22.0, 32.0))
    assert summary.mean_ms == pytest.approx(22.0)
    assert summary.median_ms == pytest.approx(22.0)
    assert summary.min_ms == pytest.approx(12.0)
    assert summary.max_ms == pytest.approx(32.0)
    assert summary.total_ms == pytest.approx(sum(map(sum, rounds)))
    assert summary.within_limit


def test_summarize_timings_limit() -> None:
    summary = summarize_timings([[COST_LIMIT_MS + 1.0]], limit_ms=COST_LIMIT_MS)
    assert not summary.within_limit


def test_summarize_timings_rejects_ragged_rounds() -> None:
    with pytest.raises(ValueError, match="图片数不一致"):
        summarize_timings([[1.0, 2.0], [1.0]])


def test_format_conclusion_mentions_rule_fields() -> None:
    timing = summarize_timings([[10.0, 20.0], [12.0, 22.0], [14.0, 24.0]])
    table = SubmissionTable(probabilities={"study_a": {1: 0.9}})
    # 只有 class 1 过 0.5，和 table 里的 {1: 0.9} 对得上。
    probabilities = _probabilities(0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
    comparison = compare_study_predictions(
        {"study_a": probabilities}, table, threshold=0.5, image_counts={"study_a": 2}
    )
    conclusion = format_conclusion(timing, comparison, precision="fp16", warmup_images=5)
    assert conclusion.startswith("[CONCLUSION] 结论：2 张样本")
    assert "连续 3 轮" in conclusion
    assert "未超过上限" in conclusion
    assert "不一致 0 张" in conclusion
    assert "推理精度 fp16" in conclusion


# --------------------------------------------------------------------------
# 推理函数口径一致
# --------------------------------------------------------------------------


def test_predict_image_matches_predict_study_single_image(env: dict) -> None:
    model = load_checkpoint_model(env["config"], env["checkpoint"], env["device"])
    transform = env["transform"]
    single = env["images"] / IMAGE_RELATIVES[2]  # study_00001 只有这一张图
    image_probabilities = predict_image(model, single, transform, env["device"])
    study_probabilities = predict_study(model, [single], transform, env["device"])
    np.testing.assert_allclose(image_probabilities, study_probabilities, atol=1e-6)


def test_predict_study_matches_predict_loader(env: dict) -> None:
    model = load_checkpoint_model(env["config"], env["checkpoint"], env["device"])
    transform = env["transform"]
    multi = [env["images"] / relative for relative in IMAGE_RELATIVES[:2]]
    study_probabilities = predict_study(model, multi, transform, env["device"])

    row = env["predictions"].loc[env["predictions"]["Study_id"] == "study_00000"].iloc[0]
    expected = np.array([row[f"p{class_id}"] for class_id in range(1, 11)], dtype=float)
    np.testing.assert_allclose(study_probabilities, expected, atol=1e-6)


# --------------------------------------------------------------------------
# cost.py CLI 端到端
# --------------------------------------------------------------------------


def _cli_arguments(env: dict, submission: Path | None = None) -> list[str]:
    return [
        str(submission or env["submission"]),
        str(env["images"]),
        str(env["warmup"]),
        "--config",
        str(env["config_path"]),
        "--checkpoint",
        str(env["checkpoint"]),
    ]


def test_main_end_to_end(env: dict, capsys: pytest.CaptureFixture[str]) -> None:
    assert cost.main(_cli_arguments(env)) == 0
    output = capsys.readouterr().out
    assert "[CONCLUSION]" in output
    assert "未超过上限" in output
    assert "不一致 0 张" in output
    assert "推理精度 fp32" in output
    assert "[INFO] 预热完成：5 张图" in output


def test_main_writes_report(env: dict, tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    assert cost.main(_cli_arguments(env) + ["--report", str(report_path)]) == 0
    payload = report_path.read_text(encoding="utf-8")
    assert '"within_limit": true' in payload
    assert '"inconsistent_studies": 0' in payload


def test_main_detects_inconsistent_submission(env: dict, tmp_path: Path) -> None:
    frame = pd.read_csv(env["submission"], dtype=str, keep_default_na=False)
    frame.loc[0, "Probability"] = "0.0"
    tampered = tmp_path / "tampered.csv"
    frame.to_csv(tampered, index=False)
    assert cost.main(_cli_arguments(env, tampered)) == 1


def test_main_detects_slow_inference(
    env: dict, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cost, "_time_one_image", lambda *args, **kwargs: 250.0)
    assert cost.main(_cli_arguments(env)) == 1
    assert "超过上限" in capsys.readouterr().err


def test_main_missing_checkpoint_returns_2(env: dict, tmp_path: Path) -> None:
    arguments = _cli_arguments(env)
    arguments[arguments.index("--checkpoint") + 1] = str(tmp_path / "no_such.pt")
    assert cost.main(arguments) == 2


def test_resolve_checkpoint_prefers_submission_directory(env: dict) -> None:
    # submission.csv 与 best.pt 同在 base 下，不传 --checkpoint 应该能找到它。
    resolved = cost.resolve_checkpoint(None, env["submission"])
    assert resolved == env["checkpoint"]
