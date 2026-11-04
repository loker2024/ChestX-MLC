"""cost.py 与 infer.profiling 的测试：study 级计时、融合、submission 比对。

2026-11-03 官方答疑把最小计时单元从「单张图片」改为「单个 study_id」：
`batch=1` 以 study 为单位，一个 study 的多张图可以同时处理。这里围绕新口径
测四件可以稳定复现的事：

1. 目录/图片按 study 分组、视角元数据解析、submission 解析的边界条件；
2. `predict_study` / `predict_study_images` / `predict_loader` 的概率口径一致；
3. 计时统计按「每个 study 3 轮均值 → 全部 study 平均」计算，并计入轮级 IO；
4. `cost.main` 端到端跑通：单模型、多模型融合、清单自动发现、预热不相交、
   初始化打印、超限与不一致的退出码。

测试全部用合成的小图与随机初始化的 resnet18（image.size=32，CPU），
不下载预训练权重、不碰真实影像。
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from shutil import copyfile, copytree

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
# conftest.py 已将 src/ 加入模块搜索路径。
import cost  # noqa: E402

from common.checkpoint import save_checkpoint  # noqa: E402
from common.config import load_config  # noqa: E402
from data.dataset import build_loader  # noqa: E402
from infer.ensemble import fuse_probability_vectors  # noqa: E402
from infer.predict import (  # noqa: E402
    StudyModelSpec,
    build_eval_inference_transform,
    load_checkpoint_model,
    predict_image,
    predict_loader,
    predict_study,
    predict_study_images,
    predict_study_multi,
)
from infer.profiling import (  # noqa: E402
    COST_LIMIT_MS,
    SubmissionTable,
    build_study_jobs,
    compare_study_predictions,
    format_conclusion,
    group_images_by_study,
    list_images,
    parse_image_identity,
    read_submission,
    read_view_map,
    select_prediction_rows,
    summarize_timings,
    views_for_job,
)
from infer.submission import build_submission_frame, write_submission  # noqa: E402
from model.network import build_model  # noqa: E402


def test_cost_entrypoints_work_from_repository_and_submission_package(tmp_path: Path) -> None:
    """仓库入口与提交包根目录入口都能启动，并从各自的 src 导入依赖。"""
    source = ROOT / "src" / "cost.py"
    assert source.is_file()
    package_src = tmp_path / "src"
    package_src.symlink_to(ROOT / "src", target_is_directory=True)
    package_cost = tmp_path / "cost.py"
    package_cost.write_bytes(source.read_bytes())
    for script, cwd in ((source, ROOT), (package_cost, tmp_path)):
        result = subprocess.run(
            [sys.executable, str(script), "--help"], cwd=cwd,
            capture_output=True, text=True, check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "pictures_100" in result.stdout
        assert "--fusion" in result.stdout


# 测试目录里图片的摆放：2 个 study、3 张图，study_00000 有两张（验证多图同批）。
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
    两个随机初始化 resnet18 checkpoint（用于多模型融合），以及分别用单模型 /
    双模型概率平均生成的 submission.csv。
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
    torch.manual_seed(1)
    model2 = build_model(config)
    checkpoint2_path = save_checkpoint(
        base / "best2.pt", model2, epoch=1, metrics={"macro_auc": 0.5}, config=config, seed=1
    )

    device = torch.device("cpu")
    loader = build_loader(config, "test")
    predictions = predict_loader(model, loader, device)
    predictions2 = predict_loader(model2, loader, device)
    study_subjects = {record.study_id: record.subject_id for record in loader.dataset.records}
    submission_frame = build_submission_frame(predictions, study_subjects)
    submission_path = write_submission(submission_frame, base / "submission.csv")

    probability_columns = [f"p{class_id}" for class_id in range(1, 11)]
    fused_rows = []
    for record in predictions.itertuples(index=False):
        study_id = str(record.Study_id)
        first = np.asarray([getattr(record, column) for column in probability_columns], dtype=float)
        second_row = predictions2.loc[predictions2["Study_id"] == study_id].iloc[0]
        second = np.asarray([second_row[column] for column in probability_columns], dtype=float)
        fused_rows.append([study_id, *fuse_probability_vectors([first, second], [1, 1], "probability")])
    fused_frame = pd.DataFrame(fused_rows, columns=["Study_id", *probability_columns])
    ensemble_submission = write_submission(
        build_submission_frame(fused_frame, study_subjects),
        base / "submission_ensemble.csv",
    )

    return {
        "base": base,
        "config": config,
        "config_path": config_path,
        "checkpoint": checkpoint_path,
        "checkpoint2": checkpoint2_path,
        "images": images_root,
        "warmup": warmup_root,
        "submission": submission_path,
        "submission_ensemble": ensemble_submission,
        "model": model,
        "model2": model2,
        "device": device,
        "loaders": loader,
        "predictions": predictions,
        "transform": build_eval_inference_transform(config),
    }


# --------------------------------------------------------------------------
# 图片发现、study 分组与视角元数据
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


def test_build_study_jobs_groups_multi_image_study(env: dict) -> None:
    jobs = build_study_jobs(list_images(env["images"]))
    assert [job.study_id for job in jobs] == ["study_00000", "study_00001"]
    assert jobs[0].subject_id == "subject_00000"
    assert jobs[0].image_count == 2
    assert [path.name for path in jobs[0].images] == ["pic_000010.png", "pic_000011.png"]
    assert jobs[1].image_count == 1


def test_read_view_map_and_views_for_job(env: dict, tmp_path: Path) -> None:
    path = tmp_path / "cost_views.csv"
    _write_csv(
        path,
        ["Study_id", "Dicom_id", "ViewPosition"],
        [
            ["study_00000", "pic_000010", "AP"],
            ["study_00000", "pic_000011", "LATERAL"],
            ["study_00001", "pic_000012", "PA"],
        ],
    )
    view_map = read_view_map(path)
    jobs = build_study_jobs(list_images(env["images"]))
    assert views_for_job(jobs[0], view_map) == (0, 1)
    assert views_for_job(jobs[1], view_map) == (0,)

    # 缺一张图就返回 None，避免静默把未知视角混进加权。
    partial = {"study_00000": {"pic_000010": 0}}
    assert views_for_job(jobs[0], partial) is None


def test_read_view_map_rejects_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    _write_csv(path, ["Study_id", "Dicom_id"], [["study_1", "pic_1"]])
    with pytest.raises(ValueError, match="缺少视角元数据列"):
        read_view_map(path)


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


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1])
def test_compare_rejects_invalid_live_probabilities(bad: float) -> None:
    table = SubmissionTable(probabilities={"study_a": {i: 0.5 for i in range(1, 11)}})
    values = np.full(10, 0.5)
    values[0] = bad
    with pytest.raises(ValueError, match="概率"):
        compare_study_predictions({"study_a": values}, table, threshold=0.5)


def test_read_submission_rejects_conflicting_subjects(tmp_path: Path) -> None:
    path = _write_submission(tmp_path / "bad.csv", [["a", "s", "1", "0.5"], ["b", "s", "2", "0.5"]],
                             ["Subject_id", "Study_id", "Predict_class", "Probability"])
    with pytest.raises(ValueError, match="Subject_id"):
        read_submission(path)


def test_time_one_study_writes_one_row_per_image(env: dict, tmp_path: Path) -> None:
    """计时单元是 study：2 张图的 study 写 2 行，同一 study 概率相同。"""
    jobs = build_study_jobs(list_images(env["images"]))
    loaded = [
        cost.LoadedModel(
            cost.ModelSpec(env["checkpoint"], env["config"]),
            env["model"],
            env["transform"],
            ("single",),
        )
    ]
    output = tmp_path / "prediction.csv"
    handle = output.open("w", encoding="utf-8", newline="")
    try:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(cost.COST_OUTPUT_COLUMNS)
        elapsed, probabilities = cost._time_one_study(
            loaded, jobs[0], None, env["device"], None, "probability", 1, writer, handle
        )
        handle.flush()
    finally:
        handle.close()
    assert elapsed >= 0
    rows = list(csv.DictReader(output.open(encoding="utf-8", newline="")))
    assert len(rows) == 2
    assert {row["Image_id"] for row in rows} == {"pic_000010", "pic_000011"}
    assert all(row["Study_id"] == "study_00000" for row in rows)
    assert all(row["Subject_id"] == "subject_00000" for row in rows)
    # 同一 study 的所有图片写同一组最终概率。
    assert rows[0]["p1"] == rows[1]["p1"]
    assert float(rows[0]["p1"]) == pytest.approx(float(probabilities[0]), abs=1e-6)


def test_fuse_probability_vectors_probability_space() -> None:
    first = np.full(10, 0.2)
    second = np.full(10, 0.8)
    fused = fuse_probability_vectors([first, second], [1, 1], "probability")
    np.testing.assert_allclose(fused, np.full(10, 0.5))


def test_fuse_probability_vectors_logit_space() -> None:
    first = np.full(10, 0.2)
    second = np.full(10, 0.8)
    fused = fuse_probability_vectors([first, second], [1, 3], "logit")
    logits = 0.25 * np.log(0.2 / 0.8) + 0.75 * np.log(0.8 / 0.2)
    expected = 1.0 / (1.0 + np.exp(-logits))
    np.testing.assert_allclose(fused, np.full(10, expected))


def test_fuse_probability_vectors_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="维度必须是 10"):
        fuse_probability_vectors([np.zeros(9)], None, "probability")


def _probabilities(*values: float) -> np.ndarray:
    assert len(values) == 10
    return np.asarray(values, dtype=float)


def test_fuse_study_passes_views_only_to_view_weighted_models(
    env: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mean 聚合模型不传 views，view_weighted 模型才传，避免改错聚合口径。"""
    mean_config = env["config"]
    view_config = replace(
        mean_config,
        model=replace(mean_config.model, aggregation="view_weighted", frontal_weight=0.7),
    )
    loaded = [
        cost.LoadedModel(
            cost.ModelSpec(env["checkpoint"], mean_config),
            env["model"],
            env["transform"],
            ("mean",),
        ),
        cost.LoadedModel(
            cost.ModelSpec(env["checkpoint"], view_config),
            env["model2"],
            env["transform"],
            ("view",),
        ),
    ]
    captured: list[tuple[str, bool, tuple[int, ...] | None]] = []

    def fake(specs, image_paths, device, views=None, max_workers=4, executor=None):
        captured.extend((spec.transform_key[0], spec.uses_views, views) for spec in specs)
        return [np.full(10, 0.5) for _ in specs]

    monkeypatch.setattr(cost, "predict_study_multi", fake)
    cost._fuse_study(
        loaded, [Path("dummy.jpg")], (0, 1), env["device"], [1, 1], "probability", 1
    )
    assert captured == [("mean", False, (0, 1)), ("view", True, (0, 1))]


def test_predict_study_multi_shares_transform_and_matches_serial(env: dict) -> None:
    """同一 transform_key 只做一遍预处理；结果与逐模型串行推理一致。"""
    model1 = load_checkpoint_model(env["config"], env["checkpoint"], env["device"])
    model2 = load_checkpoint_model(env["config"], env["checkpoint2"], env["device"])
    paths = [env["images"] / relative for relative in IMAGE_RELATIVES[:2]]
    calls: list[int] = []

    def counting(image):
        calls.append(1)
        return env["transform"](image)

    specs = [
        StudyModelSpec(model1, counting, ("320",), uses_views=False),
        StudyModelSpec(model2, counting, ("320",), uses_views=False),
    ]
    results = predict_study_multi(specs, paths, env["device"], max_workers=2)

    # 2 张图 × 1 种预处理；若没共享会是 4 次。
    assert len(calls) == len(paths)
    expected = [
        predict_study(model1, paths, env["transform"], env["device"]),
        predict_study(model2, paths, env["transform"], env["device"]),
    ]
    for got, want in zip(results, expected):
        np.testing.assert_allclose(got, want, atol=1e-6)


def test_predict_study_multi_missing_transform_key_does_not_share(
    env: dict,
) -> None:
    """缺少可比较的 key 时，每个模型都必须独立执行预处理。"""
    paths = [env["images"] / relative for relative in IMAGE_RELATIVES[:2]]
    calls = [0, 0]

    def make_transform(index: int):
        def transform(image):
            calls[index] += 1
            return env["transform"](image)

        return transform

    specs = [
        StudyModelSpec(env["model"], make_transform(0), uses_views=False),
        StudyModelSpec(env["model2"], make_transform(1), uses_views=False),
    ]
    predict_study_multi(specs, paths, env["device"], max_workers=1)

    assert calls == [len(paths), len(paths)]


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
    summary = summarize_timings(rounds, images_per_study=[1, 2, 3])
    assert summary.studies == 3
    assert summary.images == 6
    assert summary.rounds == 3
    assert summary.per_study_mean_ms == pytest.approx((12.0, 22.0, 32.0))
    assert summary.mean_ms == pytest.approx(22.0)
    assert summary.median_ms == pytest.approx(22.0)
    assert summary.min_ms == pytest.approx(12.0)
    assert summary.max_ms == pytest.approx(32.0)
    assert summary.total_ms == pytest.approx(sum(map(sum, rounds)))
    assert summary.within_limit


def test_summarize_timings_bills_round_overhead() -> None:
    rounds = [[10.0, 20.0], [10.0, 20.0], [10.0, 20.0]]
    # 每轮总墙钟比逐 study 计时之和多 6 ms（文件打开/关闭等）。
    summary = summarize_timings(
        rounds, images_per_study=[1, 2], round_total_ms=[36.0, 36.0, 36.0]
    )
    assert summary.total_ms == pytest.approx(90.0)
    assert summary.round_overhead_per_study_ms == pytest.approx(3.0)
    assert summary.billed_mean_ms == pytest.approx(summary.mean_ms + 3.0)


def test_summarize_timings_limit() -> None:
    summary = summarize_timings([[COST_LIMIT_MS + 1.0]], limit_ms=COST_LIMIT_MS)
    assert not summary.within_limit


def test_summarize_timings_rejects_ragged_rounds() -> None:
    with pytest.raises(ValueError, match="study 数不一致"):
        summarize_timings([[1.0, 2.0], [1.0]])


def test_format_conclusion_mentions_rule_fields() -> None:
    timing = summarize_timings([[10.0, 20.0], [12.0, 22.0], [14.0, 24.0]])
    table = SubmissionTable(probabilities={"study_a": {1: 0.9}})
    # 只有 class 1 过 0.5，和 table 里的 {1: 0.9} 对得上。
    probabilities = _probabilities(0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
    comparison = compare_study_predictions(
        {"study_a": probabilities}, table, threshold=0.5, image_counts={"study_a": 2}
    )
    conclusion = format_conclusion(
        timing,
        comparison,
        precision="fp16",
        warmup_studies=5,
        warmup_images=10,
        init_ms=12.3,
        model_count=2,
        fusion="probability",
    )
    assert conclusion.startswith("[CONCLUSION] 结论：2 个 study（2 张图片）")
    assert "连续 3 轮" in conclusion
    assert "未超过上限" in conclusion
    assert "折算" in conclusion and "ms/图" in conclusion
    assert "不一致 0 个" in conclusion
    assert "推理精度 fp16" in conclusion
    assert "模型 2 个（probability 融合）" in conclusion
    assert "预热 5 个 study（10 张图片" in conclusion
    assert "初始化 12.3 ms" in conclusion


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


def test_predict_study_images_matches_predict_study(env: dict) -> None:
    """解码一次、多图同批的入口与按路径读图的入口数值一致。"""
    from data.dataset import load_image

    model = load_checkpoint_model(env["config"], env["checkpoint"], env["device"])
    paths = [env["images"] / relative for relative in IMAGE_RELATIVES[:2]]
    images = [load_image(path, convert_rgb=False) for path in paths]
    from_images = predict_study_images(model, images, env["transform"], env["device"])
    from_paths = predict_study(model, paths, env["transform"], env["device"])
    np.testing.assert_allclose(from_images, from_paths, atol=1e-6)


# --------------------------------------------------------------------------
# cost.py CLI 端到端
# --------------------------------------------------------------------------


def _cli_arguments(
    env: dict,
    submission: Path | None = None,
    checkpoints: list[Path] | None = None,
    with_config: bool = True,
) -> list[str]:
    arguments = [
        str(submission or env["submission"]),
        str(env["images"]),
        str(env["warmup"]),
        "--checkpoint",
        *[str(path) for path in (checkpoints or [env["checkpoint"]])],
    ]
    if with_config:
        arguments += ["--config", str(env["config_path"])]
    return arguments


def _single_image_args(env: dict, tmp_path: Path) -> list[str]:
    sample = tmp_path / IMAGE_RELATIVES[2]
    sample.parent.mkdir(parents=True)
    copyfile(env["images"] / IMAGE_RELATIVES[2], sample)
    arguments = _cli_arguments(env)
    arguments[1] = str(tmp_path)
    return arguments


def test_main_end_to_end(env: dict, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cost.main(_single_image_args(env, tmp_path)) == 0
    output = capsys.readouterr().out
    assert "[CONCLUSION]" in output
    assert "未超过上限" in output
    assert "不一致 0 个" in output
    assert "推理精度 fp32" in output
    assert "[INFO] 初始化（仅元数据，不计时）" in output
    assert "[INFO] 预热完成：5 个 study / 5 张图" in output


def test_main_times_multi_image_study(env: dict, capsys: pytest.CaptureFixture[str]) -> None:
    """含多图 study 不再返回 2：整批前向并按 study 计时。"""
    assert cost.main(_cli_arguments(env)) == 0
    output = capsys.readouterr().out
    assert "2 个 study（3 张图片）" in output
    assert "未超过上限" in output


def test_main_writes_report(env: dict, tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    assert cost.main(_single_image_args(env, tmp_path) + ["--report", str(report_path)]) == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["timing_ms"]["within_limit"] is True
    assert payload["timing_ms"]["studies"] == 1
    assert payload["comparison"]["inconsistent_studies"] == 0
    assert payload["model_count"] == 1
    assert payload["init_within_limit"] is True


def test_main_detects_inconsistent_submission(env: dict, tmp_path: Path) -> None:
    frame = pd.read_csv(env["submission"], dtype=str, keep_default_na=False)
    frame.loc[0, "Probability"] = "0.0"
    tampered = tmp_path / "tampered.csv"
    frame.to_csv(tampered, index=False)
    assert cost.main(_cli_arguments(env, tampered)) == 1


def test_main_detects_slow_inference(
    env: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    original = cost._time_one_study

    def slow(*args, **kwargs):
        elapsed, probabilities = original(*args, **kwargs)
        return elapsed + 250.0, probabilities

    monkeypatch.setattr(cost, "_time_one_study", slow)
    assert cost.main(_single_image_args(env, tmp_path)) == 1
    assert "超过上限" in capsys.readouterr().err


def test_main_matches_two_model_ensemble(
    env: dict, capsys: pytest.CaptureFixture[str]
) -> None:
    """submission 是双模型概率平均时，cost.py 必须同时加载两个模型再融合。"""
    arguments = _cli_arguments(
        env,
        env["submission_ensemble"],
        [env["checkpoint"], env["checkpoint2"]],
    )
    arguments += ["--weights", "1,1", "--fusion", "probability"]
    assert cost.main(arguments) == 0
    output = capsys.readouterr().out
    assert "模型 2 个（probability 融合）" in output
    assert "不一致 0 个" in output


def test_main_reads_ensemble_manifest(env: dict, tmp_path: Path) -> None:
    """不传 --checkpoint 时，submission 同目录的 cost_ensemble.json 自动生效。"""
    submission = tmp_path / "submission.csv"
    copyfile(env["submission_ensemble"], submission)
    (tmp_path / "cost_ensemble.json").write_text(
        json.dumps(
            {
                "fusion": "probability",
                "weights": [1, 1],
                "checkpoints": [str(env["checkpoint"]), str(env["checkpoint2"])],
            }
        ),
        encoding="utf-8",
    )
    arguments = [str(submission), str(env["images"]), str(env["warmup"])]
    assert cost.main(arguments) == 0


def test_main_manifest_relative_checkpoints(env: dict, tmp_path: Path) -> None:
    """清单里的相对路径按清单所在目录解析。"""
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    copyfile(env["checkpoint"], weights_dir / "a.pt")
    copyfile(env["checkpoint2"], weights_dir / "b.pt")
    submission = tmp_path / "submission.csv"
    copyfile(env["submission_ensemble"], submission)
    (tmp_path / "cost_ensemble.json").write_text(
        json.dumps(
            {
                "fusion": "probability",
                "weights": [1, 1],
                "checkpoints": ["weights/a.pt", "weights/b.pt"],
            }
        ),
        encoding="utf-8",
    )
    arguments = [str(submission), str(env["images"]), str(env["warmup"])]
    assert cost.main(arguments) == 0


def test_main_rejects_warmup_overlap(env: dict, tmp_path: Path) -> None:
    """预热 study 与计时 study 必须不相交。"""
    warmup = tmp_path / "warmup"
    copytree(env["images"] / "subject_00001", warmup / "subject_00001")
    arguments = _cli_arguments(env)
    arguments[2] = str(warmup)
    assert cost.main(arguments) == 2


def test_main_works_without_test_csv(
    env: dict, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """评测环境没有 test.csv 也能跑：计时只依赖目录结构与 submission.csv。"""
    raw = yaml.safe_load(env["config_path"].read_text(encoding="utf-8"))
    raw["data"]["test"]["csv"] = str(tmp_path / "no_such.csv")
    raw["data"]["test"]["images"] = str(tmp_path / "no_such_images")
    config_path = tmp_path / "no_index.yaml"
    config_path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")

    arguments = _single_image_args(env, tmp_path)
    arguments[arguments.index("--config") + 1] = str(config_path)
    assert cost.main(arguments) == 0
    output = capsys.readouterr().out
    assert "未超过上限" in output
    assert "视角元数据：" not in output


def test_main_missing_checkpoint_returns_2(env: dict, tmp_path: Path) -> None:
    arguments = _cli_arguments(env)
    arguments[arguments.index("--checkpoint") + 1] = str(tmp_path / "no_such.pt")
    assert cost.main(arguments) == 2


def test_resolve_checkpoint_prefers_submission_directory(env: dict) -> None:
    # submission.csv 与 best.pt 同在 base 下，不传 --checkpoint 应该能找到它。
    resolved = cost.resolve_checkpoint(None, env["submission"])
    assert resolved == env["checkpoint"]
