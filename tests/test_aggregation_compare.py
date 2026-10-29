"""固定 checkpoint 的 logits / 概率聚合对照：数值、缓存、报告与 CLI。

G0 = sigmoid(mean(logits))，G1 = mean(sigmoid(logits))。这里的用例全部用
合成小图与随机权重，不读真实影像、不下载预训练权重。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

import run_aggregation_compare
from common.checkpoint import save_checkpoint
from common.config import load_config
from infer import aggregation_compare
from infer.aggregation_compare import (
    CACHE_FORMAT_VERSION,
    IMAGE_LOGITS_FILE,
    METADATA_FILE,
    _align_frame,
    aggregate_pair,
    compare_cache,
    export_run,
    load_cache,
)
from model.network import build_model

PROBABILITY_COLUMNS = [f"p{index}" for index in range(1, 11)]


# ---------------------------------------------------------------------------
# 合成 run 目录
# ---------------------------------------------------------------------------


def _write_run(run_root, raw_config: dict, name: str = "run") -> Path:
    """按原始配置造一个最小 run 目录：config.yaml + 随机权重 best.pt。"""
    run_dir = run_root / name
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(raw_config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    config = load_config(run_dir / "config.yaml")
    model = build_model(config)  # 测试配置 pretrained=False，不会联网
    save_checkpoint(
        run_dir / "best.pt",
        model,
        epoch=3,
        metrics={"valid_macro_auc": 0.5},
        config=config,
        seed=raw_config.get("seed", 42),
        metadata={"pretrained_loaded": False, "device": "cpu"},
    )
    return run_dir


def _config_raw(config_path, seed: int = 42) -> dict:
    """读回 write_config 写出的 YAML，补上测试需要的 model / seed 覆盖。"""
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["model"] = {"pretrained": False}
    raw["seed"] = seed
    return raw


@pytest.fixture
def run_dir(tmp_path, labeled_split, write_config):
    """一个只有 2 个 study（3 张图）的合成 run。"""
    config_path = write_config(
        {
            "train": (labeled_split["csv"], labeled_split["images"]),
            "valid": (labeled_split["csv"], labeled_split["images"]),
            "test": (labeled_split["csv"], labeled_split["images"]),
        }
    )
    return _write_run(tmp_path, _config_raw(config_path))


def _read_npz(cache_dir):
    with np.load(cache_dir / IMAGE_LOGITS_FILE, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


# ---------------------------------------------------------------------------
# 核心数值
# ---------------------------------------------------------------------------


def test_aggregate_pair_changes_ranking_and_preserves_single_image():
    logits = torch.tensor([[8.0], [-2.0], [2.0], [2.0], [0.0]]).repeat(1, 10)
    g0, g1 = aggregate_pair(logits, torch.tensor([0, 2, 4, 5]))
    torch.testing.assert_close(g0[:, 0], torch.tensor([0.9525741, 0.8807971, 0.5]))
    torch.testing.assert_close(g1[:, 0], torch.tensor([0.5594338, 0.8807971, 0.5]))
    assert g0[0, 0] > g0[1, 0]
    assert g1[0, 0] < g1[1, 0]
    assert torch.equal(g0[2], g1[2])


def test_aggregate_pair_single_image_and_equal_logits_are_identical():
    # 同一 study 内每张图 logits 相同 => 均值就是 logits 本身，G0 与 G1 必然一致。
    logits = torch.tensor([[1.0] * 10, [-3.0] * 10, [-3.0] * 10])
    g0, g1 = aggregate_pair(logits, torch.tensor([0, 1, 3]))

    torch.testing.assert_close(g0, g1)
    assert g0.shape == (2, 10)
    assert g0.dtype == torch.float32
    assert g1.dtype == torch.float32


def test_aggregate_pair_rejects_invalid_inputs():
    logits = torch.zeros(3, 10)
    with pytest.raises(ValueError, match="首项"):
        aggregate_pair(logits, torch.tensor([1, 3]))
    with pytest.raises(ValueError, match="严格递增"):
        aggregate_pair(logits, torch.tensor([0, 2, 2, 3]))
    with pytest.raises(ValueError, match="末项"):
        aggregate_pair(logits, torch.tensor([0, 1, 2]))
    with pytest.raises(ValueError, match="NaN"):
        aggregate_pair(torch.full((2, 10), float("nan")), torch.tensor([0, 2]))
    with pytest.raises(ValueError, match="10"):
        aggregate_pair(torch.zeros(2, 5), torch.tensor([0, 2]))
    with pytest.raises(TypeError, match="整数"):
        aggregate_pair(logits, torch.tensor([0.0, 3.0]))
    with pytest.raises(ValueError, match="二维"):
        aggregate_pair(torch.zeros(3), torch.tensor([0, 3]))


# ---------------------------------------------------------------------------
# export：缓存内容与基准校验
# ---------------------------------------------------------------------------


def test_export_writes_cache_and_verifies_g0(tmp_path, run_dir):
    cache = tmp_path / "cache"
    metadata = export_run(run_dir, cache, device="cpu")

    assert metadata["cache_format_version"] == CACHE_FORMAT_VERSION
    assert metadata["valid"]["studies"] == 2
    assert metadata["valid"]["images"] == 3
    assert metadata["baseline_check"]["max_abs_diff"] <= 1e-6
    assert metadata["num_batches"] == 1
    assert metadata["num_model_forwards"] == 1
    assert metadata["precision"] == "fp32"
    assert metadata["class_ids"] == list(range(1, 11))
    assert (cache / METADATA_FILE).is_file()

    arrays = _read_npz(cache)
    assert arrays["logits"].shape == (3, 10)
    assert arrays["offsets"].tolist() == [0, 2, 3]
    assert arrays["labels"].shape == (2, 10)
    assert arrays["study_ids"].tolist() == ["study_00000", "study_00001"]
    assert arrays["dicom_ids"].tolist() == ["pic_000010", "pic_000011", "pic_000012"]
    assert arrays["views"].tolist() == [0, 1, 0]  # AP / LATERAL / AP

    # NPZ 哈希写进 metadata，并可作为独立指纹复算。
    assert metadata["npz_sha256"] == aggregation_compare.sha256_file(
        cache / IMAGE_LOGITS_FILE
    )
    assert len(metadata["images"]) == 3
    assert metadata["images"][0]["dicom_id"] == "pic_000010"
    for entry in metadata["images"]:
        assert entry["sha256"] == aggregation_compare.sha256_file(Path(entry["path"]))


def test_export_calls_model_once_per_batch(tmp_path, run_dir, monkeypatch):
    """导出循环每批只前向一次；predict_loader 基准是额外一遍（另计）。"""
    real_loader_model = aggregation_compare.load_checkpoint_model
    forward_calls = {"count": 0}

    class CountingModel(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, images):
            forward_calls["count"] += 1
            return self.inner(images)

    def fake_load(config, checkpoint_path, device):
        return CountingModel(real_loader_model(config, checkpoint_path, device))

    monkeypatch.setattr(aggregation_compare, "load_checkpoint_model", fake_load)
    metadata = export_run(run_dir, tmp_path / "cache", device="cpu")

    assert metadata["num_model_forwards"] == metadata["num_batches"] == 1
    # 导出 1 次 + 基准 1 次，共 2 次；不会为 G1 再跑模型。
    assert forward_calls["count"] == 2


def test_export_falls_back_to_run_config(tmp_path, run_dir):
    checkpoint_path = run_dir / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint.pop("config")
    torch.save(checkpoint, checkpoint_path)

    metadata = export_run(run_dir, tmp_path / "cache", device="cpu")

    assert metadata["run"]["config_source"].startswith("config.yaml")


def test_export_rejects_non_mean_aggregation(tmp_path, run_dir):
    checkpoint_path = run_dir / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["config"]["model"]["aggregation"] = "view_weighted"
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(ValueError, match="mean"):
        export_run(run_dir, tmp_path / "cache", device="cpu")


def test_export_rejects_missing_weights(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="best.pt"):
        export_run(empty, tmp_path / "cache", device="cpu")


def test_export_refuses_non_empty_output_dir(tmp_path, run_dir):
    out = tmp_path / "cache"
    out.mkdir()
    (out / "stale.txt").write_text("old", encoding="utf-8")
    with pytest.raises(FileExistsError, match="非空"):
        export_run(run_dir, out, device="cpu")


def test_export_stops_when_baseline_mismatches(tmp_path, run_dir, monkeypatch):
    def fake_predict(model, loader, device, config=None):
        frame = pd.DataFrame(0.5, index=range(2), columns=PROBABILITY_COLUMNS)
        frame.insert(0, "Study_id", ["study_00000", "study_00001"])
        return frame

    monkeypatch.setattr(aggregation_compare, "predict_loader", fake_predict)
    with pytest.raises(RuntimeError, match="1e-6"):
        export_run(run_dir, tmp_path / "cache", device="cpu")


def test_export_handles_variable_length_and_unknown_view(
    tmp_path, make_image, write_csv, write_config
):
    entries = [
        ("pic_000010", "subject_00000", "study_00000", "AP"),
        ("pic_000011", "subject_00000", "study_00000", "UNKNOWN"),
        ("pic_000012", "subject_00001", "study_00001", "LATERAL"),
        ("pic_000013", "subject_00002", "study_00002", "AP"),
        ("pic_000014", "subject_00002", "study_00002", "LL"),
    ]
    rows = []
    for dicom, subject, study, view in entries:
        make_image(f"images/{subject}/{study}/{dicom}.png")
        rows.append([dicom, subject, study, view, "Erect", "48", "64", "1"])
    csv_path = write_csv("variable.csv", rows)
    images = tmp_path / "images"
    config_path = write_config(
        {"train": (csv_path, images), "valid": (csv_path, images), "test": (csv_path, images)}
    )
    run = _write_run(tmp_path, _config_raw(config_path))

    metadata = export_run(run, tmp_path / "cache", device="cpu")

    assert metadata["valid"]["studies"] == 3
    assert metadata["valid"]["images"] == 5
    arrays = _read_npz(tmp_path / "cache")
    assert arrays["offsets"].tolist() == [0, 2, 3, 5]
    assert arrays["views"].tolist() == [0, 2, 1, 0, 1]  # 未知视角记为 2


# ---------------------------------------------------------------------------
# compare：只读缓存与报告
# ---------------------------------------------------------------------------


def test_compare_writes_reports_and_preserves_ids(tmp_path, run_dir):
    cache = tmp_path / "cache"
    export_run(run_dir, cache, device="cpu")
    result = compare_cache(cache, [0.3, 0.5])

    g0 = pd.read_csv(
        cache / aggregation_compare.PREDICTIONS_G0_FILE,
        dtype={"Subject_id": str, "Study_id": str},
    )
    assert list(g0.columns) == ["Subject_id", "Study_id", *PROBABILITY_COLUMNS]
    assert g0["Study_id"].tolist() == ["study_00000", "study_00001"]
    assert g0["Subject_id"].tolist() == ["subject_00000", "subject_00001"]

    summary = pd.read_csv(cache / aggregation_compare.SUMMARY_CSV_FILE)
    assert summary["subset"].tolist() == ["all", "single", "multi"]
    assert summary.loc[summary["subset"] == "all", "studies"].item() == 2
    # 单图 study 的两种聚合必须完全一致（该子集只有 1 个 study，AUC 无定义）。
    g1 = pd.read_csv(
        cache / aggregation_compare.PREDICTIONS_G1_FILE,
        dtype={"Subject_id": str, "Study_id": str},
    )
    single_rows = g0["Study_id"] == "study_00001"
    np.testing.assert_allclose(
        g0.loc[single_rows, PROBABILITY_COLUMNS].to_numpy(),
        g1.loc[single_rows, PROBABILITY_COLUMNS].to_numpy(),
        atol=1e-6,
    )
    single = summary.loc[summary["subset"] == "single"].iloc[0]
    assert single["max_abs_diff"] <= 1e-6
    assert single["status"] == "ok"

    per_class = pd.read_csv(cache / aggregation_compare.PER_CLASS_FILE)
    assert len(per_class) == 10
    assert {"class_id", "support", "g0_auc", "g1_auc", "delta_auc"} <= set(per_class.columns)
    assert result["summary"].shape[0] == 3

    markdown = (cache / aggregation_compare.SUMMARY_MD_FILE).read_text(encoding="utf-8")
    assert "G0" in markdown and "G1" in markdown
    assert "判读提示" in markdown


def test_compare_does_not_load_model_or_images(tmp_path, run_dir, monkeypatch):
    cache = tmp_path / "cache"
    export_run(run_dir, cache, device="cpu")

    def forbidden(*args, **kwargs):
        raise AssertionError("compare 不应加载模型或重新构建 DataLoader")

    monkeypatch.setattr(aggregation_compare, "load_checkpoint_model", forbidden)
    monkeypatch.setattr(aggregation_compare, "build_loader", forbidden)
    monkeypatch.setattr(aggregation_compare, "predict_loader", forbidden)

    result = compare_cache(cache, [0.5])
    assert not result["summary"].empty


def test_compare_rejects_tampered_npz_hash(tmp_path, run_dir):
    cache = tmp_path / "cache"
    export_run(run_dir, cache, device="cpu")
    metadata_path = cache / METADATA_FILE
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["npz_sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256"):
        compare_cache(cache, [0.5])


def test_compare_rejects_missing_cache_files(tmp_path, run_dir):
    cache = tmp_path / "cache"
    export_run(run_dir, cache, device="cpu")
    (cache / IMAGE_LOGITS_FILE).unlink()
    with pytest.raises(FileNotFoundError, match=IMAGE_LOGITS_FILE):
        compare_cache(cache, [0.5])


def test_align_frame_rejects_ids_and_reorders():
    study_ids = ["study_00000", "study_00001"]
    shuffled = pd.DataFrame(
        {"Study_id": ["study_00001", "study_00000"], "p1": [1.0, 2.0]}
    )
    aligned = _align_frame(study_ids, shuffled, "预测")
    assert aligned["Study_id"].tolist() == study_ids
    assert aligned["p1"].tolist() == [2.0, 1.0]

    duplicated = pd.DataFrame({"Study_id": ["study_00000", "study_00000"], "p1": [1, 2]})
    with pytest.raises(ValueError, match="重复"):
        _align_frame(study_ids, duplicated, "预测")
    missing = pd.DataFrame({"Study_id": ["study_00000"], "p1": [1]})
    with pytest.raises(ValueError, match="不匹配"):
        _align_frame(study_ids, missing, "预测")
    extra = pd.DataFrame(
        {"Study_id": ["study_00000", "study_00001", "study_00002"], "p1": [1, 2, 3]}
    )
    with pytest.raises(ValueError, match="不匹配"):
        _align_frame(study_ids, extra, "预测")


def test_compare_marks_empty_subset(tmp_path, make_image, write_csv, write_config):
    """全部是单图 study 时，多图子集应标 empty，而不是填 0 冒充结果。"""
    rows = []
    for index in range(3):
        dicom = f"pic_0000{index}0"
        subject = f"subject_0000{index}"
        study = f"study_0000{index}"
        make_image(f"images/{subject}/{study}/{dicom}.png")
        rows.append([dicom, subject, study, "AP", "Erect", "48", "64", str(index + 1)])
    csv_path = write_csv("single.csv", rows)
    images = tmp_path / "images"
    config_path = write_config(
        {"train": (csv_path, images), "valid": (csv_path, images), "test": (csv_path, images)}
    )
    run = _write_run(tmp_path, _config_raw(config_path))
    cache = tmp_path / "cache"

    export_run(run, cache, device="cpu")
    compare_cache(cache, [0.3, 0.5])

    summary = pd.read_csv(cache / aggregation_compare.SUMMARY_CSV_FILE)
    multi = summary.loc[summary["subset"] == "multi"].iloc[0]
    assert multi["status"] == "empty"
    assert multi["studies"] == 0
    assert np.isnan(multi["delta_macro_auc"])
    assert "无结果" in (cache / aggregation_compare.SUMMARY_MD_FILE).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 缓存往返与 CLI
# ---------------------------------------------------------------------------


def test_cache_roundtrip_preserves_leading_zero_ids(tmp_path, run_dir):
    cache = tmp_path / "cache"
    export_run(run_dir, cache, device="cpu")

    cached = load_cache(cache)
    assert cached.study_ids == ["study_00000", "study_00001"]
    assert cached.subject_ids == ["subject_00000", "subject_00001"]
    assert cached.dicom_ids == ["pic_000010", "pic_000011", "pic_000012"]
    assert cached.offsets.tolist() == [0, 2, 3]
    assert cached.labels.shape == (2, 10)


def test_cli_export_and_compare_end_to_end(tmp_path, run_dir):
    cache = tmp_path / "cache"
    run_aggregation_compare.main(
        ["export", "--run", str(run_dir), "--device", "cpu", "--out", str(cache)]
    )
    assert (cache / METADATA_FILE).is_file()

    run_aggregation_compare.main(
        ["compare", "--cache", str(cache), "--thresholds", "0.3,0.5"]
    )
    assert (cache / aggregation_compare.SUMMARY_CSV_FILE).is_file()
    assert (cache / aggregation_compare.SUMMARY_MD_FILE).is_file()
