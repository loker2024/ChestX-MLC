"""次数先验离线研究 CLI；所有输入显式给出，test 研究始终拒绝。"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from data.labels import parse_label_string
from data.patient_counts import build_patient_counts, read_patient_metadata, align_patient_counts
from infer.patient_count_prior import PatientCountPrior, apply_prior_frame, adjust_probabilities, load_prior, save_prior, sha256_file
from infer.patient_count_prior import require_research_split
from infer.patient_count_analysis import BETA_GRID, choose_beta, metric_report, paired_bootstrap
from data.patient_folds import verify_manifest, verify_oof_receipts

LABELS = [f"label_{i}" for i in range(1, 11)]
PROBS = [f"p{i}" for i in range(1, 11)]
FIXED_CONFIG = {"cap": 10, "beta_grid": list(BETA_GRID), "probability_eps": 1e-6, "seed": 2026,
                "bootstrap": 2000, "min_delta": .002, "min_positive_folds": 4, "invalid_bootstrap_limit": .05}


def output_directory(path: str | Path) -> Path:
    out = Path(path)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"输出目录非空，拒绝覆盖: {out}")
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")


def source_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest_path.parent / path


def validate_oof(oof_path: str, manifest_path: str) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """核验来源、每 study 一次、患者不跨折及逐行标签。"""
    mp = Path(manifest_path).resolve()
    manifest = json.loads(mp.read_text(encoding="utf-8"))
    verify_manifest(manifest)
    verify_oof_receipts(manifest, mp)
    source = source_path(mp, manifest["source_csv"])
    if sha256_file(source) != manifest["source_sha256"]:
        raise ValueError("source CSV SHA-256 不匹配")
    recipe = manifest["recipe_sha256"]
    if not isinstance(recipe, str) or len(recipe) != 64 or any(c not in "0123456789abcdef" for c in recipe):
        raise ValueError("无效 recipe_sha256")
    frame = read_csv(oof_path)
    required = ["Subject_id", "Study_id", "fold", *LABELS, *PROBS]
    if not set(required).issubset(frame):
        raise ValueError("OOF 缺少 ID/fold/label/probability 列")
    if frame.empty or frame["Study_id"].duplicated().any() or (frame[["Study_id", "Subject_id", "fold"]] == "").any().any():
        raise ValueError("OOF 空 ID、空预测或重复 study")
    if not frame["fold"].map(lambda value: value.isdigit() and str(int(value)) == value).all():
        raise ValueError("OOF fold 必须为规范的非负整数字符串")
    if (frame.groupby("Subject_id")["fold"].nunique() != 1).any():
        raise ValueError("患者跨 fold")
    studies = manifest["studies"]
    expected = {str(row["Study_id"]): row for row in studies}
    if len(expected) != len(studies) or set(frame.Study_id) != set(expected):
        raise ValueError("OOF 与 manifest study 不一致")
    numeric = frame[LABELS + PROBS].apply(pd.to_numeric, errors="raise")
    y, p = numeric[LABELS].to_numpy(), numeric[PROBS].to_numpy()
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("OOF 标签或概率非法")
    for index, row in frame.iterrows():
        entry = expected[row.Study_id]
        if str(entry["Subject_id"]) != row.Subject_id or str(entry["fold"]) != row.fold or entry["labels"] != y[index].tolist():
            raise ValueError("OOF patient/fold/labels 与 manifest 不一致")
    # source CSV may contain image duplicates, but labels must agree for each study.
    source_frame = read_csv(source)
    counts = build_patient_counts(source_frame)
    align_patient_counts(frame, counts)
    source_labels = study_labels(source_frame)
    ordered = source_labels.set_index("Study_id").loc[frame.Study_id, LABELS].to_numpy()
    if not np.array_equal(ordered, y):
        raise ValueError("OOF labels 与 source CSV 不一致")
    frame[LABELS + PROBS] = numeric
    # 每个概率都必须能追溯到该折核验过的预测文件，不能拿训练内预测冒充 OOF。
    for entry in manifest["outer_folds"]:
        prediction_path = Path(entry["output_dir"]) / "predictions.csv"
        saved = read_csv(prediction_path).set_index("Study_id")
        selected = frame.loc[frame.fold == str(entry["fold"])].set_index("Study_id")
        if set(saved.index) != set(selected.index):
            raise ValueError("OOF 与折原始预测的 study 集合不一致")
        if not np.array_equal(saved.loc[selected.index, PROBS].to_numpy(dtype=float),
                              selected[PROBS].to_numpy(dtype=float)):
            raise ValueError("OOF 概率与有来源记录的外折预测不一致")
    return frame.reset_index(drop=True), manifest, counts


def study_labels(metadata: pd.DataFrame) -> pd.DataFrame:
    if "Predict_class" not in metadata:
        raise ValueError("描述/评分需要 Predict_class 标签")
    labels = np.stack([parse_label_string(v) for v in metadata.Predict_class])
    rows = metadata[["Subject_id", "Study_id"]].copy()
    rows[LABELS] = labels
    if (rows.groupby("Study_id")[LABELS].nunique() > 1).any().any():
        raise ValueError("同 study 的标签冲突")
    return rows.drop_duplicates("Study_id").reset_index(drop=True)


def strata(counts: np.ndarray) -> dict[str, np.ndarray]:
    return {"<5": counts < 5, "5-9": (counts >= 5) & (counts < 10), ">=10": counts >= 10, ">=5": counts >= 5}


def comparison(frame: pd.DataFrame, adjusted: np.ndarray, counts: np.ndarray) -> dict:
    y, raw = frame[LABELS].to_numpy(dtype=float), frame[PROBS].to_numpy(dtype=float)
    folds = []
    for fold in sorted(frame.fold.unique()):
        mask = frame.fold.to_numpy() == fold
        baseline, changed = metric_report(y[mask], raw[mask]), metric_report(y[mask], adjusted[mask])
        a, b = baseline["macro_auc_10"], changed["macro_auc_10"]
        folds.append({"fold": fold, "raw": baseline, "adjusted": changed, "delta_auc": b - a if a is not None and b is not None else None})
    deltas = [f["delta_auc"] for f in folds]
    return {"folds": folds, "mean_fold_delta_auc": float(np.mean(deltas)) if all(v is not None for v in deltas) else None,
            "positive_folds": sum(v is not None and v > 0 for v in deltas),
            "pooled_raw": metric_report(y, raw), "pooled_adjusted": metric_report(y, adjusted),
            "strata": {name: {"study_count": int(mask.sum()), "raw": metric_report(y[mask], raw[mask]), "adjusted": metric_report(y[mask], adjusted[mask])} for name, mask in strata(counts).items()}}


def historical_predictions(predictions_path: str | Path, metadata_path: str | Path
                           ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按 ID 核对历史宽表与当前完整带标签元数据，不要求 OOF 或已选参数。"""
    metadata = read_csv(metadata_path)
    counts = build_patient_counts(metadata)
    labels = study_labels(metadata).set_index('Study_id')
    frame = read_csv(predictions_path)
    align_patient_counts(frame, counts)
    if not set(PROBS).issubset(frame.columns):
        raise ValueError('历史预测缺少 p1..p10 概率列')
    probabilities = frame[PROBS].apply(pd.to_numeric, errors='raise')
    p = probabilities.to_numpy(dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('历史预测概率必须为 [0,1] 内有限数')
    present = set(LABELS) & set(frame.columns)
    if present and present != set(LABELS):
        raise ValueError('历史预测的 label_1..label_10 必须全部提供或全部省略')
    aligned = labels.loc[frame.Study_id]
    y = aligned[LABELS].to_numpy(dtype=int)
    if present:
        recorded = frame[LABELS].apply(pd.to_numeric, errors='raise').to_numpy(dtype=float)
        if not np.array_equal(recorded, y):
            raise ValueError('历史预测标签与元数据不一致')
    result = frame.copy()
    result['Subject_id'] = aligned.Subject_id.to_numpy()
    result[LABELS] = y
    result[PROBS] = probabilities
    return result, counts


def score_difference(adjusted: float | None, raw: float | None) -> float | None:
    """固定类别 AUC 不可计算时保留缺失状态。"""
    return adjusted - raw if adjusted is not None and raw is not None else None


def explore(args: argparse.Namespace, out: Path) -> None:
    """仅在 CPU 上探索历史概率；不自动冻结或部署在 valid 上挑出的参数。"""
    require_research_split(args.split)
    frame, counts = historical_predictions(args.predictions, args.metadata)
    n = align_patient_counts(frame, counts)
    y, raw = frame[LABELS].to_numpy(dtype=float), frame[PROBS].to_numpy(dtype=float)
    baseline = metric_report(y, raw)
    candidates, summary = {}, []
    frame.to_csv(out / 'raw_predictions.csv', index=False, encoding='utf-8')
    counts.to_csv(out / 'patient_counts.csv', index=False, encoding='utf-8')
    for beta in BETA_GRID:
        prior = PatientCountPrior(beta=beta)
        adjusted = adjust_probabilities(raw, n, prior)
        metrics = metric_report(y, adjusted)
        delta_auc = score_difference(metrics['macro_auc_10'], baseline['macro_auc_10'])
        delta_f1 = metrics['macro_f1_at_0_5'] - baseline['macro_f1_at_0_5']
        per_class_delta = [score_difference(a, b) for a, b in
                           zip(metrics['per_class_auc'], baseline['per_class_auc'])]
        stratified = {}
        for name, mask in strata(n).items():
            raw_group = metric_report(y[mask], raw[mask])
            adjusted_group = metric_report(y[mask], adjusted[mask])
            stratified[name] = {
                'study_count': int(mask.sum()),
                'patient_count': int(frame.loc[mask, 'Subject_id'].nunique()),
                'raw': raw_group, 'adjusted': adjusted_group,
                'delta_auc': score_difference(adjusted_group['macro_auc_10'], raw_group['macro_auc_10']),
            }
        candidates[str(beta)] = {'beta': beta, 'adjusted': metrics, 'delta_auc': delta_auc,
                                 'delta_f1': delta_f1, 'per_class_delta_auc': per_class_delta,
                                 'strata': stratified}
        summary.append({'beta': beta, 'raw_macro_auc_10': baseline['macro_auc_10'],
                        'adjusted_macro_auc_10': metrics['macro_auc_10'], 'delta_auc': delta_auc,
                        'raw_macro_f1_at_0_5': baseline['macro_f1_at_0_5'],
                        'adjusted_macro_f1_at_0_5': metrics['macro_f1_at_0_5'], 'delta_f1': delta_f1,
                        'diagnostic_auc_class2_10': metrics['diagnostic_auc_class2_10'],
                        'brier': metrics['brier']})
        pd.DataFrame({'class_id': range(1, 11), 'positive': baseline['per_class_positive'],
                      'negative': baseline['per_class_negative'], 'raw_auc': baseline['per_class_auc'],
                      'adjusted_auc': metrics['per_class_auc'], 'delta_auc': per_class_delta}
                     ).to_csv(out / f'per_class_beta_{beta:g}.csv', index=False, encoding='utf-8')
        changed = frame.copy()
        changed[PROBS] = adjusted
        changed.to_csv(out / f'adjusted_beta_{beta:g}.csv', index=False, encoding='utf-8')
        candidate_dir = out / f'beta_{beta:g}'
        candidate_dir.mkdir()
        write_audit(frame, counts, prior, candidate_dir)
    pd.DataFrame(summary).to_csv(out / 'summary.csv', index=False, encoding='utf-8')
    write_json(out / 'report.json', {
        'evidence': f'historical_{args.split}_exploration', 'split': args.split,
        'promotion': False, 'deployment_status': 'research_only',
        'interpretation': '同一数据上的候选探索，不是独立复核；不导出冻结参数，不证明泛化提升。',
        'study_count': len(frame), 'patient_count': int(frame.Subject_id.nunique()),
        'beta_grid': list(BETA_GRID), 'baseline': baseline, 'candidates': candidates,
        'provenance': {'metadata_path': str(Path(args.metadata).resolve()),
                       'metadata_sha256': sha256_file(args.metadata),
                       'predictions_path': str(Path(args.predictions).resolve()),
                       'predictions_sha256': sha256_file(args.predictions),
                       'script_sha256': sha256_file(__file__),
                       'created_at': datetime.now(timezone.utc).isoformat()},
    })
    print('历史预测探索（CPU；同一数据上的候选比较）：')
    print(pd.DataFrame(summary).to_string(index=False))
    print(f'结果目录：{out}')


def describe(args: argparse.Namespace, out: Path) -> None:
    require_research_split(args.split)
    metadata = read_csv(args.metadata)
    counts = build_patient_counts(metadata)
    studies = study_labels(metadata)
    n = align_patient_counts(studies, counts)
    y = studies[LABELS].to_numpy(dtype=float)
    groups = {**strata(n), **{f"n={value}": n == value for value in sorted(set(n))}}
    reports = []
    for name, mask in groups.items():
        selected = studies[mask]
        patient_sizes = selected.groupby("Subject_id").Study_id.transform("size").to_numpy()
        for weighting in ("study", "patient"):
            weights = np.ones(mask.sum()) if weighting == "study" else 1 / patient_sizes
            yy = y[mask]
            total = weights.sum()
            report = {"stratum": name, "weighting": weighting, "patients": int(selected.Subject_id.nunique()), "studies": int(mask.sum()),
                      "any_abnormal_rate": float(np.average(yy[:, :9].any(axis=1), weights=weights)) if total else None,
                      "mean_abnormal_labels": float(np.average(yy[:, :9].sum(axis=1), weights=weights)) if total else None,
                      "class10_rate": float(np.average(yy[:, 9], weights=weights)) if total else None}
            for c in range(10):
                report[f"class{c+1}_positive"] = int(yy[:, c].sum())
                report[f"class{c+1}_negative"] = int(len(yy) - yy[:, c].sum())
                report[f"class{c+1}_positive_rate"] = float(np.average(yy[:, c], weights=weights)) if total else None
            reports.append(report)
    cumulative = studies.groupby("Subject_id")[LABELS[:9]].max().sum(axis=1).rename("cumulative_abnormal_types")
    cumulative.to_csv(out / "patient_cumulative_abnormal_types.csv", encoding="utf-8")
    counts.to_csv(out / "patient_counts.csv", index=False, encoding="utf-8")
    write_json(out / "description.json", {"split": args.split, "source_sha256": sha256_file(args.metadata), "interpretation": "相关性；累计异常种类不等于单 study 风险", "strata": reports})


def apply(args: argparse.Namespace, out: Path) -> None:
    require_research_split(args.split)
    prior = load_prior(args.prior)
    recipe_hash = args.recipe_sha256
    if args.recipe_manifest:
        recipe_hash = json.loads(Path(args.recipe_manifest).read_text(encoding="utf-8"))["recipe_sha256"]
    if not prior.recipe_sha256 or recipe_hash != prior.recipe_sha256:
        raise ValueError("基础配方 SHA-256 不匹配")
    raw = read_csv(args.predictions)
    if any(column in raw for column in ["Predict_class", *LABELS]):
        raise ValueError("apply 不接受标签输入")
    counts = build_patient_counts(read_patient_metadata(args.metadata))
    adjusted = apply_prior_frame(raw, counts, prior)
    raw.to_csv(out / "raw_predictions.csv", index=False, encoding="utf-8")
    adjusted.to_csv(out / "adjusted_predictions.csv", index=False, encoding="utf-8")
    write_audit(raw, counts, prior, out)
    write_json(out / "manifest.json", {"split": args.split, "metadata_sha256": sha256_file(args.metadata), "prior_sha256": sha256_file(args.prior), "predictions_sha256": sha256_file(args.predictions), "recipe_sha256": recipe_hash})


def write_audit(frame: pd.DataFrame, counts: pd.DataFrame, prior: PatientCountPrior, out: Path) -> None:
    n = align_patient_counts(frame, counts)
    g = np.where(n > 1, np.log1p(np.minimum(n, 10)) - np.log(2.), 0.)
    audit = frame[[c for c in ["Subject_id", "Study_id"] if c in frame]].copy()
    audit["patient_study_count"], audit["g"], audit["beta"] = n, g, prior.beta
    p = frame[PROBS].to_numpy(dtype=float)
    for c in range(10):
        audit[f"bias{c+1}"] = prior.beta * g * (1 if c < 9 else -1)
        audit[f"clipped{c+1}"] = (prior.beta != 0) & (g != 0) & ((p[:, c] < 1e-6) | (p[:, c] > 1-1e-6))
    audit.to_csv(out / "audit.csv", index=False, encoding="utf-8")


def scan_select(args: argparse.Namespace, out: Path) -> None:
    with Path(args.config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config != FIXED_CONFIG:
        raise ValueError("研究配置必须符合事前冻结的固定设计")
    frame, manifest, counts = validate_oof(args.oof, args.manifest)
    n = align_patient_counts(frame, counts)
    raw = frame[PROBS].to_numpy(dtype=float)
    reports, scores = {}, {}
    for beta in BETA_GRID:
        adjusted = adjust_probabilities(raw, n, PatientCountPrior(beta=beta))
        report = comparison(frame, adjusted, n)
        reports[str(beta)] = report
        scores[beta] = report["pooled_adjusted"]["macro_auc_10"]
    chosen = choose_beta(scores)
    provenance = {"source_csv_sha256": manifest["source_sha256"], "manifest_sha256": sha256_file(args.manifest),
                  "manifest_path": str(Path(args.manifest).resolve()), "oof_sha256": sha256_file(args.oof),
                  "selected_folds": sorted(frame.fold.unique().tolist()), "source_subjects": sorted(frame.Subject_id.unique().tolist()),
                  "script_sha256": sha256_file(__file__), "beta_grid": list(BETA_GRID), "seed": 2026,
                  "class_order": list(range(1, 11)), "frozen_at": datetime.now(timezone.utc).isoformat()}
    prior = PatientCountPrior(beta=chosen, recipe_sha256=manifest["recipe_sha256"], provenance=provenance)
    if args.command == "select":
        save_prior(prior, out / "patient_count_prior.json")
    result = {"evidence": "exploratory_same_oof_selection_and_scoring", "candidates": reports, "selected_beta": chosen,
              "promotion": False, "deployment_status": "research_only", "provenance": provenance}
    nested_dir = getattr(args, "nested_selections_dir", None)
    if nested_dir:
        if len(frame.fold.unique()) != 5:
            raise ValueError("严格复核需要五个外折")
        adjusted = raw.copy()
        selected = {}
        for fold in sorted(frame.fold.unique()):
            frozen_path = Path(nested_dir) / f"outer_{fold}" / "patient_count_prior.json"
            frozen = load_prior(frozen_path)
            if frozen.recipe_sha256 != manifest["recipe_sha256"]:
                raise ValueError("inner/outer recipe SHA-256 不匹配")
            inner_path = Path(frozen.provenance["manifest_path"])
            if sha256_file(inner_path) != frozen.provenance["manifest_sha256"]:
                raise ValueError("inner manifest SHA-256 不匹配")
            inner = json.loads(inner_path.read_text(encoding="utf-8"))
            mask = frame.fold.to_numpy() == fold
            held_out = set(frame.loc[mask, "Subject_id"])
            inner_subjects = {str(row["Subject_id"]) for row in inner["studies"]}
            outer_train = set(frame.Subject_id) - held_out
            if inner.get("split") != "train" or inner_subjects != outer_train or held_out & inner_subjects:
                raise ValueError("inner manifest 必须恰好覆盖 outer train 患者且与外折评估患者互斥")
            if inner.get("recipe_sha256") != manifest["recipe_sha256"]:
                raise ValueError("inner manifest 配方不匹配")
            # Recheck source rather than trusting an editable cached subject list.
            inner_source = source_path(inner_path, inner["source_csv"])
            if sha256_file(inner_source) != inner["source_sha256"]:
                raise ValueError("inner source CSV SHA-256 不匹配")
            source_subjects = set(read_patient_metadata(inner_source).Subject_id)
            if source_subjects != outer_train:
                raise ValueError("inner source CSV 与 outer train 患者不一致")
            outer_entry = next(entry for entry in manifest["outer_folds"] if entry["fold"] == int(fold))
            if (inner.get("parent_manifest_sha256") != sha256_file(args.manifest)
                    or inner.get("outer_fold") != int(fold)
                    or inner["source_sha256"] != outer_entry["train_csv_sha256"]):
                raise ValueError("inner 清单未绑定当前外折训练池")
            inner_oof = inner_path.parent / "inner_oof.csv"
            if (sha256_file(inner_oof) != frozen.provenance.get("oof_sha256")
                    or frozen.provenance.get("source_csv_sha256") != inner["source_sha256"]):
                raise ValueError("冻结参数未绑定当前 inner OOF 与来源")
            validate_oof(str(inner_oof), str(inner_path))
            adjusted[mask] = adjust_probabilities(raw[mask], n[mask], frozen)
            selected[fold] = {"beta": frozen.beta, "prior_sha256": sha256_file(frozen_path)}
        strict = comparison(frame, adjusted, n)
        bootstrap = paired_bootstrap(frame, frame[LABELS].to_numpy(dtype=float), raw, adjusted)
        delta = strict["mean_fold_delta_auc"]
        ci_low = bootstrap["ci95"][0]
        promotion = delta is not None and delta >= .002 and strict["positive_folds"] >= 4 and bootstrap["ci_reliable"] and ci_low is not None and ci_low > 0
        result["nested"] = {"evidence": "nested_inner_selection_outer_evaluation", "selections": selected, "comparison": strict,
                            "bootstrap": bootstrap, "promotion": bool(promotion), "deployment_status": "research_only"}
        output = frame.copy()
        output[PROBS] = adjusted
        output.to_csv(out / "nested_adjusted_predictions.csv", index=False, encoding="utf-8")
    else:
        best = reports[str(chosen)]
        baseline = best["pooled_raw"]["macro_auc_10"]
        adjusted_auc = best["pooled_adjusted"]["macro_auc_10"]
        result["screen_pass"] = bool(baseline is not None and adjusted_auc is not None and adjusted_auc > baseline and best["positive_folds"] >= 4)
        result["screen_action"] = "仅可进入严格嵌套复核" if result["screen_pass"] else "停止；保持关闭，不扩展网格"
    frame.to_csv(out / "raw_predictions.csv", index=False, encoding="utf-8")
    for beta in BETA_GRID:
        candidate = PatientCountPrior(beta=beta)
        changed = frame.copy()
        changed[PROBS] = adjust_probabilities(raw, n, candidate)
        changed.to_csv(out / f"adjusted_beta_{beta:g}.csv", index=False, encoding="utf-8")
    write_audit(frame, counts, prior, out)
    write_json(out / "report.json", result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("describe", "apply", "scan", "select", "explore"):
        item = sub.add_parser(command)
        item.add_argument("--out", required=True)
        if command in ("describe", "apply", "explore"):
            item.add_argument("--metadata", required=True)
            item.add_argument("--split", choices=("train", "valid", "test"),
                              default="valid" if command == "explore" else None,
                              required=command != "explore")
        else:
            item.add_argument("--config", required=True)
            item.add_argument("--oof", required=True)
            item.add_argument("--manifest", required=True)
        if command == "scan":
            item.add_argument("--nested-selections-dir")
        if command == "explore":
            item.add_argument("--predictions", required=True, help="历史检查级概率宽表，标签可选")
        if command == "apply":
            item.add_argument("--predictions", required=True)
            item.add_argument("--prior", required=True)
            recipe = item.add_mutually_exclusive_group(required=True)
            recipe.add_argument("--recipe-sha256")
            recipe.add_argument("--recipe-manifest", help="读取该基础配方 manifest 的 recipe_sha256")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command in ("describe", "apply", "explore"):
        require_research_split(args.split)
    out = output_directory(args.out)
    if args.command == "describe":
        describe(args, out)
    elif args.command == "apply":
        apply(args, out)
    elif args.command == "explore":
        explore(args, out)
    else:
        scan_select(args, out)


if __name__ == "__main__":
    main()
