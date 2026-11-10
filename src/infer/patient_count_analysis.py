"""患者整簇、折内配对的固定十类研究统计。"""
from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

BETA_GRID = (0.0, 0.05, 0.10, 0.20)


def metric_report(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    """缺正/负类时固定十类 AUC 返回 None，同时列出有效类。"""
    aucs = [float(roc_auc_score(labels[:, c], probabilities[:, c]))
            if len(np.unique(labels[:, c])) == 2 else None for c in range(10)]
    predicted = probabilities >= .5
    tp = (predicted & (labels == 1)).sum(axis=0)
    denominator = predicted.sum(axis=0) + labels.sum(axis=0)
    f1 = np.divide(2 * tp, denominator, out=np.zeros(10), where=denominator != 0)
    return {"study_count": int(len(labels)), "per_class_positive": labels.sum(axis=0).astype(int).tolist(),
            "per_class_negative": (len(labels) - labels.sum(axis=0)).astype(int).tolist(),
            "macro_auc_10": float(np.mean(aucs)) if all(x is not None for x in aucs) else None,
            "diagnostic_auc_class2_10": float(np.mean(aucs[1:])) if all(x is not None for x in aucs[1:]) else None,
            "per_class_auc": aucs, "valid_classes": [i + 1 for i, x in enumerate(aucs) if x is not None],
            "macro_f1_at_0_5": float(f1.mean()), "brier": float(np.mean((labels - probabilities) ** 2)) if len(labels) else None}


def choose_beta(scores: dict[float, float | None]) -> float:
    valid = {beta: score for beta, score in scores.items() if score is not None}
    if not valid or valid.get(0.) is None:
        return 0.
    best = max(valid.values())
    if best <= valid[0.] + 1e-6:
        return 0.
    return min(beta for beta, score in valid.items() if best - score <= 1e-6)


def paired_bootstrap(frame: pd.DataFrame, labels: np.ndarray, raw: np.ndarray,
                     adjusted: np.ndarray) -> dict[str, Any]:
    """固定 2000 次/seed2026；每折抽患者，携带其全部 study 后计算折差均值。"""
    rng = np.random.default_rng(2026)
    groups = []
    for _, fold in frame.groupby("fold", sort=True):
        groups.append([np.asarray(rows, dtype=int) for rows in fold.groupby("Subject_id", sort=True).groups.values()])
    differences = []
    for _ in range(2000):
        deltas = []
        for patients in groups:
            indices = np.concatenate([patients[i] for i in rng.integers(0, len(patients), len(patients))])
            y = labels[indices]
            if any(len(np.unique(y[:, c])) != 2 for c in range(10)):
                break
            deltas.append(float(roc_auc_score(y, adjusted[indices], average="macro") - roc_auc_score(y, raw[indices], average="macro")))
        if len(deltas) == len(groups):
            differences.append(float(np.mean(deltas)))
    invalid = 2000 - len(differences)
    return {"iterations": 2000, "seed": 2026, "valid_iterations": len(differences), "invalid_iterations": invalid,
            "invalid_fraction": invalid / 2000, "ci_reliable": invalid / 2000 <= .05,
            "ci95": np.quantile(differences, [.025, .975]).tolist() if differences else [None, None],
            "scope": "固定流程的患者抽样不确定性；不覆盖训练随机性与所有选型偏差"}
