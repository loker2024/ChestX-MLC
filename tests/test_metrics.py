"""评估指标：与 sklearn 对齐、退化类别剔除、阈值对 F1 的影响。"""

import math

import numpy as np
import pytest
from sklearn.metrics import f1_score, roc_auc_score

from common.metrics import macro_auc, macro_f1, per_class_report


def _synthetic(num_studies: int = 80) -> tuple[np.ndarray, np.ndarray]:
    """造一批「每一类都既有正样本又有负样本」的合成数据。

    第 index 类的前 5 + 3*index 个 study 标为正，各类的正样本数因此不同，
    macro 平均才看得出「是跨类平均」而不是碰巧等于某一个类。
    """
    y_true = np.zeros((num_studies, 10))
    for index in range(10):
        y_true[: 5 + 3 * index, index] = 1.0

    # 正样本给更高的概率，再叠一点噪声：AUC 明显高于 0.5，且各类互不相同。
    rng = np.random.default_rng(0)
    y_score = np.clip(0.5 * y_true + 0.5 * rng.random((num_studies, 10)), 0.0, 1.0)
    return y_true, y_score


def test_macro_auc_matches_sklearn():
    """没有退化类时，结果应与 sklearn 的 macro 平均逐一相同。"""
    y_true, y_score = _synthetic()

    expected = roc_auc_score(y_true, y_score, average="macro")

    assert macro_auc(y_true, y_score) == pytest.approx(expected)


def test_macro_f1_matches_sklearn():
    """F1 的口径同样与 sklearn 对齐：先按阈值二值化，再算每类 F1 取平均。"""
    y_true, y_score = _synthetic()
    predicted = (y_score >= 0.5).astype(int)

    expected = f1_score(y_true, predicted, average="macro", zero_division=0)

    assert macro_f1(y_true, y_score) == pytest.approx(expected)


def test_all_negative_class_is_skipped():
    """全负类（一个正样本都没有）的 AUC/F1 没有定义，应置 NaN 且不参与平均。"""
    y_true, y_score = _synthetic()
    y_true[:, 2] = 0.0  # 第 3 类在「验证集」里全是负样本

    report = per_class_report(y_true, y_score)
    assert math.isnan(report.loc[2, "auc"])
    assert math.isnan(report.loc[2, "f1"])
    assert report.loc[2, "support"] == 0

    # 剩下的 9 类照常平均，sklearn 也能算（它不会碰到退化类）。
    keep = [index for index in range(10) if index != 2]
    assert macro_auc(y_true, y_score) == pytest.approx(
        roc_auc_score(y_true[:, keep], y_score[:, keep], average="macro")
    )
    predicted = (y_score >= 0.5).astype(int)
    assert macro_f1(y_true, y_score) == pytest.approx(
        f1_score(y_true[:, keep], predicted[:, keep], average="macro", zero_division=0)
    )


def test_all_positive_class_is_skipped():
    """全正类同样无法评估：没有负样本，AUC 的分母为 0。"""
    y_true, y_score = _synthetic()
    y_true[:, 5] = 1.0  # 第 6 类全是正样本

    report = per_class_report(y_true, y_score)
    assert math.isnan(report.loc[5, "auc"])
    assert report.loc[5, "support"] == len(y_true)

    keep = [index for index in range(10) if index != 5]
    assert macro_auc(y_true, y_score) == pytest.approx(
        roc_auc_score(y_true[:, keep], y_score[:, keep], average="macro")
    )


def test_macro_metrics_are_nan_when_nothing_is_evaluable():
    """全部类别都退化时返回 nan，而不是 0 或 0.5 这种看起来正常的假值。"""
    y_true = np.zeros((10, 10))
    y_score = np.full((10, 10), 0.5)

    assert math.isnan(macro_auc(y_true, y_score))
    assert math.isnan(macro_f1(y_true, y_score))


def test_threshold_controls_f1():
    """同一组概率，阈值抬高后正样本变少，F1 从 1 掉到 0。"""
    y_true = np.array([[1.0], [1.0], [0.0], [0.0]])
    y_score = np.array([[0.6], [0.6], [0.4], [0.4]])

    assert macro_f1(y_true, y_score, threshold=0.5) == pytest.approx(1.0)
    # 阈值 0.7 时两张正样本图都判成负类，precision / recall 都是 0。
    assert macro_f1(y_true, y_score, threshold=0.7) == pytest.approx(0.0)


def test_per_class_report_columns_and_support():
    """逐类明细表要能直接抄进实验记录：类别 ID 从 1 开始，support 是正样本数。"""
    y_true, y_score = _synthetic()

    report = per_class_report(y_true, y_score)

    assert list(report.columns) == [
        "class_id",
        "support",
        "pos_rate",
        "auc",
        "precision",
        "recall",
        "f1",
    ]
    assert report["class_id"].tolist() == list(range(1, 11))
    assert report["support"].tolist() == [5 + 3 * index for index in range(10)]
    assert ((report["pos_rate"] >= 0) & (report["pos_rate"] <= 1)).all()


def test_shape_mismatch_raises():
    """标签和概率对不上多半是拼接时漏了一个 split，早点报错比算出一个假指标好。"""
    with pytest.raises(ValueError, match="形状必须一致"):
        macro_auc(np.zeros((4, 10)), np.zeros((5, 10)))
