"""评估指标：macro-AUC、macro-F1 与逐类明细。

赛题是 10 类多标签，每个类别各自做一次「有 / 无」的二分类，所以指标都是
「先算每一类，再跨类平均」。平均之前必须先剔除退化类别：

- 全负类（验证集里一个正样本都没有）：AUC 的分母为 0，没有定义；
- 全正类（验证集里全是正样本）：同理，也没有可比性。

sklearn 遇到这种类会直接抛错（"Only one class present in y_true"）。
这里的做法是把这类类的指标记成 NaN，算 macro 时跳过，而不是填 0：
填 0 相当于说「模型在这个类上全错」，可这类类在验证集里根本没出现过，
硬算进去只会把整体指标无故拉低，也掩盖了「这个类没有验证样本」这件事。

数据形状统一成 [n, 10]：n 是 study 数（训练和提交都按 study 聚合），
列顺序与 CLASS_IDS 一致（第 0 列 = 类别 1，…，第 9 列 = 类别 10）。
y_true 是 0/1 标签，y_score 是 sigmoid 之后的概率（不是 logits）。
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


def per_class_report(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5
) -> pd.DataFrame:
    """返回逐类明细表，`macro_auc` / `macro_f1` 都从这张表聚合出来。

    参数：

    - `y_true`：`[n, 10]` 的 0/1 标签；
    - `y_score`：`[n, 10]` 的概率，与 y_true 形状一致；
    - `threshold`：概率 >= threshold 判为正类，只影响 precision / recall / f1，
      AUC 只看排序，不受阈值影响。

    返回的每行是一类，列含义：

    - `class_id`：赛题的类别 ID（1~10），不是数组下标；
    - `support`：这一类的正样本数；
    - `pos_rate`：正样本占比，方便一眼看出类别有多稀有；
    - `auc` / `precision` / `recall` / `f1`：退化类为 NaN。
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.shape != y_score.shape:
        raise ValueError(
            f"y_true 与 y_score 形状必须一致，当前为 {y_true.shape} 与 {y_score.shape}"
        )

    rows: list[dict[str, Any]] = []
    for index in range(y_true.shape[1]):
        true = y_true[:, index]
        score = y_score[:, index]
        positives = int(true.sum())

        row: dict[str, Any] = {
            "class_id": index + 1,  # 下标 + 1 = 赛题里的类别 ID
            "support": positives,
            "pos_rate": positives / len(true),
            "auc": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
        }
        # 0 < 正样本数 < 样本总数，才说明这类既有正也有负，指标才有定义。
        if 0 < positives < len(true):
            row["auc"] = float(roc_auc_score(true, score))
            # 先按阈值把概率转成 0/1，再算这一类的二分类 F1。
            # zero_division=0：模型一个都没预测成正类时，precision 记为 0 而不是报错。
            precision, recall, f1, _ = precision_recall_fscore_support(
                true, (score >= threshold).astype(np.int64), average="binary", zero_division=0
            )
            row["precision"] = float(precision)
            row["recall"] = float(recall)
            row["f1"] = float(f1)
        rows.append(row)

    return pd.DataFrame(
        rows, columns=["class_id", "support", "pos_rate", "auc", "precision", "recall", "f1"]
    )


def macro_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """各类 ROC-AUC 的算术平均，退化类不参与平均。

    没有退化类时，结果与
    `sklearn.metrics.roc_auc_score(y_true, y_score, average="macro")` 一致
    （tests/test_metrics.py 就是拿 sklearn 对齐的）。有退化类时 sklearn 会直接
    报错，这里选择跳过；全部类别都退化时返回 nan。

    返回 nan 通常意味着验证集出了问题（例如只取了几行），比返回一个假的 0.5
    更容易在训练日志里被发现。
    """
    report = per_class_report(y_true, y_score)
    # pandas 的 mean 默认跳过 NaN，正好就是「剔除退化类后再平均」。
    return float(report["auc"].mean())


def macro_f1(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> float:
    """各类二分类 F1 的算术平均，退化类不参与平均。

    阈值决定「概率多大算预测为正」。阈值不同 F1 会明显变化，所以记录 F1 时
    一定要带上阈值（P7 会做阈值对比）。
    """
    report = per_class_report(y_true, y_score, threshold)
    return float(report["f1"].mean())
