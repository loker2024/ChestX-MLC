"""生成、写出与校验 submission.csv。

赛题提交文件是长表，一行 = 一次检查 × 一个类别：

    Subject_id, Study_id, Predict_class, Probability
    subject_00000,study_00000,1,0.92
    ...

infer.predict 产出的是宽表（Study_id + p1..p10），这里负责把宽表摊平成长表，
再补上 Subject_id（宽表里没有，从 test.csv 的 Study_id -> Subject_id 映射取）。

默认输出全部 10 类概率：赛题主指标是 macro-AUC，它只看同一类别内的概率排序，
删掉低概率类别不会让排序变好，反而可能让评测方少拿到信息。需要按阈值过滤
（例如只提交 p >= 0.5 的类别）时，给 build_submission_frame 传 threshold。
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from common.constants import CLASS_IDS, CSV_ENCODING, NUM_CLASSES

# 赛题要求的长表列顺序，多一列少一列、顺序不对都不行。
SUBMISSION_COLUMNS = ("Subject_id", "Study_id", "Predict_class", "Probability")


def build_submission_frame(
    predictions: pd.DataFrame,
    study_subjects: Mapping[str, str],
    threshold: float | None = None,
) -> pd.DataFrame:
    """把 `Study_id + p1..p10` 的宽表摊平成提交用的长表。

    参数：
    - predictions：infer.predict.predict_loader 的输出，一行一个 study；
    - study_subjects：Study_id -> Subject_id 的映射，通常来自 test split 的
      StudyRecord（test.csv 是 subject 信息的唯一可信来源）；
    - threshold：None（默认）输出全部 10 类；给了阈值就只保留概率 >= 阈值的
      类别。某个 study 一个类都没过线时保留概率最高的那一类，保证提交文件
      仍然覆盖全部 study。

    返回列顺序固定为 SUBMISSION_COLUMNS；study 顺序与 predictions 一致，
    study 内部按类别 ID 1~10 排列。
    """
    if threshold is not None and not (
        math.isfinite(threshold) and 0.0 <= threshold <= 1.0
    ):
        raise ValueError(f"threshold 必须落在 [0, 1] 或为 None，当前为 {threshold!r}")

    rows: list[tuple[str, str, int, float]] = []
    for record in predictions.itertuples(index=False):
        study_id = str(record.Study_id)
        if study_id not in study_subjects:
            raise ValueError(f"predictions 里的 study {study_id} 不在 study_subjects 中")
        subject_id = study_subjects[study_id]
        probabilities = {
            class_id: float(getattr(record, f"p{class_id}")) for class_id in CLASS_IDS
        }

        if threshold is None:
            selected = CLASS_IDS
        else:
            selected = [
                class_id
                for class_id in CLASS_IDS
                if probabilities[class_id] >= threshold
            ]
            if not selected:
                # 一个都没过阈值时保留最高分的那一类：提交文件不能漏 study。
                selected = [max(CLASS_IDS, key=probabilities.__getitem__)]

        for class_id in selected:
            rows.append((subject_id, study_id, class_id, probabilities[class_id]))

    return pd.DataFrame(rows, columns=list(SUBMISSION_COLUMNS))


def write_submission(frame: pd.DataFrame, path: str | Path) -> Path:
    """把长表写成 UTF-8 CSV，返回实际写入的路径。

    父目录不存在会自动创建；换行统一成 Unix 风格，让 Linux 服务器和 Windows
    本地写出的文件字节布局一致。ID 列本来就是字符串，前导零不会被转掉。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding=CSV_ENCODING, lineterminator="\n")
    return path


def validate_submission(
    path: str | Path,
    test_csv: str | Path,
    require_all_classes: bool = True,
) -> pd.DataFrame:
    """按赛题要求校验 submission.csv，返回读入的长表。

    校验项：
    - 列名与顺序必须是 Subject_id, Study_id, Predict_class, Probability；
    - 文件里的 study 必须和 test.csv 一一对应（不多不少），Subject_id 与
      test.csv 一致；
    - Predict_class 落在 1~10，Probability 落在 [0, 1]；
    - 同一个 study 不会重复提交同一个类别；
    - require_all_classes=True 时还要求每个 study 的 10 个类别都在
      （默认提交策略就是输出全部 10 类概率，这样 macro-AUC 才有完整信息）。

    任何一项不通过都抛 ValueError，消息里带上具体的 study 或列名，方便定位。
    """
    test_frame = pd.read_csv(
        Path(test_csv), dtype=str, keep_default_na=False, encoding=CSV_ENCODING
    )
    missing_test_columns = sorted({"Study_id", "Subject_id"} - set(test_frame.columns))
    if missing_test_columns:
        raise ValueError(
            f"{test_csv} 缺少必需的列 {missing_test_columns}，"
            f"实际列为 {list(test_frame.columns)}"
        )
    subject_counts = test_frame.groupby("Study_id")["Subject_id"].nunique()
    conflicting = subject_counts[subject_counts > 1]
    if not conflicting.empty:
        study_id = conflicting.index[0]
        raise ValueError(
            f"test.csv 中 study {study_id} 对应多个 Subject_id，"
            "无法确定提交文件应使用的患者 ID"
        )
    subjects = dict(zip(test_frame["Study_id"], test_frame["Subject_id"]))

    frame = pd.read_csv(
        Path(path), dtype=str, keep_default_na=False, encoding=CSV_ENCODING
    )
    if list(frame.columns) != list(SUBMISSION_COLUMNS):
        raise ValueError(
            f"列名或顺序不正确：期望 {list(SUBMISSION_COLUMNS)}，"
            f"实际 {list(frame.columns)}"
        )
    if frame.empty:
        raise ValueError("提交文件为空")

    # 类型转换放在结构检查之后：列不对时先报列名，比 pandas 的类型错误好定位。
    frame["Predict_class"] = frame["Predict_class"].astype(int)
    frame["Probability"] = frame["Probability"].astype(float)

    unknown = sorted(set(frame["Study_id"]) - set(subjects))
    if unknown:
        raise ValueError(f"提交文件含 test.csv 中不存在的 study：{unknown[:5]}")

    missing = sorted(set(subjects) - set(frame["Study_id"]))
    if missing:
        raise ValueError(
            f"提交文件缺少 {len(missing)} 个 test study，例如 {missing[:5]}"
        )

    # map 对不存在的 key 会给 NaN，但前面已经保证 study 都存在。
    expected_subjects = frame["Study_id"].map(subjects)
    wrong_subject = frame["Subject_id"] != expected_subjects
    if wrong_subject.any():
        row = frame[wrong_subject].iloc[0]
        raise ValueError(
            f"{int(wrong_subject.sum())} 行的 Subject_id 与 test.csv 不一致，"
            f"例如 study {row['Study_id']} 写的是 {row['Subject_id']}，"
            f"应为 {expected_subjects[wrong_subject].iloc[0]}"
        )

    invalid_class = ~frame["Predict_class"].isin(CLASS_IDS)
    if invalid_class.any():
        values = sorted(frame.loc[invalid_class, "Predict_class"].unique())
        raise ValueError(f"Predict_class 必须在 1~{NUM_CLASSES} 内，发现 {values[:5]}")

    probabilities = frame["Probability"]
    invalid_probability = (
        ~np.isfinite(probabilities) | (probabilities < 0) | (probabilities > 1)
    )
    if invalid_probability.any():
        raise ValueError(
            f"Probability 必须在 [0, 1] 内，发现 "
            f"{probabilities[invalid_probability].iloc[0]}"
        )

    duplicated = frame.duplicated(subset=["Study_id", "Predict_class"], keep=False)
    if duplicated.any():
        row = frame[duplicated].iloc[0]
        raise ValueError(
            f"同一个 study 的同一个类别重复出现：study {row['Study_id']}，"
            f"类别 {row['Predict_class']}"
        )

    if require_all_classes:
        class_counts = frame.groupby("Study_id")["Predict_class"].nunique()
        incomplete = class_counts[class_counts != NUM_CLASSES]
        if not incomplete.empty:
            study_id = incomplete.index[0]
            raise ValueError(
                f"study {study_id} 只有 {int(incomplete.iloc[0])} 个类别，"
                f"期望 {NUM_CLASSES} 个；默认提交应输出全部类别"
            )

    return frame
