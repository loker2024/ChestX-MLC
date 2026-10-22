"""本地模拟提交格式生成及严格回读校验。"""
from pathlib import Path
import numpy as np
import pandas as pd

PROB_COLUMNS = [f'p{i}' for i in range(1, 11)]
SUBMISSION_COLUMNS = ['Subject_id', 'Study_id', 'Predict_class', 'Probability']


def aggregate_studies(frame: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.Dicom_id.duplicated().any() or set(predictions.Dicom_id) != set(frame.Dicom_id):
        raise ValueError('图片预测 ID 缺失、重复或多余')
    merged = frame[['Subject_id', 'Study_id', 'Dicom_id']].merge(predictions, on='Dicom_id', validate='one_to_one')
    values = merged[PROB_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('非法图片概率')
    if merged.groupby('Study_id').Subject_id.nunique().gt(1).any():
        raise ValueError('检查的患者归属不唯一')
    return merged.groupby(['Subject_id', 'Study_id'], sort=True, as_index=False)[PROB_COLUMNS].mean()


def make_submission(studies: pd.DataFrame) -> pd.DataFrame:
    result = studies.melt(id_vars=['Subject_id', 'Study_id'], value_vars=PROB_COLUMNS,
                          var_name='Predict_class', value_name='Probability')
    result['Predict_class'] = result['Predict_class'].str[1:].astype(int)
    return result.sort_values(['Study_id', 'Predict_class']).reset_index(drop=True)[SUBMISSION_COLUMNS]


def check_submission(path: Path, official: pd.DataFrame) -> dict:
    # ID 必须按字符串回读，避免丢失前导零。
    frame = pd.read_csv(path, dtype=str, encoding='utf-8', keep_default_na=False)
    checks = {'columns_exact': list(frame.columns) == SUBMISSION_COLUMNS}
    if not checks['columns_exact']:
        raise ValueError('提交列名或顺序不符')
    probabilities = pd.to_numeric(frame.Probability, errors='coerce').to_numpy()
    reference = official[['Subject_id', 'Study_id']].drop_duplicates()
    pairs = set(map(tuple, frame[['Subject_id', 'Study_id']].to_numpy()))
    checks.update(rows_exact=len(frame) == len(reference) * 10,
                  study_count_exact=frame.Study_id.nunique() == len(reference),
                  official_ids_exact=pairs == set(map(tuple, reference.to_numpy())),
                  no_duplicate_records=not frame.duplicated(['Study_id', 'Predict_class']).any(),
                  ten_classes_each=all(set(g.Predict_class) == set(map(str, range(1,11))) and len(g) == 10
                                       for _, g in frame.groupby('Study_id')),
                  probabilities_finite=bool(np.isfinite(probabilities).all()),
                  probabilities_in_range=bool(((probabilities >= 0) & (probabilities <= 1)).all()),
                  utf8_reread=True)
    result = {'passed': all(checks.values()), 'checks': checks, 'rows': len(frame),
              'studies': frame.Study_id.nunique(), 'columns': list(frame.columns),
              'file_bytes': path.stat().st_size, 'probability_min': float(probabilities.min()),
              'probability_max': float(probabilities.max()), 'local_simulation_only': True,
              'official_template_found': False,
              'unconfirmed_requirements': ['未提供正式 sample_submission.csv',
                   '本地文档有按阈值输出的描述；本次按用户要求输出每个检查的全部10类',
                   'Gate 列规则及最终疾病中文名称映射未在本地材料确认'],
              'gate_policy': '未添加 Gate；无自定义阈值'}
    if not result['passed']:
        raise ValueError(f'提交格式检查失败: {checks}')
    return result
