"""本地模拟提交格式生成及严格回读校验。"""
from pathlib import Path
import numpy as np
import pandas as pd

PROB_COLUMNS = [f'p{i}' for i in range(1, 11)]
SUBMISSION_COLUMNS = ['Subject_id', 'Study_id', 'Predict_class', 'Probability']


def align_image_predictions(frame: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """兼容精简/带元数据CSV；按ID核对归属，绝不依赖行序或采用预测文件标签。"""
    if frame.Dicom_id.duplicated().any() or predictions.Dicom_id.duplicated().any() or set(predictions.Dicom_id) != set(frame.Dicom_id):
        raise ValueError('图片预测 ID 缺失、重复或多余')
    for key in ('Subject_id', 'Study_id'):
        if key in predictions:
            identities = frame[['Dicom_id', key]].merge(predictions[['Dicom_id', key]],
                on='Dicom_id',validate='one_to_one',suffixes=('_official','_prediction'))
            if not identities[key+'_official'].eq(identities[key+'_prediction']).all():
                raise ValueError(f'图片预测{key}归属不匹配')
    selected = predictions[['Dicom_id', *PROB_COLUMNS]].copy()
    selected[PROB_COLUMNS] = selected[PROB_COLUMNS].apply(pd.to_numeric,errors='raise')
    merged = frame.merge(selected, on='Dicom_id', validate='one_to_one')
    values = merged[PROB_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('非法图片概率')
    return merged


def aggregate_studies(frame: pd.DataFrame, predictions: pd.DataFrame,
                      method: str = 'G0') -> pd.DataFrame:
    from .aggregation import aggregate_predictions
    return aggregate_predictions(frame,predictions,method)


def make_submission(studies: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
    if threshold is not None and (not np.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError('阈值必须在 [0,1] 内')
    values = studies[PROB_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('非法检查概率')
    result = studies.melt(id_vars=['Subject_id', 'Study_id'], value_vars=PROB_COLUMNS,
                          var_name='Predict_class', value_name='Probability')
    result['Predict_class'] = result['Predict_class'].str[1:].astype(int)
    if threshold is not None:
        result = result.loc[result.Probability >= threshold]
    return result.sort_values(['Study_id', 'Predict_class']).reset_index(drop=True)[SUBMISSION_COLUMNS]


def check_submission(path: Path, official: pd.DataFrame, threshold: float | None = None,
                     studies: pd.DataFrame | None = None,
                     columns: list[str] | None = None) -> dict:
    # ID 必须按字符串回读，避免丢失前导零。
    frame = pd.read_csv(path, dtype=str, encoding='utf-8', keep_default_na=False)
    checks = {'columns_exact': list(frame.columns) == (columns or SUBMISSION_COLUMNS)}
    if not checks['columns_exact']:
        raise ValueError('提交列名或顺序不符')
    probabilities = pd.to_numeric(frame.Probability, errors='coerce').to_numpy()
    reference = official[['Subject_id', 'Study_id']].drop_duplicates()
    pairs = set(map(tuple, frame[['Subject_id', 'Study_id']].to_numpy()))
    official_pairs = set(map(tuple, reference.to_numpy()))
    if threshold is None:
        expected = len(reference) * 10
        checks.update(rows_exact=len(frame) == expected,
                      study_count_exact=frame.Study_id.nunique() == len(reference),
                      ten_classes_each=all(set(g.Predict_class) == set(map(str, range(1,11))) and len(g) == 10
                                           for _, g in frame.groupby('Study_id')))
    else:
        if studies is None:
            raise ValueError('阈值模式必须提供完整检查概率用于验证被省略的类别')
        if set(map(tuple, studies[['Subject_id','Study_id']].to_numpy())) != official_pairs or studies.Study_id.duplicated().any():
            raise ValueError('检查概率的 ID 与官方不符')
        selected = make_submission(studies, threshold)
        keys = ['Subject_id','Study_id','Predict_class']
        expected_keys = selected[keys].astype(str)
        joined = frame.merge(expected_keys.assign(expected=True), on=keys, how='outer', indicator=True)
        checks.update(threshold_selection_exact=bool(joined['_merge'].eq('both').all()),
                      rows_exact=len(frame) == len(selected),
                      probabilities_above_threshold=bool((probabilities >= threshold).all()))
    checks.update(official_ids_exact=pairs <= official_pairs if threshold is not None else pairs == official_pairs,
                  no_duplicate_records=not frame.duplicated(['Study_id', 'Predict_class']).any(),
                  class_ids_valid=bool(frame.Predict_class.isin(list(map(str, range(1,11)))).all()),
                  probabilities_finite=bool(np.isfinite(probabilities).all()),
                  probabilities_in_range=bool(((probabilities >= 0) & (probabilities <= 1)).all()),
                  utf8_reread=True)
    if studies is not None:
        expected_frame = make_submission(studies,threshold)
        keys = ['Subject_id','Study_id','Predict_class']
        expected_frame['Predict_class'] = expected_frame.Predict_class.astype(str)
        compared = frame.merge(expected_frame,on=keys,suffixes=('_actual','_expected'),how='outer',indicator=True)
        checks['probabilities_match_source'] = bool(compared['_merge'].eq('both').all() and
            np.allclose(pd.to_numeric(compared.Probability_actual,errors='coerce'),
                        compared.Probability_expected,rtol=0,atol=1e-12))
    result = {'passed': all(checks.values()), 'checks': checks, 'rows': len(frame),
              'studies': frame.Study_id.nunique(), 'columns': list(frame.columns),
              'file_bytes': path.stat().st_size, 'probability_min': float(probabilities.min()) if len(frame) else None,
              'probability_max': float(probabilities.max()) if len(frame) else None, 'local_simulation_only': True,
              'mode': 'all_classes' if threshold is None else 'threshold', 'threshold': threshold,
              'official_template_found': False,
              'unconfirmed_requirements': ['未提供正式 sample_submission.csv',
                   '输出全部类别或仅阳性类别须由正式模板或主办方确认；命令行必须显式选择',
                   'Gate 列规则及最终疾病中文名称映射未在本地材料确认'],
              'gate_policy': '未添加 Gate；Gate 语义需要官方确认'}
    if not result['passed']:
        raise ValueError(f'提交格式检查失败: {checks}')
    return result
