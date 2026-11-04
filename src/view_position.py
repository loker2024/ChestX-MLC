"""VP-01：固定正位/侧位组权重，异常视角整Study回退G0；不调用模型。"""
import numpy as np
import pandas as pd

from .aggregation import aggregate_predictions
from .submission import align_image_predictions, PROB_COLUMNS

FRONTAL = frozenset({'AP', 'PA'})
LATERAL = frozenset({'LATERAL', 'LL'})
VP01_RULE = {
    'name': 'VP-01', 'frontal_views': sorted(FRONTAL), 'lateral_views': sorted(LATERAL),
    'frontal_group_weight': 0.7, 'lateral_group_weight': 0.3,
    'within_group': 'arithmetic mean of all image probabilities',
    'fallback': 'whole Study G0 if singleton, any missing/unknown view, or either group absent',
    'view_normalization': 'strip outer whitespace and uppercase only; no guessed aliases',
}


def canonical_view(value: object) -> str:
    return '' if pd.isna(value) else str(value).strip().upper()


def aggregate_vp01(frame: pd.DataFrame, predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """按Dicom_id连接权威视角；返回Study概率、Study决策及逐图权重审计。"""
    required = {'Dicom_id', 'Subject_id', 'Study_id', 'ViewPosition'}
    if not required.issubset(frame) or not required.issubset(predictions):
        raise ValueError('原CSV和图片预测必须含Dicom_id/Subject_id/Study_id/ViewPosition')
    baseline = aggregate_predictions(frame, predictions, 'G0')
    merged = align_image_predictions(frame, predictions)
    if any(merged[k].astype(str).str.strip().eq('').any() for k in ('Subject_id', 'Study_id', 'Dicom_id')):
        raise ValueError('图片或Study ID为空')
    views = frame[['Dicom_id', 'ViewPosition']].merge(
        predictions[['Dicom_id', 'ViewPosition']], on='Dicom_id', validate='one_to_one',
        suffixes=('_source', '_prediction'))
    if not views.ViewPosition_source.map(canonical_view).equals(views.ViewPosition_prediction.map(canonical_view)):
        raise ValueError('预测文件ViewPosition与权威验证CSV不一致，不能按错视角加权')
    result = baseline.set_index('Study_id')
    decisions, weights = [], []
    for study_id, group in merged.groupby('Study_id', sort=True):
        view = group.ViewPosition.map(canonical_view)
        front, side = view.isin(FRONTAL), view.isin(LATERAL)
        if len(group) == 1:
            reason = 'single_image'
        elif view.eq('').any():
            reason = 'missing_view'
        elif (~(front | side)).any():
            reason = 'unknown_view'
        elif not front.any() or not side.any():
            reason = 'no_explicit_front_and_side'
        else:
            reason = 'weighted_front_side'
        applied = reason == 'weighted_front_side'
        image_weights = np.full(len(group), 1.0 / len(group))
        if applied:
            image_weights[front.to_numpy()] = .7 / int(front.sum())
            image_weights[side.to_numpy()] = .3 / int(side.sum())
            # 组内均值再7:3；不是给每张正位图都分配0.7。
            value = .7 * group.loc[front, PROB_COLUMNS].mean().to_numpy() + .3 * group.loc[side, PROB_COLUMNS].mean().to_numpy()
            result.loc[study_id, PROB_COLUMNS] = value
        decisions.append({'Subject_id': group.Subject_id.iloc[0], 'Study_id': study_id,
            'image_count': len(group), 'frontal_images': int(front.sum()), 'lateral_images': int(side.sum()),
            'weighted': applied, 'reason': reason})
        for (_, row), weight, normalized in zip(group.iterrows(), image_weights, view):
            weights.append({'Dicom_id': row.Dicom_id, 'Subject_id': row.Subject_id, 'Study_id': study_id,
                'ViewPosition': row.ViewPosition, 'normalized_view': normalized,
                'weight': float(weight), 'reason': reason})
    audit = pd.DataFrame(weights)
    if (len(audit) != len(frame) or audit.Dicom_id.duplicated().any()
            or set(audit.Dicom_id) != set(frame.Dicom_id) or not audit.weight.gt(0).all()
            or not np.allclose(audit.groupby('Study_id').weight.sum(), 1, atol=1e-14, rtol=0)):
        raise RuntimeError('图片覆盖或Study权重异常')
    result = result.reset_index()[['Subject_id', 'Study_id', *PROB_COLUMNS]].sort_values(['Subject_id', 'Study_id']).reset_index(drop=True)
    values = result[PROB_COLUMNS].to_numpy(float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('VP-01概率非法')
    return result, pd.DataFrame(decisions), audit
