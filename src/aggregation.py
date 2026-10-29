"""固定权重的Study级聚合；不使用视位、ID或真实标签作为预测特征。"""
import numpy as np
import pandas as pd

from .submission import align_image_predictions, PROB_COLUMNS

METHODS = ('G0','G1','G2')
LOGIT_COLUMNS = [f'logit{i}' for i in range(1,11)]
LOGIT_EPS = 1e-7


def sigmoid(values: np.ndarray) -> np.ndarray:
    """稳定sigmoid，避免大正/负logit溢出。"""
    return np.exp(-np.logaddexp(0.0, -values))


def method_info(predictions: pd.DataFrame, method: str) -> dict:
    if method not in METHODS:
        raise ValueError('聚合方法只支持G0/G1/G2')
    source = None
    if method == 'G2':
        present = set(LOGIT_COLUMNS) & set(predictions.columns)
        if present and present != set(LOGIT_COLUMNS):
            raise ValueError('原始logit列不完整，要求logit1..logit10，不能部分回退')
        source = 'raw_logits' if present else 'inverse_probabilities'
    return {'method':method,'logits_source':source,
            'clip_epsilon':LOGIT_EPS if source == 'inverse_probabilities' else None,
            'singleton_policy':'preserve_original_probabilities'}


def aggregate_predictions(frame: pd.DataFrame, predictions: pd.DataFrame,
                          method: str = 'G0') -> pd.DataFrame:
    info = method_info(predictions,method)
    merged = align_image_predictions(frame,predictions)
    if merged.empty or merged[['Subject_id','Study_id','Dicom_id']].isna().any().any() or any(
            merged[key].astype(str).str.strip().eq('').any() for key in ('Subject_id','Study_id','Dicom_id')):
        raise ValueError('聚合ID缺失或数据为空')
    if merged.groupby('Study_id').Subject_id.nunique().gt(1).any():
        raise ValueError('同一Study_id的患者归属不唯一')
    grouped = merged.groupby('Study_id',sort=True)
    if method == 'G0':
        values = grouped[PROB_COLUMNS].mean()
    elif method == 'G1':
        values = grouped[PROB_COLUMNS].max()
    else:
        if info['logits_source'] == 'raw_logits':
            raw = predictions[['Dicom_id',*LOGIT_COLUMNS]].copy()
            raw[LOGIT_COLUMNS] = raw[LOGIT_COLUMNS].apply(pd.to_numeric,errors='raise')
            joined = merged[['Dicom_id','Study_id',*PROB_COLUMNS]].merge(raw,on='Dicom_id',validate='one_to_one')
            logits = joined[LOGIT_COLUMNS].to_numpy(dtype=float)
            if not np.isfinite(logits).all():
                raise ValueError('原始logits包含非有限值')
            if not np.allclose(sigmoid(logits),joined[PROB_COLUMNS].to_numpy(dtype=float),rtol=0,atol=1e-6):
                raise ValueError('原始logits与对应图片概率不一致')
        else:
            joined = merged[['Dicom_id','Study_id',*PROB_COLUMNS]].copy()
            probabilities = np.clip(joined[PROB_COLUMNS].to_numpy(dtype=float),LOGIT_EPS,1-LOGIT_EPS)
            logits = np.log(probabilities)-np.log1p(-probabilities)
        joined[LOGIT_COLUMNS] = logits
        mean_logits = joined.groupby('Study_id',sort=True)[LOGIT_COLUMNS].mean()
        values = pd.DataFrame(sigmoid(mean_logits.to_numpy()),index=mean_logits.index,columns=PROB_COLUMNS)
        # 概率在0/1处反变换必须裁剪，但单图Study无需聚合，保留原概率以满足等价性。
        singletons = grouped.size().loc[lambda counts: counts == 1].index
        values.loc[singletons,PROB_COLUMNS] = grouped[PROB_COLUMNS].first().loc[singletons,PROB_COLUMNS]
    subjects = grouped['Subject_id'].first().rename('Subject_id')
    result = values.join(subjects).reset_index()[['Subject_id','Study_id',*PROB_COLUMNS]]
    probabilities = result[PROB_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError('聚合结果包含非有限值或越界概率')
    # 仅按Study分组；最后排序保持旧接口的输出顺序，不改变均值算法。
    return result.sort_values(['Subject_id','Study_id']).reset_index(drop=True)
