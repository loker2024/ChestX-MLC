"""不泄露原始ID的划分/标签指纹，以及固定验证协议。"""
import hashlib
import json
import pandas as pd
from .data import parse_labels

EVALUATION_PROTOCOL = {'classes': list(range(1,11)), 'unit': 'Study_id',
                       'aggregation': 'mean image probabilities per Study_id',
                       'f1_threshold': 0.5, 'auc_undefined': 'null'}


def aggregation_protocol(method: str = 'G0', logits_source: str | None = None) -> dict:
    """G0协议字典保持原样，其他方法显式分开，不能混作同一评估协议。"""
    protocol = dict(EVALUATION_PROTOCOL)
    if method == 'G0':
        return protocol
    if method == 'G1':
        protocol['aggregation'] = 'max image probabilities per Study_id'
    elif method == 'G2' and logits_source in ('raw_logits','inverse_probabilities'):
        protocol.update(aggregation='sigmoid mean image logits per Study_id',
                        logits_source=logits_source,singleton_policy='preserve_original_probabilities')
        if logits_source == 'inverse_probabilities':
            protocol['clip_epsilon'] = 1e-7
    else:
        raise ValueError('聚合协议或logits来源不支持')
    return protocol


def dataset_fingerprint(frame: pd.DataFrame) -> str:
    """忽略CSV行顺序和标签书写顺序，保留图片归属及标签语义。"""
    columns = ['Subject_id','Study_id','Dicom_id']
    rows = []
    for row in frame.sort_values(columns).to_dict('records'):
        rows.append([*[row[key] for key in columns], parse_labels(row['Predict_class']).astype(int).tolist()])
    payload = json.dumps({'schema':1,'rows':rows},ensure_ascii=False,separators=(',',':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def comparison_group(metrics: dict) -> str | None:
    fingerprint = metrics.get('validation_fingerprint')
    protocol = metrics.get('evaluation_protocol')
    if not isinstance(fingerprint,str) or len(fingerprint) != 64 or protocol != EVALUATION_PROTOCOL:
        return None
    try:
        int(fingerprint,16)
    except ValueError:
        return None
    return fingerprint
