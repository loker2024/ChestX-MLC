"""不泄露原始ID的划分/标签指纹，以及固定验证协议。"""
import hashlib
import json
import pandas as pd
from .data import parse_labels

EVALUATION_PROTOCOL = {'classes': list(range(1,11)), 'unit': 'Study_id',
                       'aggregation': 'mean image probabilities per Study_id',
                       'f1_threshold': 0.5, 'auc_undefined': 'null'}


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
