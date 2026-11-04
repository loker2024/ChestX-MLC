"""官方train训练、官方valid选优；不读取v2或test。"""
import json
from pathlib import Path

import numpy as np

from .analysis_common import TRAIN_FINGERPRINT, VALID_FINGERPRINT, sha256
from .data import read_csv, parse_labels
from .fingerprints import dataset_fingerprint
from .patient_holdout import summarize

PATH_KEYS = {'official_train_csv', 'official_valid_csv', 'train_images',
             'valid_images', 'pretrained_weight_path'}


def read_official_paths(path: Path) -> dict[str, Path]:
    values = json.loads(path.read_text(encoding='utf-8-sig'))
    if set(values) != PATH_KEYS or not all(isinstance(v, str) and v.strip() for v in values.values()):
        raise ValueError('官方模式需要train/valid CSV、各自图片目录和预训练权重五路径；禁止split/test路径')
    return {k: (path.parent / v).resolve() for k, v in values.items()}


def audit_official(paths: dict[str, Path], *, require_official: bool = True) -> tuple[dict, dict]:
    frames = {s: read_csv(paths['official_' + s + '_csv'], True) for s in ('train', 'valid')}
    for key in ('Subject_id', 'Study_id', 'Dicom_id'):
        if set(frames['train'][key]) & set(frames['valid'][key]):
            raise ValueError('官方train/valid交叉：' + key)
    for s in frames:
        labels = np.stack(frames[s].groupby('Study_id').first().Predict_class.map(parse_labels))
        if (labels.sum(0) == 0).any() or ((1-labels).sum(0) == 0).any():
            raise ValueError(s + '类别缺少正负样本')
    fingerprints = {s: dataset_fingerprint(f) for s, f in frames.items()}
    if require_official:
        for s, count, studies, fingerprint in (
            ('train', 30344, 18513, TRAIN_FINGERPRINT),
            ('valid', 1422, 859, VALID_FINGERPRINT)):
            if (len(frames[s]), frames[s].Study_id.nunique(), fingerprints[s]) != (count, studies, fingerprint):
                raise ValueError('官方CSV数量/ID/标签指纹不符：' + s)
    hashes = {str(paths[k]): sha256(paths[k]) for k in ('official_train_csv', 'official_valid_csv')}
    return frames, {'validation_mode': 'official_valid', 'train_fingerprint': fingerprints['train'],
        'validation_fingerprint': fingerprints['valid'], 'source_hashes': hashes,
        'statistics': {s: summarize(f) for s, f in frames.items()},
        'official_valid_used_for_selection': True, 'official_valid_in_training': False,
        'v2_used': False, 'test_accessed': False, 'source_kind': 'official' if require_official else 'fixture'}


def prepare_official(c: dict, path: Path) -> tuple:
    from .swinv2_experiment import SwinDataset, make_study_loader
    paths = read_official_paths(path)
    _, audit = audit_official(paths)
    c['pretrained_weight_path'] = str(paths['pretrained_weight_path'])
    ds = {s: SwinDataset(paths['official_' + s + '_csv'], paths[s + '_images'],
                        size=384, normalization='imagenet', augment=s == 'train')
          for s in ('train', 'valid')}
    return paths, audit, ds, make_study_loader(ds['train'], c)
