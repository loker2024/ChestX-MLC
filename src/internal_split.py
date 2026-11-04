"""接入已经冻结的患者留出；核对源数据、派生行、清单和患者边界，不重新划分。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis_common import TRAIN_FINGERPRINT, VALID_FINGERPRINT, sha256
from .data import read_csv, parse_labels
from .fingerprints import dataset_fingerprint
from .patient_holdout import summarize

PATH_KEYS = {'split_directory', 'official_train_csv', 'official_valid_csv',
             'image_root', 'pretrained_weight_path'}
SPLIT_FILES = {'internal_train.csv', 'internal_valid.csv', 'subject_membership.csv',
               'study_membership.csv', 'split_config.json', 'split_report.json', 'provenance.json'}


def read_paths(path: Path) -> dict[str, Path]:
    values = json.loads(path.read_text(encoding='utf-8-sig'))
    if set(values) != PATH_KEYS or not all(isinstance(v, str) and v.strip() for v in values.values()):
        raise ValueError('路径配置必须且只能包含五个非空路径；不接受test路径')
    # 相对路径相对于路径配置文件，便于独立解压后迁移。
    return {k: (path.parent / v).resolve() for k, v in values.items()}


def _raw(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8-sig')


def verify_split(paths: dict[str, Path], *, require_official: bool = True) -> tuple[dict, dict]:
    folder = paths['split_directory']
    manifest_path = folder / 'output_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))['sha256']
    if set(manifest) != SPLIT_FILES:
        raise ValueError('必须提供患者划分工具导出的完整七文件和output_manifest.json')
    for name, digest in manifest.items():
        if sha256(folder / name) != digest:
            raise ValueError('划分文件被修改：' + name)
    origin = json.loads((folder / 'provenance.json').read_text(encoding='utf-8'))
    report = json.loads((folder / 'split_report.json').read_text(encoding='utf-8'))
    split_config = json.loads((folder / 'split_config.json').read_text(encoding='utf-8'))
    if split_config != report['config'] or split_config['validation_mode'] != 'internal_subject_holdout':
        raise ValueError('划分报告/配置不一致')
    if require_official and report.get('source_kind') != 'official':
        raise ValueError('正式入口禁止使用模拟划分')
    for key in ('official_train', 'official_valid'):
        if sha256(paths[key + '_csv']) != origin['inputs'][key]['sha256']:
            raise ValueError('原始CSV与冻结划分来源不同：' + key)
    source = read_csv(paths['official_train_csv'], True)
    official = read_csv(paths['official_valid_csv'], True)
    if require_official and ((len(source), source.Study_id.nunique()) != (30344, 18513)
            or dataset_fingerprint(source) != TRAIN_FINGERPRINT
            or (len(official), official.Study_id.nunique()) != (1422, 859)
            or dataset_fingerprint(official) != VALID_FINGERPRINT):
        raise ValueError('官方train/valid数量或标签ID指纹不匹配')
    frames = {s: read_csv(folder / ('internal_' + s + '.csv'), True) for s in ('train', 'valid')}
    for key in ('Subject_id', 'Study_id', 'Dicom_id'):
        if set(source[key]) & set(official[key]):
            raise ValueError('官方train/valid交叉：' + key)
        if set(frames['train'][key]) & set(frames['valid'][key]):
            raise ValueError('内部训练验证泄漏：' + key)
        if set(frames['train'][key]) | set(frames['valid'][key]) != set(source[key]):
            raise ValueError('内部划分遗漏或混入外部数据：' + key)
    original = _raw(paths['official_train_csv'])
    derived = [_raw(folder / ('internal_' + s + '.csv')) for s in ('train', 'valid')]
    if any(list(table.columns) != list(original.columns) for table in derived):
        raise ValueError('派生CSV字段改变')
    joined = pd.concat(derived, ignore_index=True)
    pd.testing.assert_frame_equal(original.sort_values('Dicom_id').reset_index(drop=True),
                                  joined.sort_values('Dicom_id').reset_index(drop=True))
    subjects = _raw(folder / 'subject_membership.csv')
    studies = _raw(folder / 'study_membership.csv')
    actual_subjects, actual_studies = [], []
    for split, frame in frames.items():
        actual_subjects.extend({'Subject_id': s, 'split': 'internal_' + split} for s in frame.Subject_id.unique())
        actual_studies.append(frame[['Subject_id', 'Study_id']].drop_duplicates().assign(split='internal_' + split))
        y = np.stack(frame.groupby('Study_id').first().Predict_class.map(parse_labels))
        if (y.sum(0) == 0).any() or ((1 - y).sum(0) == 0).any():
            raise ValueError('内部类别缺正样本或负样本，不能计算完整Macro-AUC')
    for expected, actual, keys in ((subjects, pd.DataFrame(actual_subjects), ['Subject_id']),
            (studies, pd.concat(actual_studies), ['Study_id'])):
        cols = list(actual.columns)
        pd.testing.assert_frame_equal(expected[cols].sort_values(keys).reset_index(drop=True),
                                      actual.sort_values(keys).reset_index(drop=True))
    if split_config.get('algorithm') == 'structure_multilabel_swap_v2':
        from .split_quality import quality, validate_v2
        validate_v2(split_config)
        current_quality = quality(source, frames['train'], frames['valid'], split_config)
        if not current_quality['qualified'] or current_quality != report.get('quality'):
            raise ValueError('v2质量未通过或记录被篡改，禁止A/B训练')
    hashes = {str((folder / name).resolve()): sha256(folder / name)
              for name in SPLIT_FILES | {'output_manifest.json'}}
    hashes.update({str(paths[k].resolve()): sha256(paths[k]) for k in ('official_train_csv', 'official_valid_csv')})
    stats = {s: summarize(frame) for s, frame in frames.items()}
    return frames, {'validation_mode': 'internal_subject_holdout', 'split_manifest_sha256': sha256(manifest_path),
        'train_fingerprint': dataset_fingerprint(frames['train']),
        'validation_fingerprint': dataset_fingerprint(frames['valid']), 'statistics': stats,
        'source_kind': report['source_kind'], 'source_hashes': hashes,
        'official_valid_used_for_selection': False, 'test_accessed': False,
        'split_warnings': report.get('warnings', []),
        'note': '冻结清单不重划；正负覆盖仅保证AUC可计算，不保证统计精度'}


def assert_sources_unchanged(hashes: dict) -> None:
    for path, digest in hashes.items():
        if sha256(path) != digest:
            raise RuntimeError('运行中源文件改变：' + path)
