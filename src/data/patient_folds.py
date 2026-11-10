"""仅在官方训练来源内建立患者互斥的、可追溯的折清单。"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from data.labels import parse_label_string
from data.patient_counts import build_patient_counts


def sha256_file(path: str | Path) -> str:
    """流式计算文件哈希。"""
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_metadata(path: str | Path) -> pd.DataFrame:
    """元数据全程字符串读取，保留前导零。"""
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8')


def study_table(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """校验患者映射与多图标签一致性，按 ID 排序。"""
    required = {'Subject_id', 'Study_id', 'Predict_class'}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError('训练元数据为空或缺少患者、study、标签列')
    build_patient_counts(frame)
    if 'Dicom_id' in frame:
        images = frame[['Dicom_id', 'Study_id']].drop_duplicates()
        if images['Dicom_id'].duplicated().any() or (images['Dicom_id'] == '').any():
            raise ValueError('Dicom_id 为空或对应多个 study')
    for key in ('Subject_id', 'Study_id'):
        if not frame[key].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f'{key} 必须为非空字符串')
    studies = []
    for study, rows in frame.groupby('Study_id', sort=True):
        if rows['Subject_id'].nunique() != 1:
            raise ValueError(f'{study} 对应多个患者')
        labels = [parse_label_string(value).astype(int).tolist() for value in rows['Predict_class']]
        if any(value != labels[0] for value in labels):
            raise ValueError(f'{study} 的图片标签不一致')
        studies.append({'Subject_id': rows.iloc[0]['Subject_id'], 'Study_id': study, 'labels': labels[0]})
    return studies


def assign_patient_folds(studies: list[dict[str, Any]], n_splits: int, seed: int) -> dict[str, int]:
    """固定排序后按 GroupKFold 分组；要求支持 shuffle 的 sklearn。"""
    ordered = sorted(studies, key=lambda row: row['Study_id'])
    groups = np.asarray([row['Subject_id'] for row in ordered])
    if len(set(groups)) < n_splits:
        raise ValueError(f'患者数不足以建立 {n_splits} 折')
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    result = {}
    for fold, (_, valid) in enumerate(splitter.split(np.zeros(len(groups)), groups=groups)):
        for index in valid:
            result[ordered[index]['Study_id']] = fold
    return result


def verify_manifest(manifest: dict[str, Any]) -> None:
    """每次运行重新核验来源、折 CSV 和配置文件。"""
    if manifest.get('split') != 'train':
        raise ValueError('OOF 仅接受官方 train 来源')
    if (manifest.get('schema_version') != 1 or type(manifest.get('n_splits')) is not int
            or manifest['n_splits'] not in {3, 5}):
        raise ValueError('OOF schema/折数不符')
    if sha256_file(manifest['source_csv']) != manifest['source_sha256']:
        raise ValueError('OOF 来源 CSV 哈希不符')
    actual = {row['Study_id']: row for row in study_table(read_metadata(manifest['source_csv']))}
    recorded = manifest['studies']
    if len(recorded) != len(actual) or {r['Study_id'] for r in recorded} != set(actual):
        raise ValueError('manifest study 覆盖不完整或重复')
    patient_folds = {}
    for row in recorded:
        reference = actual[row['Study_id']]
        if any(row[key] != reference[key] for key in ('Subject_id', 'labels')):
            raise ValueError('manifest 患者或标签不符')
        previous = patient_folds.setdefault(row['Subject_id'], row['fold'])
        if previous != row['fold']:
            raise ValueError('患者跨折')
    expected_folds = set(range(manifest['n_splits']))
    if (any(type(row['fold']) is not int for row in recorded)
            or {row['fold'] for row in recorded} != expected_folds):
        raise ValueError('manifest 折编号不完整')
    fold_entries = manifest['outer_folds']
    if len(fold_entries) != manifest['n_splits'] or {row['fold'] for row in fold_entries} != expected_folds:
        raise ValueError('折配置条目缺失或重复')
    for fold in manifest['outer_folds']:
        for key in ('train_csv', 'valid_csv', 'config'):
            if sha256_file(fold[key]) != fold[f'{key}_sha256']:
                raise ValueError(f'{key} 哈希不符')
        if set(fold['train_subjects']) & set(fold['valid_subjects']):
            raise ValueError('训练患者与留出患者相交')
        for key, is_valid in (('train_csv', False), ('valid_csv', True)):
            observed = study_table(read_metadata(fold[key]))
            expected = {r['Study_id']: actual[r['Study_id']] for r in recorded
                        if (r['fold'] == fold['fold']) == is_valid}
            subject_key = 'valid_subjects' if is_valid else 'train_subjects'
            if sorted({r['Subject_id'] for r in observed}) != fold[subject_key]:
                raise ValueError('患者清单与折 CSV 不符')
            if {r['Study_id']: r for r in observed} != expected:
                raise ValueError('折 CSV 的患者/study/标签与清单不符')


def verify_oof_receipts(manifest: dict[str, Any], manifest_path: str | Path) -> None:
    """离线选参前核验每折的固定末轮、模型配置及原始预测来源。"""
    import json
    from common.config import load_config
    from infer.patient_count_prior import recipe_fingerprint

    for entry in manifest['outer_folds']:
        receipt_path = Path(entry['output_dir']) / 'provenance.json'
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        expected = {
            'manifest_sha256': sha256_file(manifest_path),
            'config_sha256': entry['config_sha256'],
            'recipe_sha256': manifest['recipe_sha256'],
            'fold': entry['fold'], 'train_subjects': entry['train_subjects'],
            'valid_subjects': entry['valid_subjects'],
            'epoch': 5, 'weights_kind': 'fixed_last_ema', 'initialization': 'imagenet',
        }
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise ValueError('OOF 折来源记录与冻结配方或训练患者不符')
        config = load_config(entry['config'])
        if (config.data.train.csv.resolve() != Path(entry['train_csv']).resolve()
                or config.data.valid.csv.resolve() != Path(entry['valid_csv']).resolve()
                or config.data.full_train or config.data.resplit.enabled
                or config.train.epochs != 5 or config.train.early_stop_patience != 0
                or not config.train.ema_enabled or not config.train.require_pretrained
                or recipe_fingerprint([config]) != manifest['recipe_sha256']):
            raise ValueError('OOF 配置路径、冻结轮数或配方不符')
        for key in ('last_checkpoint', 'ema_checkpoint', 'predictions'):
            if sha256_file(receipt[key]) != receipt.get(f'{key}_sha256'):
                raise ValueError(f'OOF 来源记录中的 {key} 哈希不符')
        if Path(receipt['predictions']).resolve() != (Path(entry['output_dir']) / 'predictions.csv').resolve():
            raise ValueError('OOF 预测路径与折路径不同')
