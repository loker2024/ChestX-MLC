"""从官方train一次性按患者分组，按标签计数贪心平衡；无模型、无test输入。"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis_common import (TRAIN_FINGERPRINT, VALID_FINGERPRINT, dump,
                              finish_manifest, provenance, save_csv, sha256)
from .data import read_csv, parse_labels
from .fingerprints import dataset_fingerprint


DEFAULT_CONFIG = {'validation_fraction': 0.15, 'seed': 42,
                  'algorithm': 'rare_first_group_greedy_v1',
                  'validation_mode': 'internal_subject_holdout',
                  'prevalence_warning_absolute': 0.05}


def validate_config(config: dict) -> dict:
    if config.get('algorithm') == 'structure_multilabel_swap_v2':
        from .split_quality import validate_v2
        return validate_v2(config)
    if set(config) != set(DEFAULT_CONFIG):
        raise ValueError('划分配置字段缺失或未知')
    if config['algorithm'] != DEFAULT_CONFIG['algorithm'] or config['validation_mode'] != DEFAULT_CONFIG['validation_mode']:
        raise ValueError('未知患者划分算法/validation_mode')
    if type(config['seed']) is not int or not 0 <= config['seed'] < 2**32:
        raise ValueError('seed须为合法整数')
    for key in ('validation_fraction', 'prevalence_warning_absolute'):
        if type(config[key]) not in (int, float) or not math.isfinite(config[key]):
            raise ValueError('划分比例必须是有限数值')
    if not .10 <= config['validation_fraction'] <= .15 or not 0 < config['prevalence_warning_absolute'] <= 1:
        raise ValueError('内部验证患者比例须为10%—15%，告警阈值须在(0,1]')
    return dict(config)


def summarize(frame: pd.DataFrame) -> dict:
    studies = frame.groupby('Study_id', sort=True).first()
    labels = np.stack(studies.Predict_class.map(parse_labels))
    patients = frame.Subject_id.nunique()
    return {'patients': int(patients), 'studies': len(studies), 'images': len(frame),
            'positive_studies': labels.sum(0).astype(int).tolist(),
            'negative_studies': (len(studies)-labels.sum(0)).astype(int).tolist(),
            'positive_study_prevalence': labels.mean(0).tolist(),
            'positive_patients': [int(studies.loc[labels[:, i] == 1, 'Subject_id'].nunique()) for i in range(10)],
            'fingerprint': dataset_fingerprint(frame)}


def assign_subjects(frame: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    """只看train标签；一次rare-first排序、一次分配，无seed搜索或模型评分。"""
    studies = frame.groupby('Study_id', sort=True).first().reset_index()
    labels = np.stack(studies.Predict_class.map(parse_labels))
    subjects = sorted(studies.Subject_id.unique())
    if len(subjects) < 4:
        raise ValueError('患者过少，不能建立可靠内部划分')
    # 同一患者可有不同检查标签，用Study级正负计数而非随意合并患者标签。
    positive, negative = [], []
    for subject in subjects:
        y = labels[studies.Subject_id.eq(subject).to_numpy()]
        positive.append(y.sum(0)); negative.append((1-y).sum(0))
    positive, negative = np.array(positive), np.array(negative)
    if ((positive > 0).sum(0) < 2).any() or ((negative > 0).sum(0) < 2).any():
        raise ValueError('至少一类正/负样本不足两名患者，无法保证患者隔离后双侧覆盖')
    n = len(subjects); held = int(math.ceil(n*fraction))
    capacities = np.array([n-held, held])
    features = np.concatenate([positive, negative, np.ones((n, 1))], axis=1)
    total = features.sum(0); targets = capacities[:, None]/n * total
    # 稀有阳性患者优先；固定随机数仅用于稳定打破同分，输入行重排不改变结果。
    rarity = ((positive > 0)/np.maximum((positive > 0).sum(0), 1)).sum(1)
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(n), -rarity))
    counts = np.zeros_like(targets); remaining = capacities.copy(); membership = {}
    for index in order:
        candidates = np.flatnonzero(remaining > 0)
        # 两折各正负计数距目标的归一化平方误差增量，末维固定患者配额。
        changes = []
        for fold in candidates:
            before = (counts[fold]-targets[fold])/np.maximum(targets[fold], 1)
            after = (counts[fold]+features[index]-targets[fold])/np.maximum(targets[fold], 1)
            changes.append(float(np.sum(after**2-before**2)))
        fold = int(candidates[int(np.argmin(changes))])
        counts[fold] += features[index]; remaining[fold] -= 1
        membership[subjects[index]] = 'internal_train' if fold == 0 else 'internal_valid'
    return pd.DataFrame({'Subject_id': subjects, 'split': [membership[s] for s in subjects]})


def create_split(train_csv: Path, official_valid_csv: Path, output: Path,
                 config: dict, require_official: bool = True) -> Path:
    config = validate_config(config)
    if output.exists():
        raise FileExistsError('划分目录已存在；只允许一次固定划分，拒绝覆盖')
    inputs = {'official_train': train_csv, 'official_valid': official_valid_csv}
    origin = provenance(inputs, {**config, 'require_official': require_official})
    train, official = read_csv(train_csv, True), read_csv(official_valid_csv, True)
    if require_official:
        if (len(train), train.Study_id.nunique()) != (30344, 18513) or dataset_fingerprint(train) != TRAIN_FINGERPRINT:
            raise ValueError('train数量/标签/ID指纹与官方固定训练集不符')
        if (len(official), official.Study_id.nunique()) != (1422, 859) or dataset_fingerprint(official) != VALID_FINGERPRINT:
            raise ValueError('官方valid数量/标签/ID指纹不符')
    for key in ('Subject_id', 'Study_id', 'Dicom_id'):
        if set(train[key]) & set(official[key]):
            raise ValueError('官方train与valid存在交叉：'+key)
    v2 = config['algorithm'] == 'structure_multilabel_swap_v2'
    optimization = None
    if v2:
        from .split_quality import assign_v2
        assignment, optimization = assign_v2(train, config)
    else:
        assignment = assign_subjects(train, config['validation_fraction'], config['seed'])
    held = set(assignment.loc[assignment.split.eq('internal_valid'), 'Subject_id'])
    fit, valid = train[~train.Subject_id.isin(held)], train[train.Subject_id.isin(held)]
    stats = {'source_train': summarize(train), 'internal_train': summarize(fit), 'internal_valid': summarize(valid),
             'official_valid_reserved': summarize(official)}
    for split in ('internal_train', 'internal_valid'):
        if min(stats[split]['positive_studies']) < 1 or min(stats[split]['negative_studies']) < 1:
            raise ValueError('一次划分出现类别正负覆盖不足；停止，不自动换seed寻找划分')
    for key in ('Subject_id', 'Study_id', 'Dicom_id'):
        if set(fit[key]) & set(valid[key]) or set(fit[key]) | set(valid[key]) != set(train[key]):
            raise RuntimeError('患者划分存在泄漏、遗漏或错误归属')
    report = {'config': config, 'statistics': stats, 'all_classes_defined_both_sides': True,
              'patient_study_image_overlap': 0, 'official_valid_excluded': True,
              'test_accessed': False, 'assignment_passes': 1, 'model_scores_used': False,
              'source_kind': 'official' if require_official else 'synthetic_fixture_only',
              'old_full_train_models_are_independent_on_this_holdout': False,
              'training_integration_status': 'prepared_only; current full_train/continue_train still enforce official_valid_eval_only',
              'warnings': []}
    for split in ('internal_train', 'internal_valid'):
        delta = np.array(stats[split]['positive_study_prevalence'])-np.array(stats['source_train']['positive_study_prevalence'])
        report[split+'_prevalence_delta'] = delta.tolist()
        if np.max(np.abs(delta)) > config.get('prevalence_warning_absolute', config.get('max_prevalence_absolute_delta')):
            report['warnings'].append(split+'标签比例偏差超出预设告警阈值，未更换seed')
    if v2:
        from .split_quality import quality
        report['quality'] = quality(train, fit, valid, config)
        report['optimization'] = optimization
        if not report['quality']['qualified']:
            report['warnings'].append('v2未通过预定质量检查，禁止启动A/B；不自动换seed或放宽阈值')
    # 导出源CSV原有字段和字符串内容，不将新增metadata辅助列写回原始数据。
    raw = pd.read_csv(train_csv, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    output.mkdir(parents=True, exist_ok=False)
    try:
        save_csv(raw[~raw.Subject_id.isin(held)], output/'internal_train.csv')
        save_csv(raw[raw.Subject_id.isin(held)], output/'internal_valid.csv')
        save_csv(assignment, output/'subject_membership.csv')
        save_csv(train[['Subject_id', 'Study_id']].drop_duplicates().merge(assignment, on='Subject_id', validate='many_to_one'), output/'study_membership.csv')
        dump(output/'split_config.json', config); dump(output/'split_report.json', report)
        dump(output/'provenance.json', origin)
        for item in origin['inputs'].values():
            if sha256(item['path']) != item['sha256']:
                raise RuntimeError('源CSV发生变化')
        finish_manifest(output)
        print(json.dumps({'output': str(output.resolve()), 'train': stats['internal_train'],
                          'valid': stats['internal_valid'], 'training_started': False}, ensure_ascii=False))
        return output
    except Exception as exc:
        dump(output/'failure.json', {'type': type(exc).__name__, 'error': str(exc)})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-csv', type=Path, required=True)
    parser.add_argument('--official-valid-csv', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('configs/analysis/patient_holdout.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    create_split(args.train_csv, args.official_valid_csv, args.output,
                 json.loads(args.config.read_text(encoding='utf-8-sig')))


if __name__ == '__main__':
    main()
