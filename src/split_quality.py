"""固定患者结构分层+有界交换；只优化train标签/结构的预先规定目标，不使用模型成绩。"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis_common import dump, finish_manifest, save_csv, sha256
from .data import parse_labels, read_csv

V2_CONFIG = {'validation_fraction': .15, 'seed': 42, 'algorithm': 'structure_multilabel_swap_v2',
    'validation_mode': 'internal_subject_holdout', 'max_swap_steps': 80,
    'validation_candidates_per_bin': 128, 'training_candidates_per_bin': 256,
    'max_prevalence_absolute_delta': .01, 'max_volume_fraction_delta': .01,
    'max_single_image_share_delta': .03, 'max_positive_patient_target_relative_delta': .20,
    'minimum_positive_negative_patients': 20}


def validate_v2(config: dict) -> dict:
    # v2是一次固定方案，不通过反复修改权重/阈值选择真实数据划分。
    if config != V2_CONFIG:
        raise ValueError('v2须使用已固定的规则、seed42、15%及预先声明的质量阈值')
    return dict(config)


def patient_features(frame: pd.DataFrame) -> tuple[list, np.ndarray, np.ndarray]:
    studies = frame.groupby('Study_id', sort=True).first().reset_index()
    y = np.stack(studies.Predict_class.map(parse_labels)).astype(float)
    subjects = sorted(studies.Subject_id.unique()); index = pd.Index(subjects)
    codes = index.get_indexer(studies.Subject_id)
    pos = np.zeros((len(subjects), 10)); np.add.at(pos, codes, y)
    count = np.bincount(codes, minlength=len(subjects)).astype(float)
    neg = count[:, None] - pos
    image_counts = frame.groupby('Study_id').size().reindex(studies.Study_id).to_numpy()
    images = np.bincount(codes, weights=image_counts, minlength=len(subjects))
    single = np.bincount(codes, weights=image_counts == 1, minlength=len(subjects))
    # 正/负Study、独立正/负患者、Study/图片量、单图/多图Study均进入目标。
    features = np.column_stack([pos, neg, pos > 0, neg > 0, count, images, single, count-single]).astype(float)
    bins = np.select([count == 1, count <= 3, count <= 7], [0, 1, 2], default=3)
    return subjects, features, bins


def assign_v2(frame: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    config = validate_v2(config)
    subjects, features, bins = patient_features(frame)
    n = len(subjects); held = math.ceil(n * config['validation_fraction']); fraction = held/n
    sizes = np.bincount(bins, minlength=4)
    expected = sizes * fraction; quotas = np.floor(expected).astype(int)
    for b in sorted(range(4), key=lambda b: (-(expected[b]-quotas[b]), b))[:held-int(quotas.sum())]:
        quotas[b] += 1
    rng = np.random.default_rng(config['seed']); selected = np.zeros(n, dtype=bool)
    for b in range(4):
        selected[rng.permutation(np.flatnonzero(bins == b))[:quotas[b]]] = True
    target = features.sum(0) * fraction
    weights = np.r_[np.full(10, 2/10), np.full(10, 1/10), np.full(10, 2/10),
                    np.full(10, 1/10), np.full(4, 2/4)]
    scaled = features * (np.sqrt(weights)/np.maximum(target, 1))
    target_scaled = target * (np.sqrt(weights)/np.maximum(target, 1))
    residual = scaled[selected].sum(0) - target_scaled
    initial = float(residual @ residual); accepted = 0
    # 固定候选规模和最多80步；每步选同结构bin内最改善目标的一对患者交换。
    for _ in range(config['max_swap_steps']):
        best = (-1e-12, None, None)
        for b in range(4):
            vi = np.flatnonzero(selected & (bins == b)); ti = np.flatnonzero(~selected & (bins == b))
            if not len(vi) or not len(ti):
                continue
            vi = np.sort(rng.choice(vi, min(len(vi), config['validation_candidates_per_bin']), replace=False))
            ti = np.sort(rng.choice(ti, min(len(ti), config['training_candidates_per_bin']), replace=False))
            v, t = scaled[vi], scaled[ti]
            delta = (np.sum(v*v, axis=1) - 2*v@residual)[:, None] + (
                np.sum(t*t, axis=1) + 2*t@residual)[None, :] - 2*v@t.T
            flat = int(np.argmin(delta)); i, j = np.unravel_index(flat, delta.shape)
            if delta[i, j] < best[0]:
                best = (float(delta[i,j]), int(vi[i]), int(ti[j]))
        if best[1] is None:
            break
        _, remove, add = best
        selected[remove] = False; selected[add] = True
        residual += scaled[add] - scaled[remove]; accepted += 1
    assignment = pd.DataFrame({'Subject_id': subjects,
        'split': np.where(selected, 'internal_valid', 'internal_train')})
    return assignment, {'initial_objective': initial, 'final_objective': float(residual@residual),
        'accepted_swaps': accepted, 'max_swap_steps': config['max_swap_steps'],
        'patient_bin_edges': ['1', '2-3', '4-7', '8+'], 'patient_bin_totals': sizes.tolist(),
        'validation_patient_bin_quotas': quotas.tolist(), 'seed_runs': 1,
        'uses_only_train_labels_and_structure': True, 'model_scores_used': False,
        'local_optimum_guaranteed': False}


def structure(frame: pd.DataFrame) -> dict:
    studies = frame[['Subject_id','Study_id']].drop_duplicates()
    counts = studies.groupby('Subject_id').size()
    images = frame.groupby('Study_id').size()
    bins = [int((counts == 1).sum()), int(counts.between(2,3).sum()),
            int(counts.between(4,7).sum()), int((counts >= 8).sum())]
    return {'patients': len(counts), 'studies': len(studies), 'images': len(frame),
        'studies_per_patient_mean': float(counts.mean()),
        'studies_per_patient_quantiles': {str(q): float(counts.quantile(q)) for q in (.5,.9,.95,.99,1)},
        'patient_bins': dict(zip(['1','2-3','4-7','8+'], bins)),
        'single_image_studies': int((images == 1).sum()), 'multi_image_studies': int((images > 1).sum()),
        'single_image_study_share': float((images == 1).mean())}


def quality(source: pd.DataFrame, fit: pd.DataFrame, valid: pd.DataFrame, config: dict) -> dict:
    from .patient_holdout import summarize
    tables = {'source_train': source, 'internal_train': fit, 'internal_valid': valid}
    stats = {k: summarize(v) for k,v in tables.items()}
    structs = {k: structure(v) for k,v in tables.items()}
    source_stats = stats['source_train']; total_patients = source_stats['patients']
    checks = {}; details = {}
    for s, part in [('internal_train', fit), ('internal_valid', valid)]:
        actual_fraction = stats[s]['patients']/total_patients
        prevalence = np.array(stats[s]['positive_study_prevalence'])-source_stats['positive_study_prevalence']
        patient_expected = np.array(source_stats['positive_patients'])*actual_fraction
        patient_relative = np.abs(np.array(stats[s]['positive_patients'])-patient_expected)/np.maximum(patient_expected, 1)
        negative_patients = [(part.loc[part.Predict_class.map(lambda x: parse_labels(x)[i] == 0),'Subject_id'].nunique()) for i in range(10)]
        volume = {k: stats[s][k]/source_stats[k]-actual_fraction for k in ('studies','images')}
        single_delta = structs[s]['single_image_study_share']-structs['source_train']['single_image_study_share']
        checks[s+'_prevalence'] = bool(np.max(np.abs(prevalence)) <= config['max_prevalence_absolute_delta'])
        checks[s+'_positive_patient_counts'] = bool(np.max(patient_relative) <= config['max_positive_patient_target_relative_delta'])
        checks[s+'_minimum_patient_coverage'] = bool(min(stats[s]['positive_patients']+negative_patients) >= config['minimum_positive_negative_patients'])
        checks[s+'_volume'] = bool(max(map(abs,volume.values())) <= config['max_volume_fraction_delta'])
        checks[s+'_single_multi'] = bool(abs(single_delta) <= config['max_single_image_share_delta'])
        details[s] = {'prevalence_delta': prevalence.tolist(), 'positive_patient_relative_target_error': patient_relative.tolist(),
            'negative_patients': list(map(int,negative_patients)), 'volume_fraction_delta': volume,
            'single_image_share_delta': single_delta}
    overlap = {k: len(set(fit[k]) & set(valid[k])) for k in ('Subject_id','Study_id','Dicom_id')}
    checks['no_overlap'] = not any(overlap.values())
    checks['full_coverage'] = all(set(fit[k]) | set(valid[k]) == set(source[k]) for k in overlap)
    checks['exact_patient_quota'] = stats['internal_valid']['patients'] == math.ceil(total_patients*config['validation_fraction'])
    checks['patient_structure_bins'] = all(abs(structs['internal_valid']['patient_bins'][b]-
        structs['source_train']['patient_bins'][b]*stats['internal_valid']['patients']/total_patients) <= 1 for b in ['1','2-3','4-7','8+'])
    return {'qualified': all(checks.values()), 'checks': checks, 'details': details, 'statistics': stats,
        'structure': structs, 'overlap': overlap, 'thresholds': config,
        'meaning': '仅代表预定分层质量检查通过，不保证模型泛化或独立seed稳定性'}


def compare_splits(train_csv: Path, v1: Path, v2: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('质量对照目录已存在')
    source = read_csv(train_csv, True); records = {}; rows = []
    for version, folder in [('v1',v1), ('v2',v2)]:
        manifest = json.loads((folder/'output_manifest.json').read_text(encoding='utf-8'))['sha256']
        for name, digest in manifest.items():
            if Path(name).name != name or sha256(folder/name) != digest:
                raise ValueError('划分清单改变')
        record = quality(source, read_csv(folder/'internal_train.csv', True),
                         read_csv(folder/'internal_valid.csv', True), V2_CONFIG)
        records[version] = record
        for split, stats in record['statistics'].items():
            for i in range(10):
                rows.append({'version':version,'split':split,'class':i+1,
                    'positive_studies':stats['positive_studies'][i], 'positive_patients':stats['positive_patients'][i],
                    'positive_study_prevalence':stats['positive_study_prevalence'][i],
                    'prevalence_delta_from_source':stats['positive_study_prevalence'][i]-record['statistics']['source_train']['positive_study_prevalence'][i]})
    output.mkdir(parents=True)
    dump(output/'quality_comparison.json', records)
    save_csv(pd.DataFrame(rows), output/'class_distribution.csv')
    save_csv(pd.DataFrame([{'version':v,'split':s,**{k:x for k,x in data.items() if not isinstance(x,dict)}}
        for v,r in records.items() for s,data in r['structure'].items()]), output/'structure_comparison.csv')
    finish_manifest(output)


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-csv',type=Path,required=True);p.add_argument('--v1',type=Path,required=True)
    p.add_argument('--v2',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();compare_splits(a.train_csv,a.v1,a.v2,a.output)


if __name__ == '__main__': main()
