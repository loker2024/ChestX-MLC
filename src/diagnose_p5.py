"""P5验证集诊断：优先缓存；可在服务器显式加载P5，仅推理官方valid。"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .analysis_common import (P5_SHA, PROBS, VALID_FINGERPRINT, dump, finish_manifest,
                              provenance, read_predictions, save_csv, sha256)
from .data import parse_labels, read_csv
from .evaluate import score_study_predictions
from .fingerprints import dataset_fingerprint
from .aggregation import aggregate_predictions

DEFAULT_CONFIG = {'expected_model_sha256': P5_SHA, 'seed': 42, 'top_k': 25,
                  'focus_classes': [1, 4, 8], 'bootstrap_replicates': 1000,
                  'bootstrap_min_subjects': 20, 'bootstrap_min_positive_subjects': 5,
                  'bootstrap_min_negative_subjects': 5}


def validate_config(config: dict) -> dict:
    import re
    if set(config) != set(DEFAULT_CONFIG) or config['focus_classes'] != [1, 4, 8]:
        raise ValueError('诊断配置字段不完整或focus_classes不是固定1/4/8')
    if not isinstance(config['expected_model_sha256'], str) or not re.fullmatch(r'[0-9a-f]{64}', config['expected_model_sha256']):
        raise ValueError('必须指定完整模型SHA-256')
    for key in ('seed', 'top_k', 'bootstrap_replicates', 'bootstrap_min_subjects',
                'bootstrap_min_positive_subjects', 'bootstrap_min_negative_subjects'):
        if type(config[key]) is not int or config[key] < (0 if key in ('seed', 'bootstrap_replicates') else 1):
            raise ValueError(key+'须为合法整数')
    if config['seed'] >= 2**32 or config['bootstrap_replicates'] > 10000:
        raise ValueError('seed或Bootstrap次数超出范围')
    return dict(config)


def cluster_bootstrap(table: pd.DataFrame, config: dict) -> dict:
    """患者有放回抽样，携带全部Study；不把重复抽样Study误当成ID错误。"""
    groups = [g.index.to_numpy() for _, g in table.reset_index(drop=True).groupby('Subject_id', sort=True)]
    truth = table[[f'y{i}' for i in range(1, 11)]].to_numpy()
    probabilities = table[PROBS].to_numpy()
    reps = config['bootstrap_replicates']
    result = {'method': 'Subject_id cluster percentile bootstrap', 'seed': config['seed'],
              'replicates_requested': reps, 'subjects': len(groups), 'confidence_level': .95,
              'selection_bias_corrected': False, 'class_auc': {}}
    positive_patients = [sum(bool(truth[g, i].any()) for g in groups) for i in range(10)]
    negative_patients = [sum(bool((truth[g, i] == 0).any()) for g in groups) for i in range(10)]
    eligible = [positive_patients[i] >= config['bootstrap_min_positive_subjects'] and
                negative_patients[i] >= config['bootstrap_min_negative_subjects'] for i in range(10)]
    enough = len(groups) >= config['bootstrap_min_subjects'] and reps > 0
    samples = [[] for _ in range(10)]; macros = []
    rng = np.random.default_rng(config['seed'])
    if enough:
        for _ in range(reps):
            index = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
            aucs = []
            for i in range(10):
                if eligible[i] and len(np.unique(truth[index, i])) == 2:
                    value = float(roc_auc_score(truth[index, i], probabilities[index, i]))
                    samples[i].append(value); aucs.append(value)
                else:
                    aucs.append(None)
            if all(value is not None for value in aucs):
                macros.append(float(np.mean(aucs)))
    def summary(values: list, allowed: bool) -> dict:
        minimum = max(50, math.ceil(reps*.8))
        defined = len(values)
        if not enough:
            reason = 'disabled' if reps == 0 else 'insufficient_subjects'
        elif not allowed:
            reason = 'insufficient_positive_or_negative_patient_clusters'
        elif defined < minimum:
            reason = 'too_many_undefined_replicates_or_less_than_50_replicates'
        else:
            reason = None
        return {'defined_replicates': defined, 'undefined_or_not_run_replicates': reps-defined,
                'percentile_95': np.quantile(values, [.025, .975]).tolist() if reason is None else None,
                'reason': reason}
    result['macro_auc'] = summary(macros, all(eligible))
    for i in range(10):
        result['class_auc'][str(i+1)] = {**summary(samples[i], eligible[i]),
            'positive_subjects': positive_patients[i], 'negative_subjects': negative_patients[i]}
    return result


def predict_validation(model_path: Path, valid_csv: Path, images: Path,
                       config: dict, device_name: str = 'cpu') -> tuple[pd.DataFrame, dict]:
    """显式服务器路径才访问Checkpoint；沿用原Dataset、加载与Sigmoid/G0，不调用test入口。"""
    import torch
    import torchvision
    from .data import ChestXrayDataset, make_loader, normalization_for_settings
    from .devices import select_device
    from .model import load_model
    from .predict import prediction_config
    from .train import seed_everything
    actual = sha256(model_path)
    if actual != config['expected_model_sha256']:
        raise ValueError('模型SHA不是指定P5最佳权重，禁止静默换模型')
    settings, source = prediction_config(model_path)
    if (settings.get('backbone'), settings.get('size'),
            settings.get('completed_epoch', settings.get('epochs')),
            settings.get('aggregation')) != ('densenet121', 320, 6, 'mean image probabilities per Study_id'):
        raise ValueError('P5架构、尺寸、Epoch或G0聚合记录不匹配')
    normalization = normalization_for_settings(settings)
    if normalization != 'imagenet':
        raise ValueError('P5应使用ImageNet归一化')
    device = select_device(device_name)
    seed_everything(settings['seed']); torch.set_num_threads(4)
    ds = ChestXrayDataset(valid_csv, images, size=320, normalization=normalization)
    # 诊断优先稳定映射；与既有predict.py相同显式worker0，不改变训练worker设置。
    loader = make_loader(ds, settings['batch_size'], num_workers=0, seed=settings['seed'])
    model = load_model(str(model_path), device); model.eval()
    records = []
    with torch.inference_mode():
        for batch in loader:
            logits = model(batch['image'].to(device))
            if logits.shape != (len(batch['image']), 10) or not torch.isfinite(logits).all():
                raise ValueError('验证输出不是有限[B,10]logits')
            for identifier, values in zip(batch['Dicom_id'], logits.sigmoid().cpu().numpy()):
                records.append({'Dicom_id': identifier, **dict(zip(PROBS, map(float, values)))})
    if sha256(model_path) != actual:
        raise RuntimeError('模型源文件发生变化')
    return pd.DataFrame(records), {'status': 'checkpoint_sha_verified_before_and_after',
        'model_sha256': actual, 'model_loaded': True, 'config_source': source, 'backbone': 'densenet121', 'epoch': 6,
        'size': 320, 'normalization': normalization, 'aggregation': 'G0', 'num_workers': 0,
        'device': device, 'torchvision': str(torchvision.__version__), 'training_started': False}


def build_diagnostics(frame: pd.DataFrame, studies: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    scored, metrics = score_study_predictions(frame, studies)
    by_study = frame.groupby('Study_id', sort=True)
    table = scored.copy()
    labels = np.stack([parse_labels(by_study.get_group(s).Predict_class.iloc[0]) for s in table.Study_id])
    for i in range(10):
        table[f'y{i+1}'] = labels[:, i].astype(int)
    table['image_count'] = table.Study_id.map(by_study.size())
    table['Dicom_ids'] = table.Study_id.map(by_study.Dicom_id.apply(lambda x: json.dumps(sorted(x), ensure_ascii=False)))
    table['ViewPosition'] = table.Study_id.map(by_study.ViewPosition.apply(
        lambda x: json.dumps(sorted(set(value if value.strip() else '<missing>' for value in x)), ensure_ascii=False)))
    table['subject_study_count'] = table.groupby('Subject_id').Study_id.transform('size')
    probs = table[PROBS].to_numpy(); clipped = np.clip(probs, 1e-7, 1-1e-7)
    table['mean_image_independent_study_bce'] = -(labels*np.log(clipped)+(1-labels)*np.log1p(-clipped)).mean(1)
    # 这里是聚合后的Study概率诊断BCE，不是训练Study均衡逐图BCE。
    table = table.rename(columns={'mean_image_independent_study_bce': 'study_probability_bce'})
    table['mean_brier'] = ((labels-probs)**2).mean(1)
    return table, metrics


def diagnose(valid_csv: Path, output: Path, config: dict, *, study_predictions: Path | None = None,
             image_predictions: Path | None = None, model_path: Path | None = None,
             images: Path | None = None, reference_metrics: Path | None = None,
             device_name: str = 'cpu', require_official: bool = True) -> Path:
    config = validate_config(config)
    if sum(value is not None for value in (study_predictions, image_predictions, model_path)) != 1:
        raise ValueError('只允许一种预测来源：逐Study缓存、逐图缓存或模型')
    if output.exists():
        raise FileExistsError('诊断目录已存在，拒绝覆盖')
    inputs = {'official_valid': valid_csv}
    for key, path in [('study_predictions', study_predictions), ('image_predictions', image_predictions),
                      ('checkpoint', model_path), ('reference_metrics', reference_metrics)]:
        if path is not None:
            inputs[key] = path
    origin = provenance(inputs, {**config, 'require_official': require_official, 'device_requested': device_name,
        'valid_images_root': str(images.resolve()) if images is not None else None})
    frame = read_csv(valid_csv, True)
    if require_official and ((len(frame), frame.Study_id.nunique()) != (1422, 859) or dataset_fingerprint(frame) != VALID_FINGERPRINT):
        raise ValueError('只允许完整官方valid及其固定标签指纹，不接受test或内部集冒充')
    source = {'status': 'unverified_cached_model_origin', 'model_loaded': False,
              'note': '缓存身份/标签归属可验证，但仅凭概率文件不能证明它来自P5'}
    if study_predictions is not None:
        studies = read_predictions(study_predictions, 'study')
        inputs['study_predictions'] = study_predictions
    else:
        if model_path is not None:
            if images is None:
                raise ValueError('模型推理模式必须提供--valid-images')
            predictions, source = predict_validation(model_path, valid_csv, images, config, device_name)
            inputs['checkpoint'] = model_path
        else:
            predictions = read_predictions(image_predictions, 'image')
            inputs['image_predictions'] = image_predictions
        studies = aggregate_predictions(frame, predictions, 'G0')
    table, metrics = build_diagnostics(frame, studies)
    if reference_metrics is not None:
        ref = json.loads(reference_metrics.read_text(encoding='utf-8-sig'))
        if ref.get('checkpoint_sha256') != config['expected_model_sha256']:
            raise ValueError('参考指标绑定的模型不是指定P5')
        if ref.get('validation_fingerprint') != metrics['validation_fingerprint']:
            raise ValueError('参考验证标签/ID指纹不一致')
        for name in ('macro_auc', 'macro_f1'):
            if ref.get(name) is None or metrics[name] is None or abs(ref[name]-metrics[name]) > 1e-8:
                raise ValueError('缓存重算与参考'+name+'不一致')
        inputs['reference_metrics'] = reference_metrics
        source['reference_metrics_consistent'] = True
        source['note'] = '参考指标及其模型SHA一致；缓存模式仍不是对Checkpoint的独立推理复测'
    image_index = frame[['Subject_id', 'Study_id', 'Dicom_id', 'ViewPosition',
                         'ViewPosition_missing', 'PatientOrientation', 'Rows', 'Columns']].copy()
    if images is not None:
        from .data import index_images
        mapping = index_images(images)
        if not set(frame.Dicom_id).issubset(mapping):
            raise FileNotFoundError('验证图片路径匹配不完整')
        image_index['image_path'] = [str(mapping[x].resolve()) for x in frame.Dicom_id]
        path_status = 'all_valid_image_paths_resolved; pixels_not_inspected_in_cache_mode'
    else:
        image_index['image_path'] = ''
        path_status = 'not_checked_no_image_root_supplied; Dicom/Study/Subject_checked_against_valid_csv'
    class_rows = []
    for i in range(1, 11):
        positive = table[f'y{i}'].eq(1); values = table[f'p{i}']
        class_rows.append({'class_id': i, 'positive_studies': int(positive.sum()),
            'negative_studies': int((~positive).sum()),
            'positive_subjects': int(table.loc[positive, 'Subject_id'].nunique()),
            'negative_subjects': int(table.loc[~positive, 'Subject_id'].nunique()),
            'auc': metrics['class_auc'][str(i)],
            'average_precision': float(average_precision_score(positive, values)) if positive.any() else None,
            'mean_positive_probability': float(values[positive].mean()) if positive.any() else None,
            'mean_negative_probability': float(values[~positive].mean()) if (~positive).any() else None,
            'predicted_positive_at_05': int(values.ge(.5).sum())})
    subgroup_rows = []
    masks = {'single_image': table.image_count.eq(1), 'multi_image': table.image_count.gt(1),
             'subject_1_study': table.subject_study_count.eq(1),
             'subject_2_to_3_studies': table.subject_study_count.between(2, 3),
             'subject_4plus_studies': table.subject_study_count.ge(4)}
    for name, mask in masks.items():
        part = table[mask]; aucs = {}
        for i in range(1, 11):
            aucs[str(i)] = float(roc_auc_score(part[f'y{i}'], part[f'p{i}'])) if part[f'y{i}'].nunique() == 2 else None
        undefined = [int(k) for k, value in aucs.items() if value is None]
        subgroup_rows.append({'group': name, 'studies': len(part), 'subjects': part.Subject_id.nunique(),
            'macro_auc': float(np.mean(list(aucs.values()))) if not undefined else None,
            'undefined_auc_classes': json.dumps(undefined),
            'mean_brier': float(part.mean_brier.mean()) if len(part) else None,
            **{f'class_{i}_auc': aucs[str(i)] for i in range(1, 11)}})
    subject_summary = table.groupby('Subject_id', sort=True).agg(
        study_count=('Study_id', 'size'), image_count=('image_count', 'sum'),
        mean_study_probability_bce=('study_probability_bce', 'mean'), mean_brier=('mean_brier', 'mean')).reset_index()
    for i in range(1, 11):
        subject_summary[f'positive_studies_class_{i}'] = subject_summary.Subject_id.map(table.groupby('Subject_id')[f'y{i}'].sum())
    correlation = (float(subject_summary.study_count.corr(subject_summary.mean_brier, method='spearman'))
                   if subject_summary.study_count.nunique() > 1 and subject_summary.mean_brier.nunique() > 1 else None)
    if correlation is not None and not math.isfinite(correlation):
        correlation = None
    bootstrap = cluster_bootstrap(table, config)
    metrics.update(source_identity=source, image_path_verification=path_status,
        probability_columns=PROBS, subjects=int(table.Subject_id.nunique()),
        subject_study_count_vs_mean_brier_spearman=correlation,
        interpretation='低概率阳性/高概率阴性只是候选；AUC排序取决于同类正负样本相对次序。子组关系非因果。',
        fixture_only=not require_official, test_accessed=False, training_started=False)
    output.mkdir(parents=True, exist_ok=False)
    try:
        save_csv(table[['Subject_id', 'Study_id', *PROBS]], output/'val_study_predictions.csv')
        save_csv(table, output/'study_diagnostics.csv'); save_csv(image_index, output/'study_image_index.csv')
        save_csv(pd.DataFrame(class_rows), output/'class_metrics.csv')
        save_csv(pd.DataFrame(subgroup_rows), output/'subgroup_metrics.csv')
        save_csv(subject_summary, output/'subject_summary.csv')
        for i in config['focus_classes']:
            pos = table[table[f'y{i}'].eq(1)].copy(); neg = table[table[f'y{i}'].eq(0)].copy()
            pv, nv = pos[f'p{i}'].to_numpy(), neg[f'p{i}'].to_numpy()
            for part, kind, opposite, ascending in ((pos, 'low_probability_positive', nv, True),
                                                       (neg, 'high_probability_negative', pv, False)):
                part['class_id'] = i; part['true_label'] = part[f'y{i}']; part['probability'] = part[f'p{i}']
                part['candidate_kind'] = kind
                part['opposite_label_outrank_fraction'] = [float(((opposite > x) if ascending else (opposite < x)).mean()
                    + .5*(opposite == x).mean()) if len(opposite) else None for x in part.probability]
                part = part.sort_values(['probability', 'Study_id'], ascending=[ascending, True], kind='stable').head(config['top_k'])
                fields = ['Subject_id', 'Study_id', 'class_id', 'true_label', 'probability', 'candidate_kind',
                          'opposite_label_outrank_fraction', 'image_count', 'Dicom_ids', 'ViewPosition', 'subject_study_count']
                save_csv(part[fields], output/f'class_{i}_{kind}.csv')
        dump(output/'validation_metrics.json', metrics); dump(output/'bootstrap_intervals.json', bootstrap)
        dump(output/'provenance.json', origin); dump(output/'analysis_config.json', config)
        for item in origin['inputs'].values():
            if sha256(item['path']) != item['sha256']:
                raise RuntimeError('诊断输入文件在运行期间发生变化')
        finish_manifest(output)
        print(json.dumps({'output': str(output.resolve()), 'studies': len(table),
            'macro_auc': metrics['macro_auc'], 'source_identity': source, 'test_accessed': False}, ensure_ascii=False))
        return output
    except Exception as exc:
        dump(output/'failure.json', {'type': type(exc).__name__, 'error': str(exc)})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--valid-csv', type=Path, required=True)
    parser.add_argument('--valid-images', type=Path)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--study-predictions', type=Path)
    modes.add_argument('--image-predictions', type=Path)
    modes.add_argument('--model', type=Path)
    parser.add_argument('--reference-metrics', type=Path)
    parser.add_argument('--config', type=Path, default=Path('configs/analysis/p5_diagnosis.json'))
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps', 'auto'], default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    diagnose(args.valid_csv, args.output, json.loads(args.config.read_text(encoding='utf-8-sig')),
             study_predictions=args.study_predictions, image_predictions=args.image_predictions,
             model_path=args.model, images=args.valid_images, reference_metrics=args.reference_metrics,
             device_name=args.device)


if __name__ == '__main__':
    main()
