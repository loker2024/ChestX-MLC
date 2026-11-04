"""同一冻结患者划分的A/B自动对照；只读训练结果，不访问test或自动选正式模型。"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from .analysis_common import PROBS, dump, finish_manifest, read_predictions, save_csv, sha256, provenance
from .data import parse_labels, read_csv
from .evaluate import score_study_predictions
from .internal_ab import validate_config


def read_run(folder: Path) -> dict:
    if (folder / 'failure.json').exists():
        raise ValueError('实验存在failure.json，不能作为完成结果比较')
    manifest = json.loads((folder / 'output_manifest.json').read_text(encoding='utf-8'))['sha256']
    for name, digest in manifest.items():
        if Path(name).name != name or sha256(folder / name) != digest:
            raise ValueError('实验输出清单损坏：' + name)
    config = validate_config(json.loads((folder / 'experiment_config.json').read_text(encoding='utf-8')))
    audit = json.loads((folder / 'split_audit.json').read_text(encoding='utf-8'))
    summary = json.loads((folder / 'run_summary.json').read_text(encoding='utf-8'))
    metrics = json.loads((folder / 'validation_metrics.json').read_text(encoding='utf-8'))
    checkpoint = torch.load(folder / 'best_model.pth', map_location='cpu', weights_only=True)
    metadata = checkpoint['metadata']
    if checkpoint['architecture'] != 'densenet121' or checkpoint['num_classes'] != 10:
        raise ValueError('对照模型不是DenseNet121十类')
    history = pd.read_csv(folder / 'history.csv', encoding='utf-8')
    if (history.epoch.tolist() != list(range(1, len(history)+1)) or len(history) != summary['completed_epochs']
            or not np.isfinite(history[['macro_auc','macro_f1','train_loss','validation_loss']].to_numpy()).all()):
        raise ValueError('训练历史缺轮次或含非有限指标')
    best = history.loc[history.macro_auc.idxmax()]
    if (metadata['completed_epoch'] != int(best.epoch) or metrics['epoch'] != int(best.epoch)
            or summary['best_epoch'] != int(best.epoch) or sha256(folder / 'best_model.pth') != metrics['checkpoint_sha256']
            or metadata['experiment_config'] != config or metadata['split_audit'] != audit
            or metadata['initial_state_sha256'] != summary['initial_state_sha256']):
        raise ValueError('最佳权重、配置、轮次、初始模型或指标身份不一致')
    for key in ('macro_auc', 'macro_f1', 'validation_loss', 'train_loss'):
        if not np.isclose(metrics[key], best[key], rtol=0, atol=1e-12):
            raise ValueError('最佳指标没有对应同一个历史Epoch')
    snapshot = folder / 'split_snapshot'
    split_manifest = json.loads((snapshot / 'output_manifest.json').read_text(encoding='utf-8'))['sha256']
    for name, digest in split_manifest.items():
        if Path(name).name != name or sha256(snapshot / name) != digest:
            raise ValueError('冻结划分快照改变')
    if sha256(snapshot / 'output_manifest.json') != audit['split_manifest_sha256']:
        raise ValueError('划分快照与模型绑定不同')
    frame = read_csv(snapshot / 'internal_valid.csv', True)
    studies, recomputed = score_study_predictions(frame, read_predictions(folder / 'val_study_predictions.csv', 'study'))
    for key in ('macro_auc', 'macro_f1', 'class_auc', 'validation_fingerprint', 'evaluation_protocol'):
        if recomputed[key] != metrics[key]:
            # CSV十进制回读仅允许数值舍入误差；ID/协议不得容错。
            if key not in ('macro_auc', 'macro_f1') or not np.isclose(recomputed[key], metrics[key], rtol=0, atol=1e-12):
                raise ValueError('预测复算与保存指标不同：' + key)
    if recomputed['validation_fingerprint'] != audit['validation_fingerprint']:
        raise ValueError('内部验证ID/标签与冻结划分不同')
    return {'config': config, 'audit': audit, 'summary': summary, 'metrics': metrics, 'history': history,
            'metadata': metadata, 'frame': frame, 'studies': studies}


def paired_bootstrap(frame: pd.DataFrame, a: pd.DataFrame, b: pd.DataFrame,
                     replicates: int = 1000, seed: int = 42) -> dict:
    """同一患者重采样同时用于A/B，报告差值；不修正模型选择及训练随机性。"""
    if type(replicates) is not int or not 50 <= replicates <= 10000:
        raise ValueError('Bootstrap次数须为50—10000')
    truth = frame.groupby('Study_id', sort=True).first().reset_index()
    y = np.stack(truth.Predict_class.map(parse_labels))
    predictions = [table.set_index('Study_id').loc[truth.Study_id, PROBS].to_numpy(float) for table in (a, b)]
    groups = [g.index.to_numpy() for _, g in truth.groupby('Subject_id', sort=True)]
    positive = [sum(bool(y[g, i].any()) for g in groups) for i in range(10)]
    negative = [sum(bool((1-y[g, i]).any()) for g in groups) for i in range(10)]
    eligible = [min(positive[i], negative[i]) >= 5 for i in range(10)]
    deltas = [[] for _ in range(10)]; macros = []; weak = []
    rng = np.random.default_rng(seed)
    if len(groups) >= 20:
        for _ in range(replicates):
            selected = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
            values = []
            for i in range(10):
                if eligible[i] and np.unique(y[selected, i]).size == 2:
                    delta = float(roc_auc_score(y[selected, i], predictions[1][selected, i])
                                  - roc_auc_score(y[selected, i], predictions[0][selected, i]))
                    deltas[i].append(delta); values.append(delta)
                else:
                    values.append(None)
            if all(v is not None for v in values):
                macros.append(float(np.mean(values)))
            if all(values[i] is not None for i in (0, 3, 7)):
                weak.append(float(np.mean([values[i] for i in (0, 3, 7)])))
    def interval(values: list, allowed: bool) -> dict:
        reason = ('insufficient_subjects' if len(groups) < 20 else
                  'insufficient_positive_or_negative_subjects' if not allowed else
                  'too_many_undefined_replicates' if len(values) < max(50, math.ceil(.8*replicates)) else None)
        return {'defined_replicates': len(values), 'undefined_replicates': replicates-len(values),
                'percentile_95': np.quantile(values, [.025, .975]).tolist() if reason is None else None, 'reason': reason}
    return {'seed': seed, 'replicates': replicates, 'method': 'paired Subject_id cluster percentile bootstrap; B minus A',
        'selection_bias_corrected': False, 'training_seed_variability_estimated': False,
        'macro_auc_delta': interval(macros, all(eligible)), 'weak_1_4_8_delta': interval(weak, all(eligible[i] for i in (0, 3, 7))),
        'class_auc_delta': {str(i+1): {**interval(deltas[i], eligible[i]), 'positive_subjects': positive[i],
                                     'negative_subjects': negative[i]} for i in range(10)}}


def compare(a_path: Path, b_path: Path, output: Path, replicates: int = 1000) -> Path:
    if output.exists():
        raise FileExistsError('拒绝覆盖已有对照')
    a, b = read_run(a_path), read_run(b_path)
    differences = {k for k in a['config'] if a['config'][k] != b['config'][k]}
    if differences != {'experiment_id', 'backbone_lr'} or a['config']['backbone_lr'] != .0001 or b['config']['backbone_lr'] != .00003:
        raise ValueError('不是仅骨干LR不同的固定A/B对照，或A/B顺序错误')
    for key in ('train_fingerprint', 'validation_fingerprint', 'split_manifest_sha256', 'source_kind'):
        if a['audit'][key] != b['audit'][key]:
            raise ValueError('两组冻结划分/来源不一致：' + key)
    if a['summary']['initial_state_sha256'] != b['summary']['initial_state_sha256']:
        raise ValueError('A/B初始模型张量不同，不能作同起点对照')
    if a['metadata']['runtime'] != b['metadata']['runtime']:
        raise ValueError('A/B设备或Torch环境不同；本工具拒绝自动归为纯LR对照')
    if a['metadata']['data'] != b['metadata']['data']:
        raise ValueError('A/B实际CSV或图片根目录不同')
    pa = json.loads((a_path/'provenance.json').read_text(encoding='utf-8'))
    pb = json.loads((b_path/'provenance.json').read_text(encoding='utf-8'))
    if pa['code_sha256'] != pb['code_sha256'] or pa['runtime'] != pb['runtime']:
        raise ValueError('A/B代码或依赖版本不同')
    score_study_predictions(a['frame'], b['studies'])
    ci = paired_bootstrap(a['frame'], a['studies'], b['studies'], replicates)
    rows = []
    for name, run, folder in [('A', a, a_path), ('B', b, b_path)]:
        m = run['metrics']; s = run['summary']
        rows.append({'arm': name, 'best_epoch': s['best_epoch'], 'completed_epochs': s['completed_epochs'],
            'backbone_lr': run['config']['backbone_lr'], 'head_lr': run['config']['head_lr'],
            'macro_auc': m['macro_auc'], 'macro_f1': m['macro_f1'], 'train_loss': m['train_loss'],
            'validation_loss': m['validation_loss'], 'elapsed_seconds': s['elapsed_seconds'],
            'optimizer_steps_total': int(run['history'].optimizer_steps.sum()),
            'training_image_visits_total': int(run['history'].training_images.sum()),
            'weak_1_4_8_auc': float(np.mean([m['class_auc'][str(i)] for i in (1, 4, 8)])),
            'model_path': str((folder/'best_model.pth').resolve()), 'model_sha256': m['checkpoint_sha256']})
    per_class = pd.DataFrame([{'class': i, 'auc_A': a['metrics']['class_auc'][str(i)],
        'auc_B': b['metrics']['class_auc'][str(i)],
        'delta_B_minus_A': b['metrics']['class_auc'][str(i)]-a['metrics']['class_auc'][str(i)]} for i in range(1, 11)])
    delta = rows[1]['macro_auc']-rows[0]['macro_auc']; interval = ci['macro_auc_delta']['percentile_95']
    output.mkdir(parents=True, exist_ok=False)
    save_csv(pd.DataFrame(rows), output/'comparison.csv'); save_csv(per_class, output/'class_auc_comparison.csv')
    dump(output/'paired_bootstrap.json', ci)
    dump(output/'comparison.json', {'delta_macro_auc_B_minus_A': delta, 'minimum_practical_delta': .005,
        'passes_auc_screen': bool(delta >= .005 and interval is not None and interval[0] > 0),
        'automatic_model_promotion': False, 'source_kind': a['audit']['source_kind'],
        'note': '还需人工检查十类退化；单seed及被选最佳Epoch的区间不代表未知分布泛化保证',
        'same_initial_state': True, 'same_frozen_split': True, 'different_config_keys': sorted(differences),
        'best_checkpoint_sha256': {r['arm']: r['model_sha256'] for r in rows}})
    dump(output/'provenance.json', provenance({'A_manifest': a_path/'output_manifest.json',
        'B_manifest': b_path/'output_manifest.json'}, {'replicates': replicates, 'seed': 42}))
    finish_manifest(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a', type=Path, required=True); parser.add_argument('--b', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args(); print(compare(args.a, args.b, args.output, args.bootstrap))


if __name__ == '__main__':
    main()
