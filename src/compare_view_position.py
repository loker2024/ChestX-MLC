"""VP-01固定v2离线评价；需用户手动执行，无训练/图片读取/Checkpoint加载。"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis_common import read_predictions, sha256, dump, save_csv, provenance, finish_manifest
from .aggregation import aggregate_predictions
from .compare_internal_ab import paired_bootstrap
from .data import read_csv
from .evaluate import score_study_predictions
from .fingerprints import dataset_fingerprint, EVALUATION_PROTOCOL
from .submission import PROB_COLUMNS
from .view_position import aggregate_vp01, VP01_RULE

V2_MANIFEST_SHA256 = '5c8cc5a132454472254973f1ad3b22d800e05c0215660f6f41e4ddd1f05a6b91'


def load_v2(run: Path, split: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """仅允许已冻结的项目v2，不接受任意valid/test路径替换。"""
    manifest_path = split / 'output_manifest.json'
    if sha256(manifest_path) != V2_MANIFEST_SHA256:
        raise ValueError('不是本项目固定v2清单；禁止用官方valid/test选权重')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))['sha256']
    for name, value in manifest.items():
        if Path(name).name != name or sha256(split / name) != value:
            raise ValueError('冻结v2文件改变：' + name)
    audit = json.loads((run / 'split_audit.json').read_text(encoding='utf-8'))
    config = json.loads((run / 'train_config.json').read_text(encoding='utf-8'))
    metrics = json.loads((run / 'validation_metrics.json').read_text(encoding='utf-8'))
    summary = json.loads((run / 'run_summary.json').read_text(encoding='utf-8'))
    history = pd.read_csv(run / 'history.csv', encoding='utf-8')
    epoch = int(history.loc[history.macro_auc.idxmax(), 'epoch'])
    if ((run / 'failure.json').exists() or audit['split_manifest_sha256'] != V2_MANIFEST_SHA256
            or config['validation_mode'] != 'internal_subject_holdout'
            or config['aggregation'] != EVALUATION_PROTOCOL['aggregation']
            or config['size'] != 320 or config['augmentation'] != 'none'
            or epoch != metrics['epoch'] or epoch != summary['best_epoch'] or epoch != config['completed_epoch']):
        raise ValueError('实验非完成v2/G0/T0结果，或最佳轮不一致')
    for name in ('internal_train.csv', 'internal_valid.csv'):
        if sha256(run / 'split_snapshot' / name) != sha256(split / name):
            raise ValueError('实验划分不是同一固定v2')
    frame = read_csv(split / 'internal_valid.csv', True)
    if (len(frame), frame.Study_id.nunique(), frame.Subject_id.nunique()) != (4554, 2779, 1659):
        raise ValueError('固定v2规模不符')
    if dataset_fingerprint(frame) != audit['validation_fingerprint'] or metrics['validation_fingerprint'] != audit['validation_fingerprint']:
        raise ValueError('v2标签/ID指纹不匹配')
    if metrics['evaluation_protocol'] != EVALUATION_PROTOCOL:
        raise ValueError('历史参考不是原始G0协议')
    for name in ('valid_image_predictions.csv', 'val_study_predictions.csv'):
        epoch_copy = run / 'epochs' / f'epoch_{epoch}' / name
        if sha256(run / name) != sha256(epoch_copy):
            raise ValueError('预测不是最佳轮副本：' + name)
    return frame, read_predictions(run / 'valid_image_predictions.csv', 'image'), read_predictions(run / 'val_study_predictions.csv', 'study'), metrics


def evaluate_tables(frame: pd.DataFrame, predictions: pd.DataFrame, reference: pd.DataFrame,
                    output: Path, replicates: int = 1000) -> dict:
    """纯表格评价，合成测试可复用；正式CLI先经过固定v2门禁。"""
    if output.exists():
        raise FileExistsError('拒绝覆盖已有VP-01实验')
    g0, m0 = score_study_predictions(frame, aggregate_predictions(frame, predictions))
    ref, _ = score_study_predictions(frame, reference)
    matched = g0.merge(ref, on=['Subject_id', 'Study_id'], validate='one_to_one', suffixes=('_new', '_saved'))
    np.testing.assert_allclose(matched[[p+'_new' for p in PROB_COLUMNS]], matched[[p+'_saved' for p in PROB_COLUMNS]], rtol=0, atol=1e-12)
    vp, decisions, weights = aggregate_vp01(frame, predictions)
    protocol = {**EVALUATION_PROTOCOL, 'aggregation': 'VP-01 fixed front/side group probability mean', 'view_rule': VP01_RULE}
    vp, mv = score_study_predictions(frame, vp, protocol)
    ci = paired_bootstrap(frame, g0, vp, replicates, 42)
    ci['method'] = 'paired Subject_id cluster percentile bootstrap; VP-01 minus G0'
    output.mkdir(parents=True, exist_ok=False)
    for name, table in [('G0_study_predictions.csv', g0), ('VP01_study_predictions.csv', vp),
                         ('study_view_decisions.csv', decisions), ('image_weights.csv', weights)]:
        save_csv(table, output / name)
    rows = []
    for name, metrics in [('G0', m0), ('VP-01', mv)]:
        rows.append({'method': name, 'study_count': len(g0), 'macro_auc': metrics['macro_auc'],
            'macro_f1': metrics['macro_f1'], 'threshold': .5, **{'class_'+k+'_auc': v for k,v in metrics['class_auc'].items()}})
    save_csv(pd.DataFrame(rows), output / 'comparison.csv')
    save_csv(pd.DataFrame([{'class_id': i, 'auc_G0': m0['class_auc'][str(i)], 'auc_VP01': mv['class_auc'][str(i)],
        'delta': None if m0['class_auc'][str(i)] is None else mv['class_auc'][str(i)]-m0['class_auc'][str(i)]} for i in range(1,11)]), output / 'class_auc_comparison.csv')
    dump(output / 'metrics.json', {'G0': m0, 'VP01': mv})
    dump(output / 'paired_bootstrap.json', ci)
    coverage = {'images': len(frame), 'studies': len(g0), 'weighted_studies': int(decisions.weighted.sum()),
        'reasons': {k:int(v) for k,v in decisions.reason.value_counts().items()}, 'rule': VP01_RULE,
        'all_images_retained': True, 'g0_reproduced': True, 'training': False, 'inference': False,
        'test_accessed': False, 'weight_search': False, 'automatic_promotion': False}
    # 预测及逐图权重CSV回读，确认ID/概率与权重审计覆盖完整。
    for name, original in [('G0', g0), ('VP01', vp)]:
        reread = read_predictions(output / (name+'_study_predictions.csv'), 'study')
        pd.testing.assert_frame_equal(reread[['Subject_id','Study_id']], original[['Subject_id','Study_id']])
        np.testing.assert_allclose(reread[PROB_COLUMNS], original[PROB_COLUMNS], rtol=0, atol=1e-15)
    weights_read = pd.read_csv(output/'image_weights.csv', dtype={'Dicom_id':str,'Study_id':str,'Subject_id':str}, encoding='utf-8')
    if len(weights_read) != len(frame) or set(weights_read.Dicom_id) != set(frame.Dicom_id):
        raise ValueError('逐图权重CSV回读覆盖不完整')
    np.testing.assert_allclose(weights_read.groupby('Study_id').weight.sum(), 1, rtol=0, atol=1e-14)
    dump(output / 'coverage.json', coverage)
    delta = None if mv['macro_auc'] is None else mv['macro_auc']-m0['macro_auc']
    (output / 'report.md').write_text(
        '# VP-01离线结果\n\n固定正位/侧位组7:3，各组内平均；缺失/未知或无明确两组则整Study回退G0。\n\n'
        + pd.DataFrame(rows)[['method','study_count','macro_auc','macro_f1']].to_string(index=False)
        + f'\n\nVP-01−G0 Macro-AUC差值：{delta}；患者配对95%区间：{ci["macro_auc_delta"]["percentile_95"]}。\n'
        + f'\n应用加权Study数：{coverage["weighted_studies"]}；回退原因及全图片非零权重见CSV/coverage.json。\n'
        + '\n不搜索权重、不自动部署；单次内部留出和最佳轮选择仍有选型偏差，未测T4。\n', encoding='utf-8')
    return {'rows': rows, 'coverage': coverage, 'delta_macro_auc': delta}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=Path('outputs/Internal_AB_local_v2/A'))
    parser.add_argument('--split-directory', type=Path, default=Path('outputs/internal_subject_split_v2'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('输出已存在，拒绝覆盖')
    inputs = {'v2_manifest': args.split_directory/'output_manifest.json',
        'v2_valid': args.split_directory/'internal_valid.csv'}
    inputs.update({name:args.run_dir/name for name in ('valid_image_predictions.csv','val_study_predictions.csv',
        'train_config.json','validation_metrics.json','run_summary.json','history.csv','split_audit.json')})
    hashes = {str(p):sha256(p) for p in inputs.values()}
    frame, predictions, reference, metrics = load_v2(args.run_dir, args.split_directory)
    _, checked = score_study_predictions(frame, reference)
    for key in ('macro_auc', 'macro_f1'):
        np.testing.assert_allclose(checked[key], metrics[key], rtol=0, atol=1e-12)
    for i in range(1,11):
        np.testing.assert_allclose(checked['class_auc'][str(i)], metrics['class_auc'][str(i)], rtol=0, atol=1e-12)
    result = evaluate_tables(frame, predictions, reference, args.output, args.bootstrap)
    if hashes != {p:sha256(p) for p in hashes}:
        raise RuntimeError('离线评价过程中源文件改变')
    dump(args.output/'provenance.json', provenance(inputs, {'experiment':'VP-01','bootstrap':args.bootstrap,
        'seed':42,'rule':VP01_RULE,'source_files_unchanged':True,'checkpoint_loaded':False}))
    finish_manifest(args.output)
    print(pd.DataFrame(result['rows'])[['method','study_count','macro_auc','macro_f1']].to_string(index=False))
    print('COMPLETE '+str(args.output.resolve()))


if __name__ == '__main__':
    main()
