"""统一验证/测试预测入口；新建独立目录，不修改实验与权重。"""
import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import time
import warnings
import platform
import torchvision
from uuid import uuid4

import numpy as np
import pandas as pd
import torch

from .data import ChestXrayDataset, make_loader, normalization_for_settings
from .evaluate import study_metrics
from .experiments import sha256
from .model import load_model
from .submission import aggregate_studies, make_submission, check_submission, PROB_COLUMNS
from .train import seed_everything
from .devices import select_device, synchronize
from .prediction_outputs import image_prediction_table


def prediction_config(model_path: Path, data_config: Path | None = None) -> tuple[dict, str]:
    """优先使用检查点内的训练元数据；兼容旧模型的同目录配置。"""
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    config = checkpoint.get('metadata', {})
    if all(key in config for key in ('size','data','preprocess','batch_size','seed')):
        config, source = copy.deepcopy(config), 'checkpoint.metadata'
    else:
        companion = model_path.parent / 'train_config.json'
        if not companion.is_file():
            raise ValueError('模型未携带完整配置且同目录无 train_config.json')
        external = json.loads(companion.read_text(encoding='utf-8'))
        if 'size' in config and config['size'] != external['size']:
            raise ValueError('模型与配置的图片尺寸冲突')
        config, source = external, str(companion)
    if data_config is not None:
        override = json.loads(data_config.read_text(encoding='utf-8'))
        if not isinstance(override,dict) or set(override)-{'size','batch_size','train','valid','test'}:
            raise ValueError('数据配置只允许路径，以及用于核对的size和不生效的batch_size')
        if 'size' in override and override['size'] != config['size']:
            raise ValueError('路径覆盖不能更改模型输入尺寸')
        for split in ('train','valid','test'):
            if split in override:
                paths = override[split]
                if not isinstance(paths,dict) or set(paths) != {'csv','images'} or not all(isinstance(v,str) and v.strip() for v in paths.values()):
                    raise ValueError(f'{split}必须只含csv和images两个非空路径')
                config['data'][split] = dict(paths)
    return config, source


def _compare_reference(studies: pd.DataFrame, metrics: dict, folder: Path,
                      strict: bool, metric_atol: float, probability_atol: float) -> None:
    reference_path, previous_path = folder / 'validation_metrics.json', folder / 'val_study_predictions.csv'
    if not reference_path.is_file() or not previous_path.is_file():
        metrics['reference_comparison'] = 'unavailable'
        if strict:
            raise RuntimeError('严格复现模式要求同目录历史验证指标和预测')
        return
    reference = json.loads(reference_path.read_text(encoding='utf-8'))
    previous = pd.read_csv(previous_path, dtype={'Subject_id':str,'Study_id':str}, encoding='utf-8')
    for name in ('macro_auc','macro_f1'):
        value = reference[name]
        if value is None and name == 'macro_auc':
            continue
        if not isinstance(value,(float,int)) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('历史参考指标包含非法值')
    numeric = previous[PROB_COLUMNS].apply(pd.to_numeric,errors='raise').to_numpy()
    if not np.isfinite(numeric).all() or ((numeric < 0) | (numeric > 1)).any():
        raise ValueError('历史参考概率非法')
    previous[PROB_COLUMNS] = numeric
    keys = ['Subject_id','Study_id']
    aligned = studies.merge(previous, on=keys, suffixes=('_new','_old'), validate='one_to_one', how='outer', indicator=True)
    same_ids = bool(aligned['_merge'].eq('both').all())
    difference = float(np.max(np.abs(aligned[[p+'_new' for p in PROB_COLUMNS]].to_numpy()-aligned[[p+'_old' for p in PROB_COLUMNS]].to_numpy()))) if same_ids else None
    auc_difference = metrics['macro_auc'] - reference['macro_auc'] if metrics['macro_auc'] is not None and reference['macro_auc'] is not None else None
    f1_difference = metrics['macro_f1'] - reference['macro_f1']
    matched = same_ids and auc_difference is not None and abs(auc_difference) <= metric_atol and abs(f1_difference) <= metric_atol and difference <= probability_atol
    identity_matches = None
    if reference.get('validation_fingerprint') and metrics.get('validation_fingerprint'):
        identity_matches = (reference['validation_fingerprint'] == metrics['validation_fingerprint'] and
                            reference.get('evaluation_protocol') == metrics.get('evaluation_protocol'))
        matched = matched and identity_matches
    metrics.update(reference_macro_auc=reference['macro_auc'], macro_auc_difference=auc_difference,
                   macro_f1_difference=f1_difference, max_probability_difference=difference,
                   reference_ids_match=same_ids, reference_identity_matches=identity_matches,
                   reference_matches=bool(matched),
                   reference_comparison='matched' if matched else 'different',
                   comparison_tolerances={'metric_atol':metric_atol,'probability_atol':probability_atol})
    if not matched:
        message = '验证复测与历史有差异，详见 validation_metrics.json；跨环境数值差异不等同于格式错误'
        if strict:
            raise RuntimeError(message)
        warnings.warn(message, RuntimeWarning)


def compare_reference(studies: pd.DataFrame, metrics: dict, folder: Path,
                      strict: bool, metric_atol: float, probability_atol: float) -> None:
    """只容错可选历史参考文件，当前数据的校验仍在主流程中严格执行。"""
    try:
        _compare_reference(studies,metrics,folder,strict,metric_atol,probability_atol)
    except (OSError,ValueError,KeyError,TypeError) as exc:
        metrics.update(reference_comparison='invalid',reference_matches=False,
                       reference_error={'type':type(exc).__name__,'message':str(exc)})
        if strict:
            raise RuntimeError('严格复现失败：历史参考文件损坏或格式不正确') from exc
        warnings.warn('历史参考文件损坏，跳过比较并继续预测：'+str(exc),RuntimeWarning)


def dump(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def run(model_path: Path, split: str, submission_mode: str | None = None,
        threshold: float | None = None, strict_reproduce: bool = False,
        metric_atol: float = 1e-4, probability_atol: float = 1e-4,
        device_name: str = 'auto', template: Path | None = None,
        allow_nonofficial_counts: bool = False, data_config: Path | None = None) -> Path:
    if split in ('test','both') and submission_mode not in ('all_classes','threshold'):
        raise ValueError('测试输出必须显式选择 --all-classes 或 --threshold；官方语义尚未确认')
    if submission_mode == 'threshold' and (threshold is None or not np.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError('阈值必须在 [0,1] 内')
    if not np.isfinite([metric_atol,probability_atol]).all() or min(metric_atol,probability_atol) < 0:
        raise ValueError('复现容差必须是非负有限数值')
    columns = None
    if template:
        columns = list(pd.read_csv(template, nrows=0, encoding='utf-8-sig').columns)
        if len(columns) != 4 or set(columns) != {'Subject_id','Study_id','Predict_class','Probability'}:
            raise ValueError('模板包含不支持的字段（如 Gate），请先确认官方语义，不能猜填')
    folder = model_path.parent
    config, config_source = prediction_config(model_path,data_config)
    normalization = normalization_for_settings(config)
    if config.get('threshold') != 0.5:
        raise ValueError('归一化记录与检查点预处理冲突，或阈值不是0.5')
    if config.get('aggregation') != 'mean image probabilities per Study_id':
        raise ValueError('检查聚合配置不支持')
    seed_everything(config['seed'])
    torch.set_num_threads(4)
    device = select_device(device_name)
    before = sha256(model_path)
    model_start = time.perf_counter()
    model = load_model(str(model_path), device)
    synchronize(device)
    model_load_seconds = time.perf_counter() - model_start
    output = folder / 'predictions' / (datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    report = {'model': str(model_path.resolve()), 'model_sha256': before,
              'model_bytes': model_path.stat().st_size, 'model_under_500MB': model_path.stat().st_size <= 500_000_000,
              'device': device, 'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None,
              'batch_size': config['batch_size'], 'size': config['size'], 'model_load_seconds': model_load_seconds,
              'official_T4_tested': False, 'splits': {}}
    report.update(config_source=config_source, submission_mode=submission_mode,
                  submission_threshold=threshold, strict_reproduce=strict_reproduce,
                  data_config_override=str(data_config.resolve()) if data_config else None,
                  effective_data_paths=config['data'],
                  nonofficial_counts_allowed=allow_nonofficial_counts,
                  runtime={'python':platform.python_version(),'torch':str(torch.__version__),
                           'torchvision':str(torchvision.__version__),'cudnn':torch.backends.cudnn.version()})
    log = output / 'prediction_log.jsonl'
    try:
        for name in (('valid', 'test') if split == 'both' else (split,)):
            started = time.perf_counter()
            data = config['data'][name]
            dataset = ChestXrayDataset(data['csv'], data['images'], labeled=name == 'valid', size=config['size'],
                                       normalization=normalization)
            expected = (1422,859) if name == 'valid' else (2218,1414)
            if not allow_nonofficial_counts and (len(dataset), dataset.frame.Study_id.nunique()) != expected:
                raise ValueError('官方数据规模不符')
            expected = (len(dataset), dataset.frame.Study_id.nunique())
            loader = make_loader(dataset, config['batch_size'], num_workers=0)
            records, total_loss, count = [], 0.0, 0
            inference_start = time.perf_counter()
            with torch.no_grad():
                for step, batch in enumerate(loader, 1):
                    images = batch['image'].to(device)
                    logits = model(images)
                    if logits.shape != (len(images),10) or not torch.isfinite(logits).all():
                        raise ValueError('预测输出非法')
                    if name == 'valid':
                        total_loss += torch.nn.functional.binary_cross_entropy_with_logits(logits, batch['label'].to(device)).item() * len(images)
                    for image_id, probs in zip(batch['Dicom_id'], logits.sigmoid().cpu().numpy()):
                        records.append({'Dicom_id': image_id, **dict(zip(PROB_COLUMNS, map(float, probs)))})
                    count += len(images)
                    if step == 1 or step % 25 == 0 or step == len(loader):
                        event = {'split': name, 'step': step, 'images': count, 'elapsed_seconds': time.perf_counter()-inference_start}
                        with log.open('a', encoding='utf-8') as handle:
                            handle.write(json.dumps(event) + '\n')
                        print(json.dumps(event), flush=True)
            synchronize(device)
            seconds = time.perf_counter() - inference_start
            image_predictions = pd.DataFrame(records)
            studies = aggregate_studies(dataset.frame, image_predictions)
            if name == 'valid':
                scored, metrics = study_metrics(dataset.frame, image_predictions)
                aligned = scored.merge(studies, on=['Subject_id','Study_id'], suffixes=('_a','_b'), validate='one_to_one')
                np.testing.assert_allclose(aligned[[p+'_a' for p in PROB_COLUMNS]], aligned[[p+'_b' for p in PROB_COLUMNS]], rtol=0, atol=1e-15)
                studies = scored
                metrics['validation_loss'] = total_loss/count
                try:
                    compare_reference(studies,metrics,folder,strict_reproduce,metric_atol,probability_atol)
                except RuntimeError:
                    dump(output / 'validation_metrics.json', metrics)
                    raise
                dump(output / 'validation_metrics.json', metrics)
            image_table = image_prediction_table(dataset.frame, image_predictions)
            image_path = output / f'{name}_image_predictions.csv'
            image_table.to_csv(image_path, index=False, encoding='utf-8')
            image_reread = pd.read_csv(image_path, dtype=str, keep_default_na=False, encoding='utf-8')
            reconstructed = aggregate_studies(dataset.frame, image_reread[['Dicom_id', *PROB_COLUMNS]].assign(
                **{p: pd.to_numeric(image_reread[p], errors='raise') for p in PROB_COLUMNS}))
            matched_images = studies[['Subject_id','Study_id']].merge(reconstructed,
                on=['Subject_id','Study_id'],how='left',validate='one_to_one',sort=False)
            np.testing.assert_allclose(matched_images[PROB_COLUMNS], studies[PROB_COLUMNS], rtol=0, atol=1e-15)
            if not image_reread[['Dicom_id','Subject_id','Study_id']].equals(
                    image_table[['Dicom_id','Subject_id','Study_id']]):
                raise ValueError('图片级CSV回读ID错位')
            path = output / ('val_study_predictions.csv' if name == 'valid' else 'test_study_predictions.csv')
            studies.to_csv(path, index=False, encoding='utf-8')
            reread = pd.read_csv(path, dtype={'Subject_id':str,'Study_id':str}, encoding='utf-8')
            if len(reread) != expected[1] or reread.Study_id.duplicated().any():
                raise ValueError('预测 CSV 回读数量异常')
            np.testing.assert_allclose(reread[PROB_COLUMNS], studies[PROB_COLUMNS], rtol=0, atol=1e-15)
            info = {'images': count, 'studies': len(studies), 'decode_failures': 0,
                    'probability_min': float(studies[PROB_COLUMNS].min().min()),
                    'probability_max': float(studies[PROB_COLUMNS].max().max()),
                    'decode_preprocess_transfer_forward_seconds': seconds,
                    'amortized_ms_per_image': seconds/count*1000,
                    'split_total_seconds': time.perf_counter()-started, 'csv_reread_passed': True,
                    'image_csv_reread_passed': True, 'image_to_study_mean_verified': True}
            report['splits'][name] = info
            if name == 'test':
                submission_path = output / 'submission.csv'
                cutoff = threshold if submission_mode == 'threshold' else None
                submitted = make_submission(studies, cutoff)
                submitted[columns or list(submitted.columns)].to_csv(submission_path, index=False, encoding='utf-8')
                check = check_submission(submission_path, dataset.frame, cutoff, studies, columns)
                check['official_template_found'] = template is not None
                check['template_path'] = str(template) if template else None
                check['template_validation_scope'] = 'column names and order only; semantics require organizer confirmation'
                check.update(model_bytes=report['model_bytes'], model_under_500MB=report['model_under_500MB'],
                             inference=info, official_T4_tested=False)
                dump(output / 'submission_check.json', check)
        report['model_unchanged'] = before == sha256(model_path)
        if not report['model_unchanged']:
            raise RuntimeError('模型权重发生变化')
        dump(output / 'prediction_report.json', report)
        print('COMPLETE ' + str(output.resolve()), flush=True)
        return output
    except Exception as exc:
        dump(output / 'failure.json', {'error': str(exc), 'type': type(exc).__name__})
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--split', choices=['valid','test','both'], default='both')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--all-classes', action='store_true')
    modes.add_argument('--threshold', type=float)
    parser.add_argument('--strict-reproduce', action='store_true')
    parser.add_argument('--metric-atol', type=float, default=1e-4)
    parser.add_argument('--probability-atol', type=float, default=1e-4)
    parser.add_argument('--device', choices=['auto','cuda','mps','cpu'], default='auto')
    parser.add_argument('--template', type=Path)
    parser.add_argument('--data-config',type=Path,help='仅覆盖CSV/图片路径；保留检查点预处理及batch_size')
    parser.add_argument('--allow-nonofficial-counts', action='store_true', help='仅限小样本开发测试，记录到报告')
    args = parser.parse_args()
    if args.split != 'valid' and not args.all_classes and args.threshold is None:
        parser.error('请根据官方确认结果选择 --all-classes 或 --threshold；没有默认提交模式')
    run(args.model, args.split, 'all_classes' if args.all_classes else 'threshold' if args.threshold is not None else None,
        args.threshold, args.strict_reproduce, args.metric_atol, args.probability_atol,
        args.device, args.template, args.allow_nonofficial_counts,args.data_config)


if __name__ == '__main__':
    main()
