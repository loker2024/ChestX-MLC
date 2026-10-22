"""统一验证/测试预测入口；新建独立目录，不修改实验与权重。"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time
from uuid import uuid4

import numpy as np
import pandas as pd
import torch

from .data import ChestXrayDataset, make_loader
from .evaluate import study_metrics
from .experiments import sha256
from .model import load_model
from .submission import aggregate_studies, make_submission, check_submission, PROB_COLUMNS
from .train import seed_everything


def dump(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def run(model_path: Path, split: str) -> Path:
    folder = model_path.parent
    config = json.loads((folder / 'train_config.json').read_text(encoding='utf-8'))
    if config.get('preprocess') != 'RGB letterbox /255' or config.get('threshold') != 0.5:
        raise ValueError('此入口仅支持现有 RGB 等比例补边 /255 与0.5阈值配置')
    if config.get('aggregation') != 'mean image probabilities per Study_id':
        raise ValueError('检查聚合配置不支持')
    seed_everything(config['seed'])
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    before = sha256(model_path)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    if checkpoint['metadata']['size'] != config['size']:
        raise ValueError('模型与预处理尺寸配置不符')
    del checkpoint
    model_start = time.perf_counter()
    model = load_model(str(model_path), device)
    if device == 'cuda':
        torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - model_start
    output = folder / 'predictions' / (datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    report = {'model': str(model_path.resolve()), 'model_sha256': before,
              'model_bytes': model_path.stat().st_size, 'model_under_500MB': model_path.stat().st_size <= 500_000_000,
              'device': device, 'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None,
              'batch_size': config['batch_size'], 'size': config['size'], 'model_load_seconds': model_load_seconds,
              'official_T4_tested': False, 'splits': {}}
    log = output / 'prediction_log.jsonl'
    try:
        for name in (('valid', 'test') if split == 'both' else (split,)):
            started = time.perf_counter()
            data = config['data'][name]
            dataset = ChestXrayDataset(data['csv'], data['images'], labeled=name == 'valid', size=config['size'])
            expected = (1422,859) if name == 'valid' else (2218,1414)
            if (len(dataset), dataset.frame.Study_id.nunique()) != expected:
                raise ValueError('官方数据规模不符')
            loader = make_loader(dataset, config['batch_size'])
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
            if device == 'cuda':
                torch.cuda.synchronize()
            seconds = time.perf_counter() - inference_start
            image_predictions = pd.DataFrame(records)
            studies = aggregate_studies(dataset.frame, image_predictions)
            if name == 'valid':
                scored, metrics = study_metrics(dataset.frame, image_predictions)
                aligned = scored.merge(studies, on=['Subject_id','Study_id'], suffixes=('_a','_b'), validate='one_to_one')
                np.testing.assert_allclose(aligned[[p+'_a' for p in PROB_COLUMNS]], aligned[[p+'_b' for p in PROB_COLUMNS]], rtol=0, atol=1e-15)
                studies = scored
                metrics['validation_loss'] = total_loss/count
                reference = json.loads((folder / 'validation_metrics.json').read_text(encoding='utf-8'))
                metrics['reference_macro_auc'] = reference['macro_auc']
                metrics['macro_auc_difference'] = metrics['macro_auc'] - reference['macro_auc']
                previous = pd.read_csv(folder / 'val_study_predictions.csv', dtype={'Subject_id':str,'Study_id':str}, encoding='utf-8')
                aligned = studies.merge(previous, on=['Subject_id','Study_id'], suffixes=('_new','_old'), validate='one_to_one')
                if len(aligned) != expected[1]:
                    raise ValueError('历史验证 ID 对齐失败')
                difference = np.max(np.abs(aligned[[p+'_new' for p in PROB_COLUMNS]].to_numpy()-aligned[[p+'_old' for p in PROB_COLUMNS]].to_numpy()))
                metrics['max_probability_difference'] = float(difference)
                metrics['reference_matches'] = bool(abs(metrics['macro_auc_difference']) < 1e-10 and abs(metrics['macro_f1']-reference['macro_f1']) < 1e-10 and difference < 1e-6)
                dump(output / 'validation_metrics.json', metrics)
                if not metrics['reference_matches']:
                    raise RuntimeError('验证复测与历史成绩不一致，已记录差异，请排查')
            image_predictions.to_csv(output / f'{name}_image_predictions.csv', index=False, encoding='utf-8')
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
                    'split_total_seconds': time.perf_counter()-started, 'csv_reread_passed': True}
            report['splits'][name] = info
            if name == 'test':
                submission_path = output / 'submission.csv'
                make_submission(studies).to_csv(submission_path, index=False, encoding='utf-8')
                check = check_submission(submission_path, dataset.frame)
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
    args = parser.parse_args()
    run(args.model, args.split)


if __name__ == '__main__':
    main()
