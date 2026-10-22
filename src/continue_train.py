"""从指定实验权重继续训练，保留来源实验及各轮模型。"""
import hashlib
import argparse
import json
from pathlib import Path
import shutil
import time

import pandas as pd
import torch

from .data import ChestXrayDataset, make_loader
from .evaluate import evaluate
from .model import load_model
from .train import seed_everything, train_step, save_model
from .devices import select_device, synchronize, reset_peak, peak_memory


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def better_auc(candidate: float | None, best: float) -> bool:
    return candidate is not None and candidate > best


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def run(baseline: Path = Path('outputs/baseline_v1'), output: Path = Path('outputs/baseline_v2'),
        additional_epochs: int = 3, learning_rate: float | None = None,
        device_name: str = 'auto') -> None:
    if additional_epochs <= 0 or (learning_rate is not None and not 0 < learning_rate < 1):
        raise ValueError('轮数或学习率无效')
    if output.exists():
        raise FileExistsError(f'拒绝覆盖已有实验: {output}')
    hashes = {p.name: file_hash(p) for p in baseline.iterdir() if p.is_file()}
    settings = json.loads((baseline / 'train_config.json').read_text(encoding='utf-8'))
    initial = json.loads((baseline / 'validation_metrics.json').read_text(encoding='utf-8'))
    history = pd.read_csv(baseline / 'history.csv', encoding='utf-8').to_dict('records')
    start_epoch = int(initial['epoch']) + 1
    end_epoch = start_epoch + additional_epochs - 1
    history = [row for row in history if row['epoch'] < start_epoch]
    if learning_rate is not None:
        settings['learning_rate'] = learning_rate
    if shutil.disk_usage('.').free < 2 * 1024**3:
        raise RuntimeError('磁盘空间不足 2 GiB')
    device = select_device(device_name)
    settings['device'] = device
    settings['gpu'] = torch.cuda.get_device_name(0) if device == 'cuda' else None
    seed_everything(settings['seed'])
    torch.set_num_threads(4)
    config = settings['data']
    datasets = {s: ChestXrayDataset(config[s]['csv'], config[s]['images'], size=settings['size'])
                for s in ('train', 'valid')}
    if (len(datasets['train']), len(datasets['valid']), datasets['valid'].frame.Study_id.nunique()) != (30344, 1422, 859):
        raise ValueError('数据规模异常')
    for key in ('Subject_id', 'Study_id', 'Dicom_id'):
        if set(datasets['train'].frame[key]) & set(datasets['valid'].frame[key]):
            raise ValueError('训练验证集合重叠')
    model = load_model(str(baseline / 'best_model.pth'), device)
    # 此入口仅恢复模型权重，AdamW 动量重新初始化，明确记录为新实验。
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'],
                                 weight_decay=settings['weight_decay'])
    train_loader = make_loader(datasets['train'], settings['batch_size'], shuffle=True, seed=settings['seed'])
    valid_loader = make_loader(datasets['valid'], settings['batch_size'])
    settings.update(start_epoch=start_epoch, end_epoch=end_epoch, additional_epochs=additional_epochs, epochs=end_epoch,
                    resume_from=str(baseline / 'best_model.pth'), optimizer_state_restored=False,
                    optimizer_reset_reason='This entry resumes model weights only; optimizer is reinitialized',
                    loader_rng_reinitialized=True)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'train_config.json', settings)
    write_json(output / 'baseline1_manifest.json', {'name': 'Baseline-1', 'path': str(baseline), 'sha256': hashes})
    best_auc, best_epoch = initial['macro_auc'], initial['epoch']
    # 初始候选包含 Baseline-1，续训成绩变差也不会丢失最佳模型。
    for name in ('best_model.pth', 'val_study_predictions.csv', 'validation_metrics.json'):
        shutil.copy2(baseline / name, output / name)
    all_metrics = [initial]
    start = time.perf_counter()
    try:
        for epoch in range(start_epoch, end_epoch + 1):
            epoch_dir = output / f'epoch_{epoch}'
            epoch_dir.mkdir(exist_ok=False)
            reset_peak(device)
            train_start = time.perf_counter()
            total, seen = 0.0, 0
            for step, batch in enumerate(train_loader, 1):
                loss, _ = train_step(model, batch['image'].to(device), batch['label'].to(device), optimizer)
                seen += len(batch['image'])
                total += loss * len(batch['image'])
                if step == 1 or step % 50 == 0 or step == len(train_loader):
                    print(f'epoch={epoch} step={step}/{len(train_loader)} images={seen} loss={total/seen:.6f} seconds={time.perf_counter()-train_start:.1f}', flush=True)
            synchronize(device)
            train_seconds = time.perf_counter() - train_start
            if seen != 30344:
                raise RuntimeError('训练图片数量异常')
            val_start = time.perf_counter()
            predictions, metrics = evaluate(model, valid_loader, device)
            metrics.update(epoch=epoch, train_loss=total/seen, training_images=seen,
                           train_seconds=train_seconds, validation_seconds=time.perf_counter()-val_start,
                           **peak_memory(device))
            if metrics['study_count'] != 859 or metrics['macro_auc'] is None:
                raise RuntimeError('验证数量异常或 Macro-AUC 未定义')
            save_model(model, epoch_dir / 'model.pth', {**settings, 'completed_epoch': epoch})
            write_json(epoch_dir / 'train_config.json', {**settings, 'completed_epoch':epoch})
            restored = load_model(str(epoch_dir / 'model.pth'), device)
            for key, value in model.state_dict().items():
                if not torch.equal(value, restored.state_dict()[key]):
                    raise RuntimeError('模型重载参数不一致')
            with torch.no_grad():
                images = next(iter(valid_loader))['image'].to(device)
                torch.testing.assert_close(model(images), restored(images), rtol=0, atol=0)
            del restored
            metrics['reload_equal'] = True
            # 后续继续训练可恢复 AdamW 状态与随机状态，避免再次丢失动量。
            torch.save({'epoch': epoch, 'optimizer': optimizer.state_dict(),
                        'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all() if device == 'cuda' else [],
                        'loader_rng': train_loader.generator.get_state()},
                       epoch_dir / 'training_state.pth')
            predictions.to_csv(epoch_dir / 'val_study_predictions.csv', index=False, encoding='utf-8')
            write_json(epoch_dir / 'validation_metrics.json', metrics)
            history.append({key: metrics[key] for key in ('epoch','train_loss','validation_loss','macro_auc','macro_f1','train_seconds','validation_seconds')})
            pd.DataFrame(history).to_csv(output / 'history.csv', index=False, encoding='utf-8')
            all_metrics.append(metrics)
            if better_auc(metrics['macro_auc'], best_auc):
                best_auc, best_epoch = metrics['macro_auc'], epoch
                shutil.copy2(epoch_dir / 'model.pth', output / 'best_model.pth')
                for name in ('val_study_predictions.csv', 'validation_metrics.json'):
                    shutil.copy2(epoch_dir / name, output / name)
            write_json(output / 'comparison.json', {'best_epoch': best_epoch, 'best_macro_auc': best_auc,
                       'elapsed_seconds': time.perf_counter()-start, 'epochs': all_metrics})
            print(json.dumps({'finished_epoch':epoch, **metrics, 'best_epoch':best_epoch}, ensure_ascii=False), flush=True)
        if any(file_hash(baseline / name) != digest for name, digest in hashes.items()):
            raise RuntimeError('Baseline-1 文件发生变化')
        print(f'COMPLETE best_epoch={best_epoch} best_auc={best_auc:.9f} baseline1_hashes_unchanged=True', flush=True)
    except Exception as exc:
        write_json(output / 'failure.json', {'error_type': type(exc).__name__, 'message': str(exc),
                                            'elapsed_seconds': time.perf_counter()-start})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path('outputs/baseline_v1'))
    parser.add_argument('--output', type=Path, default=Path('outputs/baseline_v2'))
    parser.add_argument('--additional-epochs', type=int, default=3)
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--device', choices=['auto','cuda','mps','cpu'], default='auto')
    args = parser.parse_args()
    run(args.source, args.output, args.additional_epochs, args.learning_rate, args.device)
