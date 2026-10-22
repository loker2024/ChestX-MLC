"""完整训练集的一轮训练和完整验证；固定只运行一个 epoch。"""
import gc
import json
from pathlib import Path
import shutil
import sys
import time

import pandas as pd
import torch
import torchvision

from .data import ChestXrayDataset, make_loader
from .evaluate import evaluate
from .model import build_model, load_model
from .train import seed_everything, train_step, save_model


def run() -> None:
    output = Path('outputs/baseline_v1')
    if output.exists():
        raise FileExistsError('输出目录已存在，拒绝覆盖 outputs/baseline_v1')
    free_disk = shutil.disk_usage('.').free
    if free_disk < 2 * 1024**3:
        raise RuntimeError('磁盘剩余空间不足 2 GiB')
    if not torch.cuda.is_available():
        raise RuntimeError('本轮完整训练要求可用 CUDA，请检查设备')
    device = 'cuda'
    torch.set_num_threads(4)
    seed_everything(42)
    config = json.loads(Path('configs/data.json').read_text(encoding='utf-8'))
    datasets = {split: ChestXrayDataset(config[split]['csv'], config[split]['images'], size=config['size']) for split in ('train', 'valid')}
    for key in ('Subject_id', 'Study_id', 'Dicom_id'):
        if set(datasets['train'].frame[key]) & set(datasets['valid'].frame[key]):
            raise ValueError(f'训练验证数据泄漏: {key}')
    if (len(datasets['train']), len(datasets['valid']), datasets['valid'].frame['Study_id'].nunique()) != (30344, 1422, 859):
        raise ValueError('官方数据规模不符，需核查')
    free_vram, total_vram = torch.cuda.mem_get_info()
    print(f'Preflight GPU={torch.cuda.get_device_name(0)} free={free_vram/1024**3:.2f}GiB disk={free_disk/1024**3:.2f}GiB', flush=True)
    # 探测模型独立于正式训练模型；探测后重新设种子并从头初始化。
    for batch_size in (16, 8):
        probe = build_model().to(device)
        optimizer = torch.optim.AdamW(probe.parameters(), lr=0.001, weight_decay=0.01)
        batch = next(iter(make_loader(datasets['train'], batch_size, shuffle=True)))
        try:
            train_step(probe, batch['image'].to(device), batch['label'].to(device), optimizer)
            break
        except torch.cuda.OutOfMemoryError:
            if batch_size == 8:
                raise
        finally:
            del probe, optimizer
            gc.collect()
            torch.cuda.empty_cache()
    print(f'Batch size {batch_size} probe passed', flush=True)
    seed_everything(42)
    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    train_loader = make_loader(datasets['train'], batch_size, shuffle=True, seed=42)
    valid_loader = make_loader(datasets['valid'], batch_size)
    settings = {'epochs': 1, 'seed': 42, 'batch_size': batch_size, 'size': config['size'],
                'learning_rate': 0.001, 'weight_decay': 0.01, 'optimizer': 'AdamW',
                'weights': None, 'device': device, 'gpu': torch.cuda.get_device_name(0),
                'gpu_total_bytes': total_vram, 'gpu_free_before_bytes': free_vram,
                'disk_free_before_bytes': free_disk, 'data': config,
                'python': sys.version, 'torch': str(torch.__version__), 'torchvision': str(torchvision.__version__),
                'threshold': 0.5, 'aggregation': 'mean image probabilities per Study_id',
                'preprocess': 'RGB letterbox /255', 'num_workers': 0}
    output.mkdir(parents=True, exist_ok=False)
    (output / 'train_config.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    total_loss, seen = 0.0, 0
    for step, batch in enumerate(train_loader, 1):
        loss, _ = train_step(model, batch['image'].to(device), batch['label'].to(device), optimizer)
        count = len(batch['image'])
        total_loss += loss * count
        seen += count
        if step == 1 or step % 25 == 0 or step == len(train_loader):
            print(f'train {step}/{len(train_loader)} images={seen} loss={total_loss/seen:.6f} elapsed={time.perf_counter()-start:.1f}s', flush=True)
    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - start
    if seen != len(datasets['train']):
        raise RuntimeError('未完成全部训练图片')
    val_start = time.perf_counter()
    predictions, metrics = evaluate(model, valid_loader, device)
    validation_seconds = time.perf_counter() - val_start
    if len(predictions) != 859:
        raise RuntimeError('验证检查数不正确')
    metrics.update(epoch=1, train_loss=total_loss/seen, training_images=seen,
                   train_seconds=train_seconds, validation_seconds=validation_seconds,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved())
    save_model(model, output / 'best_model.pth', settings)
    restored = load_model(str(output / 'best_model.pth'), device)
    for key, value in model.state_dict().items():
        if not torch.equal(value, restored.state_dict()[key]):
            raise RuntimeError('重载参数不一致')
    with torch.no_grad():
        images = next(iter(valid_loader))['image'].to(device)
        torch.testing.assert_close(model(images), restored(images), rtol=0, atol=0)
    metrics['reload_equal'] = True
    predictions.to_csv(output / 'val_study_predictions.csv', index=False, encoding='utf-8')
    (output / 'validation_metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding='utf-8')
    pd.DataFrame([{key: metrics[key] for key in ('epoch','train_loss','validation_loss','macro_auc','macro_f1','train_seconds','validation_seconds')}]).to_csv(output / 'history.csv', index=False, encoding='utf-8')
    print(json.dumps(metrics, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    run()
