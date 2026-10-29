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
from .train import seed_everything, train_step, save_model, make_optimizer
from .devices import select_device, synchronize, reset_peak, peak_memory
from .fingerprints import dataset_fingerprint
from .training_state import save_training_state
from .lr_schedule import apply_epoch_lr
from .pretrained import file_sha256


def run(device_name: str = 'auto', config_path: Path = Path('configs/data.json'),
        output: Path = Path('outputs/baseline_v1'), seed: int = 42,
        allow_nonofficial_counts: bool = False, experiment: dict | None = None,
        pt1_config: dict | None = None) -> None:
    # 旧入口的自动批量/固定学习率保持兼容；新入口严格执行已校验的实验配置。
    if pt1_config is not None:
        if experiment is not None:
            raise ValueError('不能混用P0与PT1配置')
        from .pt1 import validate_config, data_settings, approved_model
        from .pretrained import require_approved_weight
        pt1_config = validate_config(pt1_config)
        provenance = require_approved_weight(Path(pt1_config['pretrained_weight_path']),
            pt1_config['pretrained_weight_sha256'],Path(pt1_config['pretrained_registry']),
            **({'backbone':pt1_config['backbone'],**({'variant':pt1_config['pretrained_variant']} if 'pretrained_variant' in pt1_config else {})} if pt1_config['backbone'] != 'resnet18' else {}))
        config,seed = data_settings(pt1_config),pt1_config['seed']
    elif experiment is not None:
        from .experiment_config import validate_config, load_data_config
        experiment = validate_config(experiment)
        config = load_data_config(experiment)
        seed = experiment['random_seed']
    else:
        config = json.loads(config_path.read_text(encoding='utf-8'))
    learning_rate = experiment['learning_rate'] if experiment else 0.001
    weight_decay = experiment['weight_decay'] if experiment else 0.01
    if pt1_config is not None:
        learning_rate,weight_decay = pt1_config['backbone_lr'],pt1_config['weight_decay']
    normalization = 'imagenet' if pt1_config is not None else 'none'
    study_mode=pt1_config is not None and pt1_config.get('sampler')=='study_batch'
    from .study_training import make_study_loader, run_study_epoch, write_sampling_epoch
    # 合成测试仍可显式使用0；正式训练默认4，设置写入检查点供续训沿用。
    num_workers = config.get('num_workers', 0 if allow_nonofficial_counts else 4)
    optimizer_settings = {'learning_rate':learning_rate,'weight_decay':weight_decay}
    if pt1_config is not None:
        optimizer_settings.update(backbone_lr=pt1_config['backbone_lr'],head_lr=pt1_config['head_lr'])
    if output.exists():
        raise FileExistsError(f'输出目录已存在，拒绝覆盖 {output}')
    free_disk = shutil.disk_usage('.').free
    if free_disk < 2 * 1024**3:
        raise RuntimeError('磁盘剩余空间不足 2 GiB')
    device = select_device(device_name)
    torch.set_num_threads(4)
    seed_everything(seed)
    datasets = {split: ChestXrayDataset(config[split]['csv'], config[split]['images'], size=config['size'],
                normalization=normalization, training=split=='train',
                augmentation=pt1_config.get('augmentation','none') if pt1_config and split=='train' else 'none') for split in ('train', 'valid')}
    for key in ('Subject_id', 'Study_id', 'Dicom_id'):
        if set(datasets['train'].frame[key]) & set(datasets['valid'].frame[key]):
            raise ValueError(f'训练验证数据泄漏: {key}')
    if not allow_nonofficial_counts and (len(datasets['train']), len(datasets['valid']), datasets['valid'].frame['Study_id'].nunique()) != (30344, 1422, 859):
        raise ValueError('官方数据规模不符，需核查')
    free_vram, total_vram = torch.cuda.mem_get_info() if device == 'cuda' else (None,None)
    print(f'Preflight device={device} disk={free_disk/1024**3:.2f}GiB', flush=True)
    # 探测模型独立于正式训练模型；探测后重新设种子并从头初始化。
    candidates = (pt1_config['batch_size'],) if pt1_config else (experiment['batch_size'],) if experiment else ((16,8) if device == 'cuda' else (2,))
    for batch_size in candidates:
        probe = (approved_model(pt1_config) if pt1_config else build_model()).to(device)
        optimizer = make_optimizer(probe,optimizer_settings)
        probe_loader=make_study_loader(datasets['train'],pt1_config) if study_mode else make_loader(datasets['train'],batch_size,shuffle=True,num_workers=num_workers)
        batch = next(iter(probe_loader))
        try:
            if study_mode:
                train_step(probe,batch['image'].to(device),batch['label'].to(device),optimizer,study_index=batch['study_index'])
            else:
                train_step(probe, batch['image'].to(device), batch['label'].to(device), optimizer)
            break
        except torch.cuda.OutOfMemoryError:
            if batch_size == candidates[-1]:
                raise
        finally:
            del probe, optimizer, probe_loader, batch
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()
    print(f'Batch size {batch_size} probe passed', flush=True)
    seed_everything(seed)
    model = (approved_model(pt1_config) if pt1_config else build_model()).to(device)
    optimizer = make_optimizer(model,optimizer_settings)
    train_loader = make_study_loader(datasets['train'],pt1_config) if study_mode else make_loader(datasets['train'], batch_size, shuffle=True, seed=seed, num_workers=num_workers)
    valid_loader = make_loader(datasets['valid'], batch_size, num_workers=num_workers,pin_memory=config.get('pin_memory',False))
    settings = {'epochs': 1, 'seed': seed, 'batch_size': batch_size, 'size': config['size'],
                'learning_rate': learning_rate, 'weight_decay': weight_decay, 'optimizer': 'AdamW',
                'weights': None, 'device': device, 'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None,
                'gpu_total_bytes': total_vram, 'gpu_free_before_bytes': free_vram,
                'disk_free_before_bytes': free_disk, 'data': config,
                'python': sys.version, 'torch': str(torch.__version__), 'torchvision': str(torchvision.__version__),
                'threshold': 0.5, 'aggregation': 'mean image probabilities per Study_id',
                'preprocess': 'RGB letterbox /255', 'num_workers': num_workers,
                'nonofficial_counts_allowed':allow_nonofficial_counts}
    if experiment is not None:
        settings.update(experiment_id=experiment['experiment_id'], backbone=experiment['backbone'],
                        experiment_config=experiment)
    if pt1_config is not None:
        from .data import IMAGENET_PREPROCESS
        from .pretrained import WEIGHT_SPECS
        settings.update(experiment_id=pt1_config['experiment_id'],backbone=pt1_config['backbone'],pretrained=True,
            weights=WEIGHT_SPECS[pt1_config.get('pretrained_variant',pt1_config['backbone'])][0],
            pretrained_weight_sha256=pt1_config['pretrained_weight_sha256'],
            pretrained_provenance=provenance,experiment_config=pt1_config,
            pretrained_backbone_verification=model.pretrained_backbone_verification,
            backbone_lr=pt1_config['backbone_lr'],head_lr=pt1_config['head_lr'],
            preprocess=IMAGENET_PREPROCESS,normalization='imagenet',
            normalization_mean=pt1_config['normalization_mean'],normalization_std=pt1_config['normalization_std'],
            validation_mode=pt1_config['validation_mode'],augmentation=pt1_config['augmentation'])
        if pt1_config.get('model_backend')=='timm':
            settings.update(model_backend='timm',pretrained_model=pt1_config['pretrained_model'],
                pretrained_variant=pt1_config['pretrained_variant'],pretrained_recipe='RA',
                model_backend_version=model.cxr_backend_version)
        if isinstance(pt1_config['scheduler'], dict):
            settings['lr_schedule'] = pt1_config['scheduler']
        if study_mode:
            settings.update({key:pt1_config[key] for key in ('sampler','loss_reduction','studies_per_batch','max_images_per_batch','pin_memory')})
            settings['scheduler_unit']='epoch'
    epoch_lrs = apply_epoch_lr(optimizer, settings, 1)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'train_config.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
    reset_peak(device)
    start = time.perf_counter()
    total_loss, seen = 0.0, 0
    study_result=None
    if study_mode:
        study_result=run_study_epoch(model,optimizer,train_loader,device)
        seen=study_result['training_images'];total_loss=study_result['train_loss']*seen
        write_sampling_epoch(output,datasets['train'],train_loader,1,study_result)
    else:
        for step, batch in enumerate(train_loader, 1):
            loss, _ = train_step(model, batch['image'].to(device), batch['label'].to(device), optimizer)
            count = len(batch['image'])
            total_loss += loss * count
            seen += count
            if step == 1 or step % 25 == 0 or step == len(train_loader):
                print(f'train {step}/{len(train_loader)} images={seen} loss={total_loss/seen:.6f} elapsed={time.perf_counter()-start:.1f}s', flush=True)
    synchronize(device)
    train_seconds = time.perf_counter() - start
    if seen != len(datasets['train']):
        raise RuntimeError('未完成全部训练图片')
    val_start = time.perf_counter()
    predictions, metrics = evaluate(model, valid_loader, device, output / 'valid_image_predictions.csv')
    validation_seconds = time.perf_counter() - val_start
    if len(predictions) != datasets['valid'].frame.Study_id.nunique():
        raise RuntimeError('验证检查数不正确')
    metrics.update(epoch=1, train_loss=total_loss/seen, training_images=seen,
                   train_seconds=train_seconds, validation_seconds=validation_seconds,
                   **peak_memory(device))
    if study_result:
        metrics.update({k:v for k,v in study_result.items() if k!='epoch_seconds_including_data'})
    if settings.get('lr_schedule'):
        metrics['learning_rates'] = epoch_lrs
    save_model(model, output / 'best_model.pth', {**settings, 'completed_epoch':1})
    metrics['checkpoint_sha256'] = file_sha256(output / 'best_model.pth')
    restored = load_model(str(output / 'best_model.pth'), device)
    for key, value in model.state_dict().items():
        if not torch.equal(value, restored.state_dict()[key]):
            raise RuntimeError('重载参数不一致')
    with torch.no_grad():
        images = next(iter(valid_loader))['image'].to(device)
        torch.testing.assert_close(model(images), restored(images), rtol=0, atol=0)
    metrics['reload_equal'] = True
    save_training_state(output/'training_state.pth',output/'best_model.pth',optimizer,
                        train_loader,valid_loader,1,settings,
                        dataset_fingerprint(datasets['train'].frame),dataset_fingerprint(datasets['valid'].frame))
    predictions.to_csv(output / 'val_study_predictions.csv', index=False, encoding='utf-8')
    (output / 'validation_metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding='utf-8')
    history_row = {key: metrics[key] for key in ('epoch','train_loss','validation_loss','macro_auc','macro_f1','train_seconds','validation_seconds')}
    if study_result:
        history_row.update({k:metrics[k] for k in ('training_studies','training_images','optimizer_steps','max_batch_images')})
    if settings.get('lr_schedule'):
        history_row.update({name+'_lr':lr for name,lr in epoch_lrs.items()})
    pd.DataFrame([history_row]).to_csv(output / 'history.csv', index=False, encoding='utf-8')
    print(json.dumps(metrics, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    run()
