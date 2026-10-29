"""从指定实验权重继续训练，保留来源实验及各轮模型。"""
import hashlib
import argparse
import json
from pathlib import Path
import shutil
import time

import pandas as pd
import torch

from .data import ChestXrayDataset, make_loader, normalization_for_settings
from .evaluate import evaluate
from .model import load_model
from .train import seed_everything, train_step, save_model, make_optimizer
from .devices import select_device, synchronize, reset_peak, peak_memory
from .fingerprints import dataset_fingerprint, EVALUATION_PROTOCOL
from .training_state import load_training_state, restore_training_state, save_training_state
from .early_stopping import EarlyStopping
from .lr_schedule import apply_epoch_lr, epoch_factor


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def better_auc(candidate: float | None, best: float) -> bool:
    return candidate is not None and candidate > best


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def run(baseline: Path = Path('outputs/baseline_v1'), output: Path = Path('outputs/baseline_v2'),
        additional_epochs: int = 3, learning_rate: float | None = None,
        device_name: str = 'auto', resume_state: Path | None = None,
        allow_nonofficial_counts: bool = False, early_stopping_patience: int = 0,
        early_stopping_min_delta: float = 0.0) -> None:
    stopper = EarlyStopping(early_stopping_patience, early_stopping_min_delta)
    if additional_epochs <= 0 or (learning_rate is not None and not 0 < learning_rate < 1):
        raise ValueError('轮数或学习率无效')
    if output.exists():
        raise FileExistsError(f'拒绝覆盖已有实验: {output}')
    hashes = {p.name: file_hash(p) for p in baseline.iterdir() if p.is_file()}
    settings = json.loads((baseline / 'train_config.json').read_text(encoding='utf-8'))
    initial = json.loads((baseline / 'validation_metrics.json').read_text(encoding='utf-8'))
    history = pd.read_csv(baseline / 'history.csv', encoding='utf-8').to_dict('records')
    restored_state = None
    model_path = baseline/'best_model.pth'
    if resume_state is not None:
        if learning_rate is not None:
            raise ValueError('完整恢复不能同时修改学习率；调参请使用仅权重新实验')
        restored_state,model_path = load_training_state(resume_state)
        settings = torch.load(model_path,map_location='cpu',weights_only=True)['metadata']
        if restored_state['epoch'] not in {row['epoch'] for row in history}:
            raise ValueError('来源实验历史缺少待恢复轮次，请指定该轮所属实验根目录')
        if initial['epoch'] > restored_state['epoch']:
            raise ValueError('来源最佳模型晚于恢复轮次，不能使用未来成绩选优')
    start_epoch = int(restored_state['epoch'] if restored_state else initial['epoch']) + 1
    end_epoch = start_epoch + additional_epochs - 1
    if settings.get('lr_schedule'):
        if restored_state is None:
            raise ValueError('带调度器的实验必须用完整状态恢复，不能只加载最佳权重重启调度')
        epoch_factor(settings['lr_schedule'], end_epoch)
    history = [row for row in history if row['epoch'] < start_epoch]
    if early_stopping_patience > 0 and [row['epoch'] for row in history] != list(range(1,start_epoch)):
        raise ValueError('启用早停要求从第1轮到恢复轮次的连续完整history')
    # 计数由来源完整历史重建，换目录或恢复训练不会无意清零；新参数显式定义本次策略。
    for row in history:
        stopper.step(float(row['macro_auc']),int(row['epoch']))
    if stopper.last_epoch != start_epoch - 1:
        raise ValueError('来源history缺少恢复轮次，无法可靠恢复早停计数')
    if learning_rate is not None:
        if 'backbone_lr' in settings or 'head_lr' in settings:
            raise ValueError('PT1分组学习率不能用单一--learning-rate静默覆盖')
        settings['learning_rate'] = learning_rate
    if shutil.disk_usage('.').free < 2 * 1024**3:
        raise RuntimeError('磁盘空间不足 2 GiB')
    device = select_device(device_name)
    settings['device'] = device
    settings['gpu'] = torch.cuda.get_device_name(0) if device == 'cuda' else None
    seed_everything(settings['seed'])
    torch.set_num_threads(4)
    config = settings['data']
    normalization = normalization_for_settings(settings)
    datasets = {s: ChestXrayDataset(config[s]['csv'], config[s]['images'], size=settings['size'],normalization=normalization,training=s=='train',
                augmentation=settings.get('augmentation','none') if s=='train' else 'none')
                for s in ('train', 'valid')}
    if not allow_nonofficial_counts and (len(datasets['train']), len(datasets['valid']), datasets['valid'].frame.Study_id.nunique()) != (30344, 1422, 859):
        raise ValueError('数据规模异常')
    for key in ('Subject_id', 'Study_id', 'Dicom_id'):
        if set(datasets['train'].frame[key]) & set(datasets['valid'].frame[key]):
            raise ValueError('训练验证集合重叠')
    train_fingerprint = dataset_fingerprint(datasets['train'].frame)
    validation_fingerprint = dataset_fingerprint(datasets['valid'].frame)
    if initial.get('validation_fingerprint') and (initial['validation_fingerprint'] != validation_fingerprint or initial.get('evaluation_protocol') != EVALUATION_PROTOCOL):
        raise ValueError('来源实验使用不同的验证标签/划分或协议，不能沿用其最佳成绩')
    refreshed_predictions = None
    if not initial.get('validation_fingerprint'):
        # 旧指标缺少指纹，重新评估旧最佳权重，而不是给历史结果补造身份。
        incumbent = load_model(str(baseline/'best_model.pth'),device)
        refreshed_predictions,refreshed_metrics = evaluate(incumbent,make_loader(datasets['valid'],settings['batch_size'],num_workers=settings.get('num_workers',0)),device)
        initial = {**initial,**refreshed_metrics}
        del incumbent
    model = load_model(str(model_path), device)
    # 此入口仅恢复模型权重，AdamW 动量重新初始化，明确记录为新实验。
    optimizer = make_optimizer(model,settings)
    # 新实验为4；旧检查点未记录时沿用历史0，不静默改变恢复条件。
    workers = settings.get('num_workers',0)
    from .study_training import make_study_loader, run_study_epoch, write_sampling_epoch
    study_mode=settings.get('sampler')=='study_batch'
    train_loader = make_study_loader(datasets['train'],settings) if study_mode else make_loader(datasets['train'], settings['batch_size'], shuffle=True, seed=settings['seed'],num_workers=workers)
    valid_loader = make_loader(datasets['valid'], settings['batch_size'],num_workers=workers,pin_memory=settings.get('pin_memory',False))
    if restored_state is not None:
        restore_training_state(restored_state,optimizer,train_loader,valid_loader,settings,
                               train_fingerprint,validation_fingerprint)
    settings.update(start_epoch=start_epoch, end_epoch=end_epoch, additional_epochs=additional_epochs, epochs=end_epoch,
                    resume_from=str(model_path),optimizer_state_restored=restored_state is not None,
                    optimizer_reset_reason=None if restored_state else 'Weights-only new experiment',
                    loader_rng_reinitialized=restored_state is None,
                    resume_state=str(resume_state) if resume_state else None,
                    train_fingerprint=train_fingerprint,validation_fingerprint=validation_fingerprint,
                    nonofficial_counts_allowed=allow_nonofficial_counts,
                    requested_end_epoch=end_epoch,
                    early_stopping={'patience':early_stopping_patience,'min_delta':early_stopping_min_delta,
                                    'monitor':'study_macro_auc','enabled':early_stopping_patience > 0})
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'train_config.json', settings)
    write_json(output / 'baseline1_manifest.json', {'name': 'Baseline-1', 'path': str(baseline), 'sha256': hashes})
    sampling_report=json.loads((baseline/'study_sampling_report.json').read_text(encoding='utf-8')) if study_mode else None
    if study_mode:
        sampling_report['epochs']=[r for r in sampling_report['epochs'] if r['epoch']<start_epoch]
        write_json(output/'study_sampling_report.json',sampling_report)
    best_auc, best_epoch = initial['macro_auc'], initial['epoch']
    # 初始候选包含 Baseline-1，续训成绩变差也不会丢失最佳模型。
    for name in ('best_model.pth', 'val_study_predictions.csv', 'validation_metrics.json'):
        shutil.copy2(baseline / name, output / name)
    if (baseline / 'valid_image_predictions.csv').is_file():
        shutil.copy2(baseline / 'valid_image_predictions.csv', output / 'valid_image_predictions.csv')
    if refreshed_predictions is not None:
        refreshed_predictions.to_csv(output/'val_study_predictions.csv',index=False,encoding='utf-8')
        write_json(output/'validation_metrics.json',initial)
    pd.DataFrame(history).to_csv(output / 'history.csv',index=False,encoding='utf-8')
    all_metrics = [initial]
    start = time.perf_counter()
    last_completed_epoch = start_epoch - 1
    try:
        for epoch in range(start_epoch, end_epoch + 1):
            if stopper.should_stop:
                break
            epoch_dir = output / f'epoch_{epoch}'
            epoch_dir.mkdir(exist_ok=False)
            reset_peak(device)
            train_start = time.perf_counter()
            epoch_lrs = apply_epoch_lr(optimizer, settings, epoch)
            total, seen = 0.0, 0
            study_result=None
            if study_mode:
                study_result=run_study_epoch(model,optimizer,train_loader,device)
                seen=study_result['training_images'];total=study_result['train_loss']*seen
                sampling_report=write_sampling_epoch(output,datasets['train'],train_loader,epoch,study_result,sampling_report)
                write_json(epoch_dir/'study_sampling_report.json',sampling_report)
            else:
                for step, batch in enumerate(train_loader, 1):
                    loss, _ = train_step(model, batch['image'].to(device), batch['label'].to(device), optimizer)
                    seen += len(batch['image'])
                    total += loss * len(batch['image'])
                    if step == 1 or step % 50 == 0 or step == len(train_loader):
                        print(f'epoch={epoch} step={step}/{len(train_loader)} images={seen} loss={total/seen:.6f} seconds={time.perf_counter()-train_start:.1f}', flush=True)
            synchronize(device)
            train_seconds = time.perf_counter() - train_start
            if seen != len(datasets['train']):
                raise RuntimeError('训练图片数量异常')
            val_start = time.perf_counter()
            predictions, metrics = evaluate(model, valid_loader, device, epoch_dir / 'valid_image_predictions.csv')
            metrics.update(epoch=epoch, train_loss=total/seen, training_images=seen,
                           train_seconds=train_seconds, validation_seconds=time.perf_counter()-val_start,
                           **peak_memory(device))
            if study_result:
                metrics.update({k:v for k,v in study_result.items() if k!='epoch_seconds_including_data'})
            if metrics['study_count'] != datasets['valid'].frame.Study_id.nunique() or metrics['macro_auc'] is None:
                raise RuntimeError('验证数量异常或 Macro-AUC 未定义')
            stopper.step(float(metrics['macro_auc']),epoch)
            metrics['early_stopping'] = stopper.as_dict()
            if settings.get('lr_schedule'):
                metrics['learning_rates'] = epoch_lrs
            save_model(model, epoch_dir / 'model.pth', {**settings, 'completed_epoch': epoch})
            metrics['checkpoint_sha256'] = file_hash(epoch_dir / 'model.pth')
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
            save_training_state(epoch_dir/'training_state.pth',epoch_dir/'model.pth',optimizer,
                                train_loader,valid_loader,epoch,settings,train_fingerprint,validation_fingerprint)
            predictions.to_csv(epoch_dir / 'val_study_predictions.csv', index=False, encoding='utf-8')
            write_json(epoch_dir / 'validation_metrics.json', metrics)
            write_json(epoch_dir / 'early_stopping.json',stopper.as_dict())
            history_row = {key: metrics[key] for key in ('epoch','train_loss','validation_loss','macro_auc','macro_f1','train_seconds','validation_seconds')}
            if study_result:
                history_row.update({k:metrics[k] for k in ('training_studies','training_images','optimizer_steps','max_batch_images')})
            if settings.get('lr_schedule'):
                history_row.update({name+'_lr':lr for name,lr in epoch_lrs.items()})
            history.append(history_row)
            pd.DataFrame(history).to_csv(output / 'history.csv', index=False, encoding='utf-8')
            all_metrics.append(metrics)
            if better_auc(metrics['macro_auc'], best_auc):
                best_auc, best_epoch = metrics['macro_auc'], epoch
                shutil.copy2(epoch_dir / 'model.pth', output / 'best_model.pth')
                for name in ('val_study_predictions.csv', 'validation_metrics.json', 'valid_image_predictions.csv'):
                    shutil.copy2(epoch_dir / name, output / name)
            write_json(output / 'comparison.json', {'best_epoch': best_epoch, 'best_macro_auc': best_auc,
                       'elapsed_seconds': time.perf_counter()-start, 'epochs': all_metrics,
                       'early_stopping':stopper.as_dict(),'last_completed_epoch':epoch,
                       'requested_end_epoch':end_epoch})
            last_completed_epoch = epoch
            print(json.dumps({'finished_epoch':epoch, **metrics, 'best_epoch':best_epoch}, ensure_ascii=False), flush=True)
        stop_reason = 'early_stopping' if stopper.should_stop else 'max_epochs_reached'
        # 分开记录实际轮数与请求上限；停止前已保存最后一轮完整状态，最佳模型始终保留。
        settings.update(epochs=last_completed_epoch,end_epoch=last_completed_epoch,
                        completed_epoch=last_completed_epoch,stop_reason=stop_reason,
                        completed_additional_epochs=last_completed_epoch-start_epoch+1)
        write_json(output/'train_config.json',settings)
        write_json(output/'early_stopping.json',stopper.as_dict())
        write_json(output/'comparison.json',{'best_epoch':best_epoch,'best_macro_auc':best_auc,
            'elapsed_seconds':time.perf_counter()-start,'epochs':all_metrics,
            'early_stopping':stopper.as_dict(),'last_completed_epoch':last_completed_epoch,
            'requested_end_epoch':end_epoch,'stop_reason':stop_reason})
        print(f'STOP reason={stop_reason} completed_epoch={last_completed_epoch} '
              f'max_epoch={end_epoch} bad_epochs={stopper.bad_epochs}',flush=True)
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
    parser.add_argument('--resume-state',type=Path,help='恢复新格式完整epoch边界状态；不修改原实验')
    parser.add_argument('--allow-nonofficial-counts',action='store_true',help='仅限小样本开发测试')
    parser.add_argument('--early-stopping-patience',type=int,default=0,help='连续多少轮无显著AUC改善后停止；0关闭')
    parser.add_argument('--early-stopping-min-delta',type=float,default=0.0,help='显著AUC改善的绝对增量，默认0')
    args = parser.parse_args()
    run(args.source, args.output, args.additional_epochs, args.learning_rate, args.device,
        args.resume_state,args.allow_nonofficial_counts,args.early_stopping_patience,args.early_stopping_min_delta)
