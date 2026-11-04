"""内部患者留出A/B入口；默认只显示计划，--check只预检，--execute才在服务器训练。"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import time

import pandas as pd
import torch

from .analysis_common import dump, finish_manifest, provenance, save_csv, sha256
from .data import ChestXrayDataset, IMAGENET_PREPROCESS, IMAGENET_MEAN, IMAGENET_STD, make_loader
from .devices import select_device, synchronize, reset_peak, peak_memory
from .early_stopping import EarlyStopping
from .evaluate import evaluate
from .internal_split import PATH_KEYS, read_paths, verify_split, assert_sources_unchanged
from .lr_schedule import apply_epoch_lr, validate_schedule
from .model import build_model, load_model
from .pretrained import require_approved_weight
from .study_training import make_study_loader, run_study_epoch, write_sampling_epoch
from .train import seed_everything, make_optimizer, save_model

CONFIGS = {a: Path('configs/experiments/Internal_' + a + '_DenseNet121_320_Study.json') for a in ('A', 'B')}
WEIGHT_SHA = 'a639ec97d7c33b07ae66f0b5fb7d0192f95a3b11b7576c66c0126c2a727c4395'


def validate_config(config: dict) -> dict:
    if config.get('experiment_id') in ('R1_DenseNet121_384_BCE_v2','R2_DenseNet121_384_ASL_v2'):
        from .resolution_experiments import validate_resolution_config
        return validate_resolution_config(config)
    if config.get('experiment_id')=='C1_ASL_DenseNet121_320_v2':
        from .asl import ASL_PARAMETERS
        if config.get('loss')!='ASL' or config.get('asl')!=ASL_PARAMETERS or config.get('loss_reduction')!='study_balanced_image_asl':
            raise ValueError('C1只允许预定ASL参数和原Study均衡归约')
        proxy=copy.deepcopy(config);proxy.pop('asl')
        proxy.update(experiment_id='Internal_A_DenseNet121_320_Study',loss='BCEWithLogitsLoss',loss_reduction='study_balanced_image_bce')
        validate_config(proxy)
        return copy.deepcopy(config)
    fixed = {'backbone': 'densenet121', 'model_backend': 'torchvision', 'pretrained': True,
        'pretrained_weight_sha256': WEIGHT_SHA, 'input_size': 320, 'num_classes': 10,
        'optimizer': 'AdamW', 'loss': 'BCEWithLogitsLoss', 'head_lr': .001, 'weight_decay': .01,
        'study_aggregation': 'mean', 'validation_mode': 'internal_subject_holdout', 'seed': 42,
        'batch_size': 16, 'normalization': 'imagenet', 'augmentation': 'none',
        'sampler': 'study_batch', 'loss_reduction': 'study_balanced_image_bce',
        'studies_per_batch': 8, 'max_images_per_batch': 16, 'pin_memory': True,
        'early_stopping': {'patience': 3, 'min_delta': .0001}}
    if set(config) != set(fixed) | {'experiment_id', 'backbone_lr', 'epochs', 'scheduler', 'num_workers', 'pretrained_registry'}:
        raise ValueError('内部A/B配置字段缺失或未知')
    if any(config[k] != v or type(config[k]) is not type(v) for k, v in fixed.items()):
        raise ValueError('必须保持原P5模型/采样/Loss/预处理及固定对照设置')
    arm = {'Internal_A_DenseNet121_320_Study': .0001, 'Internal_B_DenseNet121_320_Study': .00003}
    if config['experiment_id'] not in arm or config['backbone_lr'] != arm[config['experiment_id']]:
        raise ValueError('A必须使用1e-4，B必须使用3e-5骨干LR')
    if type(config['epochs']) is not int or not 3 <= config['epochs'] <= 12:
        raise ValueError('预定预算必须为3至12轮')
    if type(config['num_workers']) is not int or config['num_workers'] < 0:
        raise ValueError('num_workers须为非负整数')
    if not isinstance(config['pretrained_registry'], str) or not config['pretrained_registry'].strip():
        raise ValueError('必须指定原权重审查登记表')
    validate_schedule(config['scheduler'], config['epochs'])
    if config['scheduler'] != {'name': 'warmup_cosine', 'warmup_epochs': 2,
            'total_epochs': config['epochs'], 'start_factor': .1, 'min_factor': .01}:
        raise ValueError('保持P5的两轮Warmup+Cosine配方')
    return copy.deepcopy(config)


def load_pair(num_workers: int | None = None) -> dict:
    configs = {a: validate_config(json.loads(p.read_text(encoding='utf-8-sig'))) for a, p in CONFIGS.items()}
    if num_workers is not None:
        for a in configs:
            configs[a]['num_workers'] = num_workers
            validate_config(configs[a])
    differences = [k for k in configs['A'] if configs['A'][k] != configs['B'][k]]
    if set(differences) != {'experiment_id', 'backbone_lr'}:
        raise ValueError('A/B除编号和骨干LR外不能有任何配置差异')
    return configs


def prepare(config: dict, paths: dict, *, require_official: bool = True) -> tuple[dict, dict, dict]:
    config = validate_config(config)
    _, audit = verify_split(paths, require_official=require_official)
    approval = require_approved_weight(paths['pretrained_weight_path'], config['pretrained_weight_sha256'],
                                      Path(config['pretrained_registry']), backbone='densenet121')
    datasets = {s: ChestXrayDataset(paths['split_directory'] / ('internal_' + s + '.csv'),
        paths['image_root'], size=config['input_size'], normalization='imagenet', training=s == 'train') for s in ('train', 'valid')}
    audit['source_hashes'][str(paths['pretrained_weight_path'])] = sha256(paths['pretrained_weight_path'])
    audit['source_hashes'][str(Path(config['pretrained_registry']).resolve())] = sha256(config['pretrained_registry'])
    assert_sources_unchanged(audit['source_hashes'])
    return datasets, audit, approval


def state_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode()); digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode()); digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def execute(config: dict, paths: dict, output: Path, device_name: str = 'cuda', *,
            require_official: bool = True, reference_run: Path | None = None) -> Path:
    """测试只通过Python mock使用模拟数据；CLI不提供绕过官方来源或许可的开关。"""
    config = validate_config(config)
    if output.exists():
        raise FileExistsError('拒绝覆盖已有实验：' + str(output))
    datasets, audit, approval = prepare(config, paths, require_official=require_official)
    label_loss=None; reference=None
    resolution_arm=config['experiment_id'] in ('R1_DenseNet121_384_BCE_v2','R2_DenseNet121_384_ASL_v2')
    if resolution_arm:
        from .resolution_experiments import check_reference
        if reference_run is None:raise ValueError('384实验必须绑定对应原始320实验')
        reference=check_reference(config,reference_run,audit,image_root=paths['image_root'])
        for name in ('train_config.json','experiment_config.json','run_summary.json'):
            p=(reference_run/name).resolve();audit['source_hashes'][str(p)]=sha256(p)
        if config['loss']=='ASL':
            from .asl import AsymmetricLabelLoss
            label_loss=AsymmetricLabelLoss(**config['asl'])
    elif config.get('loss')=='ASL':
        from .asl import AsymmetricLabelLoss
        from .compare_internal_ab import read_run
        if reference_run is None:raise ValueError('C1必须绑定原A对照')
        reference=read_run(reference_run)
        differences={k for k in config if config[k]!=reference['config'].get(k)}
        if differences!={'experiment_id','loss','loss_reduction','asl'}:
            raise ValueError('C1与A存在Loss之外的配置差异')
        if audit['split_manifest_sha256']!=reference['audit']['split_manifest_sha256']:
            raise ValueError('C1与A不是同一冻结划分')
        label_loss=AsymmetricLabelLoss(**config['asl'])
    if shutil.disk_usage('.').free < 2 * 1024**3:
        raise RuntimeError('磁盘不足2GiB')
    device = select_device(device_name)
    seed_everything(config['seed']); torch.set_num_threads(4)
    model = build_model(paths['pretrained_weight_path'], config['pretrained_weight_sha256'],
                        config['pretrained_registry'], backbone='densenet121').to(device)
    if not model.pretrained_backbone_verification or not model.pretrained_backbone_verification['all_equal']:
        raise RuntimeError('骨干没有通过原始ImageNet权重逐张量核验')
    if not all(p.requires_grad for p in model.parameters()):
        raise RuntimeError('对照实验要求完整微调')
    initial_digest = state_digest(model)
    if reference is not None and initial_digest!=reference['summary']['initial_state_sha256']:
        raise RuntimeError('与原对照的ImageNet/随机头初始张量不同，停止训练')
    settings = {**config, 'experiment_config': copy.deepcopy(config), 'size': config['input_size'],
        'preprocess': IMAGENET_PREPROCESS, 'normalization_mean': list(IMAGENET_MEAN),
        'normalization_std': list(IMAGENET_STD), 'aggregation': 'mean image probabilities per Study_id',
        'threshold': .5, 'device': device, 'learning_rate': config['backbone_lr'],
        'lr_schedule': config['scheduler'], 'scheduler_unit': 'epoch', 'weights': 'DenseNet121_Weights.IMAGENET1K_V1',
        'pretrained_provenance': approval, 'pretrained_backbone_verification': model.pretrained_backbone_verification,
        'initial_state_sha256': initial_digest, 'split_audit': audit,
        'data': {s: {'csv': str((paths['split_directory'] / ('internal_' + s + '.csv')).resolve()),
                     'images': str(paths['image_root'].resolve())} for s in ('train', 'valid')},
        'runtime': {'torch': str(torch.__version__), 'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None},
        'resume_supported': False, 'source_kind': audit['source_kind']}
    if reference is not None and label_loss is not None:
        settings.update(reference_model_sha256=reference['metrics']['checkpoint_sha256'],
            train_loss_definition='ASL per label -> mean classes -> mean images per Study -> mean Studies',
            validation_loss_definition='BCEWithLogitsLoss image mean; same as A',
            train_validation_loss_directly_comparable=False)
    if resolution_arm:
        settings.update(resolution_reference=str(reference_run.resolve()),
            resolution_reference_model_sha256=reference['metrics']['checkpoint_sha256'],
            resolution_changed_config_keys=['experiment_id','input_size'],
            input_source='original image files; no resized image cache',
            initial_state_matches_320_reference=True)
    optimizer = make_optimizer(model, settings)
    train_loader = make_study_loader(datasets['train'], settings)
    valid_loader = make_loader(datasets['valid'], config['batch_size'], num_workers=config['num_workers'],
                               seed=config['seed'], pin_memory=config['pin_memory'])
    origin = provenance({str(i): Path(p) for i, p in enumerate(audit['source_hashes'])}, config)
    origin['training_started'] = True
    output.mkdir(parents=True, exist_ok=False)
    dump(output / 'provenance.json', origin); dump(output / 'split_audit.json', audit)
    snapshot = output / 'split_snapshot'; snapshot.mkdir()
    for name in ('internal_train.csv', 'internal_valid.csv', 'subject_membership.csv', 'study_membership.csv',
                 'split_config.json', 'split_report.json', 'output_manifest.json', 'provenance.json'):
        shutil.copy2(paths['split_directory'] / name, snapshot / name)
    dump(output / 'experiment_config.json', config)
    stopper = EarlyStopping(**config['early_stopping'])
    best_auc, best_epoch, history, report = -1., None, [], None
    started = time.perf_counter()
    try:
        for epoch in range(1, config['epochs'] + 1):
            epoch_dir = output / 'epochs' / ('epoch_' + str(epoch)); epoch_dir.mkdir(parents=True)
            assert_sources_unchanged(audit['source_hashes'])
            lrs = apply_epoch_lr(optimizer, settings, epoch); reset_peak(device)
            t = time.perf_counter(); trained = run_study_epoch(model, optimizer, train_loader, device,
                **({'label_loss':label_loss} if label_loss is not None else {}))
            synchronize(device); train_seconds = time.perf_counter() - t
            report = write_sampling_epoch(output, datasets['train'], train_loader, epoch, trained, report)
            t = time.perf_counter()
            predictions, metrics = evaluate(model, valid_loader, device, epoch_dir / 'valid_image_predictions.csv')
            synchronize(device); valid_seconds = time.perf_counter() - t
            score = metrics['macro_auc']
            if score is None:
                raise RuntimeError('内部验证至少一类AUC未定义，停止选优')
            stopper.step(score, epoch)
            metrics.update(epoch=epoch, train_loss=trained['train_loss'], training_images=trained['training_images'],
                training_studies=trained['training_studies'], optimizer_steps=trained['optimizer_steps'],
                train_seconds=train_seconds, validation_seconds=valid_seconds, learning_rates=lrs,
                validation_mode='internal_subject_holdout', split_manifest_sha256=audit['split_manifest_sha256'],
                early_stopping=stopper.as_dict(), **peak_memory(device))
            if label_loss is not None:
                metrics.update(train_loss_name='study_balanced_ASL',validation_bce=metrics['validation_loss'],
                    validation_loss_name='image_mean_BCE',train_validation_loss_directly_comparable=False)
            save_model(model, epoch_dir / 'model.pth', {**settings, 'completed_epoch': epoch})
            metrics['checkpoint_sha256'] = sha256(epoch_dir / 'model.pth')
            # 真实服务器使用原load_model严格加载；模拟测试只替换此函数，不冒称DenseNet验收。
            restored = load_model(str(epoch_dir / 'model.pth'), device)
            if state_digest(restored) != state_digest(model):
                raise RuntimeError('检查点重载参数不一致')
            with torch.no_grad():
                image = next(iter(valid_loader))['image'].to(device)
                torch.testing.assert_close(model(image), restored(image), rtol=0, atol=0)
            del restored
            metrics['reload_equal'] = True
            save_csv(predictions, epoch_dir / 'val_study_predictions.csv')
            dump(epoch_dir / 'validation_metrics.json', metrics)
            dump(epoch_dir / 'train_config.json', {**settings, 'completed_epoch': epoch})
            if score > best_auc:
                best_auc, best_epoch = score, epoch
                for source, dest in [('model.pth', 'best_model.pth'), ('validation_metrics.json', 'validation_metrics.json'),
                        ('train_config.json', 'train_config.json'), ('val_study_predictions.csv', 'val_study_predictions.csv'),
                        ('valid_image_predictions.csv', 'valid_image_predictions.csv')]:
                    shutil.copy2(epoch_dir / source, output / dest)
            history.append({k: metrics[k] for k in ('epoch', 'train_loss', 'validation_loss', 'macro_auc', 'macro_f1',
                'training_images', 'training_studies', 'optimizer_steps', 'train_seconds', 'validation_seconds')})
            history[-1].update({k + '_lr': v for k, v in lrs.items()})
            if label_loss is not None:
                history[-1].update(validation_bce=metrics['validation_loss'],train_loss_name='study_balanced_ASL',
                    validation_loss_name='image_mean_BCE')
            pd.DataFrame(history).to_csv(output / 'history.csv', index=False, encoding='utf-8')
            print(json.dumps({'epoch': epoch, 'macro_auc': score, 'best_epoch': best_epoch, 'arm': config['experiment_id']}), flush=True)
            if stopper.should_stop:
                break
        assert_sources_unchanged(audit['source_hashes'])
        dump(output / 'run_summary.json', {'best_epoch': best_epoch, 'best_macro_auc': best_auc,
            'completed_epochs': len(history), 'requested_epochs': config['epochs'], 'elapsed_seconds': time.perf_counter() - started,
            'stop_reason': 'early_stopping' if stopper.should_stop else 'max_epochs_reached',
            'early_stopping': stopper.as_dict(), 'initial_state_sha256': initial_digest,
            'source_kind': audit['source_kind'], 'official_valid_used_for_selection': False, 'test_accessed': False})
        finish_manifest(output)
        return output
    except Exception as exc:
        dump(output / 'failure.json', {'type': type(exc).__name__, 'error': str(exc)})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paths', type=Path, help='服务器五路径JSON；相对值基于此文件目录')
    parser.add_argument('--arm', choices=['A', 'B', 'both'], default='both')
    parser.add_argument('--num-workers', type=int, default=None, help='同时覆盖A/B worker，默认8')
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--check', action='store_true'); actions.add_argument('--execute', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('outputs/Internal_AB_v1'))
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps', 'auto'], default='cuda')
    args = parser.parse_args(); configs = load_pair(args.num_workers)
    if not args.check and not args.execute:
        print(json.dumps({'configs': configs, 'required_paths': sorted(PATH_KEYS),
                          'training_started': False}, ensure_ascii=False, indent=2)); return
    if args.paths is None:
        parser.error('--check/--execute必须指定--paths')
    paths = read_paths(args.paths)
    if args.check:
        datasets, audit, approval = prepare(configs['A'], paths)
        shapes = {s: list(next(iter(make_loader(d, 1, num_workers=0)))['image'].shape) for s, d in datasets.items()}
        print(json.dumps({'split_audit': audit, 'weight': approval, 'sample_shapes': shapes,
                          'pretrained_model_loaded': False, 'training_started': False}, ensure_ascii=False, indent=2)); return
    if args.arm == 'both':
        if args.output.exists():
            raise FileExistsError('A/B根目录已存在，拒绝覆盖')
        # 同卡顺序执行；需要两卡并行时分别指定--arm和独立输出目录。
        execute(configs['A'], paths, args.output / 'A', args.device)
        execute(configs['B'], paths, args.output / 'B', args.device)
        from .compare_internal_ab import compare
        compare(args.output / 'A', args.output / 'B', args.output / 'comparison')
    else:
        execute(configs[args.arm], paths, args.output, args.device)
    print('COMPLETE ' + str(args.output.resolve()), flush=True)


if __name__ == '__main__':
    main()
