"""官方 train 的五外折 / 三内折编排；只有显式 run 才启动 GPU 工作。"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
import yaml

from common.config import Config, load_config
from data.patient_folds import assign_patient_folds, read_metadata, sha256_file, study_table, verify_manifest
from infer.patient_count_prior import recipe_fingerprint

REPOSITORY = Path(__file__).resolve().parents[1]


def fixed_recipe(base: Config) -> Config:
    """固定预算、纯 ImageNet 初始化；保留 D002a 调度及增强参数。"""
    if base.model.arch != 'timm' or base.model.backbone != 'convnext_tiny.fb_in1k':
        raise ValueError('OOF 预算配方要求 ImageNet ConvNeXt-Tiny')
    if base.train.loss != 'bce' or base.train.study_sampler != 'uniform':
        raise ValueError('OOF 基础配方要求 BCE 与 uniform 采样，不与其他训练实验混用')
    return replace(base, seed=42,
                   data=replace(base.data, full_train=False, resplit=replace(base.data.resplit, enabled=False)),
                   model=replace(base.model, pretrained=True),
                   train=replace(base.train, epochs=5, ema_enabled=True, early_stop_patience=0,
                                 save_last=True, distill_enabled=False, distill_cache=None,
                                 require_pretrained=True))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def _prepare_pool(source: Path, base: Config, root: Path, n_splits: int,
                  seed: int, recipe_hash: str, manifest_name: str) -> dict[str, Any]:
    frame = read_metadata(source)
    studies = study_table(frame)
    assignments = assign_patient_folds(studies, n_splits, seed)
    manifest = {'schema_version': 1, 'split': 'train', 'source_csv': str(source.resolve()),
                'source_sha256': sha256_file(source), 'recipe_sha256': recipe_hash,
                'seed': seed, 'n_splits': n_splits, 'studies': [dict(row, fold=assignments[row['Study_id']]) for row in studies],
                'outer_folds': []}
    root.mkdir(parents=True, exist_ok=True)
    for fold in range(n_splits):
        folder = root / f'fold_{fold}'
        folder.mkdir()
        train_csv, valid_csv = folder / 'train.csv', folder / 'valid.csv'
        mask = frame['Study_id'].map(assignments) == fold
        frame.loc[~mask].to_csv(train_csv, index=False, encoding='utf-8')
        frame.loc[mask].to_csv(valid_csv, index=False, encoding='utf-8')
        config = replace(base, data=replace(base.data,
                          train=replace(base.data.train, csv=train_csv.resolve()),
                          valid=replace(base.data.train, csv=valid_csv.resolve())),
                         output=replace(base.output, run_dir=(folder / 'runs').resolve()))
        config_path = folder / 'config.yaml'
        config_path.write_text(yaml.safe_dump(config.to_dict(), allow_unicode=True, sort_keys=False), encoding='utf-8')
        entry = {'fold': fold, 'train_csv': str(train_csv.resolve()), 'valid_csv': str(valid_csv.resolve()),
                 'config': str(config_path.resolve()), 'output_dir': str(folder.resolve()),
                 'train_subjects': sorted(frame.loc[~mask, 'Subject_id'].unique().tolist()),
                 'valid_subjects': sorted(frame.loc[mask, 'Subject_id'].unique().tolist())}
        for key in ('train_csv', 'valid_csv', 'config'):
            entry[f'{key}_sha256'] = sha256_file(entry[key])
        manifest['outer_folds'].append(entry)
    _write_json(root / manifest_name, manifest)
    return manifest


def prepare(config_path: str | Path, output: str | Path) -> Path:
    """仅准备 CSV/YAML，不读取官方 valid/test 标签，不运行训练。"""
    root = Path(output).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError('输出目录非空，拒绝覆盖')
    original = load_config(config_path)
    source = original.data.train.csv.resolve()
    if source in {original.data.valid.csv.resolve(), original.data.test.csv.resolve()}:
        raise ValueError('官方 train 来源不能是 valid/test')
    base = fixed_recipe(original)
    base = replace(base, data=replace(base.data,
                   train=replace(base.data.train, images=base.data.train.images.resolve())))
    root.mkdir(parents=True, exist_ok=True)
    full_config = replace(base, data=replace(base.data,
                          train=replace(base.data.train, csv=source),
                          valid=replace(base.data.train, csv=source)),
                          output=replace(base.output, run_dir=root / 'train_only_runs'))
    (root / 'train_only.yaml').write_text(yaml.safe_dump(full_config.to_dict(), allow_unicode=True,
                                                       sort_keys=False), encoding='utf-8')
    review_config = replace(full_config, data=replace(full_config.data,
                            valid=replace(original.data.valid, csv=original.data.valid.csv.resolve(),
                                          images=original.data.valid.images.resolve())))
    (root / 'train_only_eval.yaml').write_text(yaml.safe_dump(review_config.to_dict(), allow_unicode=True,
                                                            sort_keys=False), encoding='utf-8')
    recipe_hash = recipe_fingerprint([base])
    manifest = _prepare_pool(source, base, root, 5, 2026, recipe_hash, 'manifest.json')
    for entry in manifest['outer_folds']:
        inner_root = root / f"outer_{entry['fold']}"
        inner = _prepare_pool(Path(entry['train_csv']), base, inner_root, 3,
                              2026 + entry['fold'], recipe_hash, 'inner_manifest.json')
        inner['parent_manifest'] = str(root / 'manifest.json')
        inner['parent_manifest_sha256'] = sha256_file(root / 'manifest.json')
        inner['outer_fold'] = entry['fold']
        _write_json(inner_root / 'inner_manifest.json', inner)
    return root / 'manifest.json'


def join_oof(frames: list[pd.DataFrame], manifest: dict[str, Any]) -> pd.DataFrame:
    """以 study ID 精确对齐并核验标签、患者、折及概率。"""
    expected = {row['Study_id']: row for row in manifest['studies']}
    frame = pd.concat(frames, ignore_index=True)
    columns = ['Subject_id', 'Study_id', 'fold'] + [f'label_{i}' for i in range(1, 11)] + [f'p{i}' for i in range(1, 11)]
    if not set(columns).issubset(frame.columns) or frame['Study_id'].duplicated().any() or set(frame['Study_id']) != set(expected):
        raise ValueError('OOF study 必须完整且恰好一次')
    for row in frame.to_dict('records'):
        reference = expected[row['Study_id']]
        if row['Subject_id'] != reference['Subject_id'] or int(row['fold']) != reference['fold']:
            raise ValueError('OOF 患者或折不符')
        if [float(row[f'label_{i}']) for i in range(1, 11)] != reference['labels']:
            raise ValueError('OOF 标签不符')
    values = frame[[f'p{i}' for i in range(1, 11)]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError('OOF 概率无效')
    return frame.set_index('Study_id').loc[list(expected)].reset_index()[columns]


def _check_checkpoint(path: Path, config: Config) -> dict[str, Any]:
    from common.checkpoint import load_checkpoint
    checkpoint = load_checkpoint(path)
    if checkpoint.get('config') != config.to_dict() or checkpoint.get('seed') != 42:
        raise ValueError('checkpoint 配方/seed 不符')
    metadata = checkpoint.get('metadata') or {}
    if metadata.get('warm_start_sha256') or metadata.get('warm_start_from'):
        raise ValueError('OOF 禁止比赛 checkpoint 初始化')
    if not metadata.get('pretrained_loaded'):
        raise ValueError('checkpoint 未证明 ImageNet 权重已加载')
    return checkpoint


def export_last_ema(checkpoint_path: str | Path, output_path: str | Path) -> Path:
    """从第 5 轮完整断点导出 EMA，用于 OOF 或全 train 固定配方复核。"""
    from common.checkpoint import load_checkpoint, save_checkpoint
    from common.config import config_from_dict
    from model.network import build_model

    path = Path(checkpoint_path).resolve()
    last = load_checkpoint(path)
    config = config_from_dict(last['config'])
    if config != fixed_recipe(config):
        raise ValueError('断点不满足固定 OOF 配方')
    _check_checkpoint(path, config)
    state = (last.get('extras') or {}).get('ema_state')
    if last.get('epoch') != 5 or not state or not state.get('model') or state.get('updates', 0) <= 0:
        raise ValueError('必须具备固定第5轮 last.pt 的 EMA 状态与正的 updates')
    output = Path(output_path).resolve()
    if output == path or output.exists():
        raise ValueError('EMA 导出路径必须为不存在的新文件')
    model = build_model(replace(config, model=replace(config.model, pretrained=False)))
    model.load_state_dict(state['model'])
    return save_checkpoint(output, model, epoch=5, config=config, seed=42,
                           metadata={**(last.get('metadata') or {}), 'weights_kind': 'fixed_last_ema',
                                     'ema_updates': state['updates'], 'source_last_sha256': sha256_file(path)})


def _run_fold(entry: dict[str, Any], manifest: dict[str, Any], manifest_path: Path) -> pd.DataFrame:
    from common.checkpoint import load_checkpoint
    folder = Path(entry['output_dir'])
    config = load_config(entry['config'])
    if config != fixed_recipe(config) or recipe_fingerprint([config]) != manifest['recipe_sha256']:
        raise ValueError('折配置不满足冻结配方')
    if set(entry['train_subjects']) & set(entry['valid_subjects']):
        raise ValueError('训练与留出患者相交')
    receipt_path, prediction_path = folder / 'provenance.json', folder / 'predictions.csv'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        if receipt['manifest_sha256'] != sha256_file(manifest_path) or receipt['config_sha256'] != entry['config_sha256']:
            raise ValueError('已有产物 manifest/config 来源不符')
        for key in ('fold', 'train_subjects', 'valid_subjects'):
            if receipt[key] != entry[key]:
                raise ValueError('已有产物折或患者来源不符')
        if receipt['recipe_sha256'] != manifest['recipe_sha256']:
            raise ValueError('已有产物模型配方不符')
        if (receipt.get('epoch') != 5 or receipt.get('weights_kind') != 'fixed_last_ema'
                or receipt.get('initialization') != 'imagenet'):
            raise ValueError('已有产物不是固定轮数 ImageNet 初始化的 EMA')
        for key in ('last_checkpoint', 'ema_checkpoint', 'predictions'):
            if sha256_file(receipt[key]) != receipt[f'{key}_sha256']:
                raise ValueError('已有 OOF 产物哈希不符')
        last = _check_checkpoint(Path(receipt['last_checkpoint']), config)
        ema = load_checkpoint(receipt['ema_checkpoint'])
        if ema.get('config') != config.to_dict() or (ema.get('metadata') or {}).get('source_last_sha256') != receipt['last_checkpoint_sha256']:
            raise ValueError('末轮 EMA 与 last.pt 来源不符')
        if last.get('epoch') != 5 or ema.get('epoch') != 5 or (ema.get('metadata') or {}).get('weights_kind') != 'fixed_last_ema':
            raise ValueError('已有产物不是固定第5轮 EMA')
        if (ema.get('metadata') or {}).get('ema_updates', 0) <= 0:
            raise ValueError('已有 EMA 缺少有效更新次数')
    else:
        runs = config.output.run_dir
        existing = sorted(runs.iterdir()) if runs.exists() else []
        if len(existing) > 1:
            raise ValueError('同折存在多个训练目录，拒绝猜测来源')
        if existing:
            run_dir = existing[0]
            last = _check_checkpoint(run_dir / 'last.pt', config)
            if last.get('epoch', 0) < 5:
                subprocess.run([sys.executable, str(REPOSITORY / 'src/run_train.py'), '--config', entry['config'],
                                '--resume', str(run_dir)], check=True, cwd=REPOSITORY)
        else:
            subprocess.run([sys.executable, str(REPOSITORY / 'src/run_train.py'), '--config', entry['config'],
                            '--tag', 'patient_oof'], check=True, cwd=REPOSITORY)
            discovered = sorted(runs.iterdir())
            if len(discovered) != 1:
                raise ValueError('无法唯一定位时间戳训练目录')
            run_dir = discovered[0]
        last_path = run_dir / 'last.pt'
        last = _check_checkpoint(last_path, config)
        ema_state = (last.get('extras') or {}).get('ema_state')
        if last.get('epoch') != 5 or not ema_state or not ema_state.get('model'):
            raise ValueError('必须具备固定第5轮 last.pt 的 EMA 状态')
        ema_path = folder / 'fixed_last_ema.pt'
        if ema_path.exists():
            exported = load_checkpoint(ema_path)
            if (exported.get('epoch') != 5 or exported.get('config') != config.to_dict()
                    or (exported.get('metadata') or {}).get('weights_kind') != 'fixed_last_ema'
                    or (exported.get('metadata') or {}).get('source_last_sha256') != sha256_file(last_path)):
                raise ValueError('中断前的 EMA 导出与当前 last.pt 不匹配')
        else:
            export_last_ema(last_path, ema_path)
        subprocess.run([sys.executable, str(REPOSITORY / 'src/run_eval.py'), '--config', entry['config'],
                        '--checkpoint', str(ema_path), '--pred-out', str(prediction_path)], check=True, cwd=REPOSITORY)
        receipt = {'manifest_sha256': sha256_file(manifest_path), 'config_sha256': entry['config_sha256'],
                   'recipe_sha256': manifest['recipe_sha256'], 'fold': entry['fold'],
                   'train_subjects': entry['train_subjects'], 'valid_subjects': entry['valid_subjects'],
                   'epoch': 5, 'weights_kind': 'fixed_last_ema', 'initialization': 'imagenet',
                   'last_checkpoint': str(last_path.resolve()), 'ema_checkpoint': str(ema_path.resolve()),
                   'predictions': str(prediction_path.resolve())}
        for key in ('last_checkpoint', 'ema_checkpoint', 'predictions'):
            receipt[f'{key}_sha256'] = sha256_file(receipt[key])
        _write_json(receipt_path, receipt)
    predictions = read_metadata(prediction_path)
    subjects = {row['Study_id']: row['Subject_id'] for row in manifest['studies']}
    predictions['Subject_id'] = predictions['Study_id'].map(subjects)
    predictions['fold'] = entry['fold']
    local_manifest = {'studies': [row for row in manifest['studies'] if row['fold'] == entry['fold']]}
    return join_oof([predictions], local_manifest)


def run(root: str | Path, stage: str, outer_fold: int | None, fold: int | None) -> None:
    """训练选定折，并在全部折均已有可核验产物时聚合。"""
    root = Path(root).resolve()
    top_path = root / 'manifest.json'
    top = json.loads(top_path.read_text(encoding='utf-8'))
    verify_manifest(top)
    if stage == 'inner':
        if outer_fold is None or outer_fold not in range(5):
            raise ValueError('inner 必须指定 --outer-fold 0..4')
        folder = root / f'outer_{outer_fold}'
        path = folder / 'inner_manifest.json'
    else:
        folder, path = root, top_path
    manifest = json.loads(path.read_text(encoding='utf-8'))
    verify_manifest(manifest)
    if stage == 'inner' and (manifest['parent_manifest_sha256'] != sha256_file(top_path)
                            or manifest['source_sha256'] != top['outer_folds'][outer_fold]['train_csv_sha256']):
        raise ValueError('内层来源与外层训练池不符')
    if fold is not None and fold not in range(manifest['n_splits']):
        raise ValueError('fold 编号越界')
    for entry in manifest['outer_folds']:
        if fold is None or fold == entry['fold']:
            _run_fold(entry, manifest, path)
    if all((Path(entry['output_dir']) / 'provenance.json').exists() for entry in manifest['outer_folds']):
        frames = [_run_fold(entry, manifest, path) for entry in manifest['outer_folds']]
        output = folder / ('inner_oof.csv' if stage == 'inner' else 'outer_oof.csv')
        join_oof(frames, manifest).to_csv(output, index=False, encoding='utf-8')


def build_parser() -> argparse.ArgumentParser:
    """构造 CLI；prepare 不训练，run 才显式启动作业。"""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare_parser = sub.add_parser('prepare')
    prepare_parser.add_argument('--config', default='configs/exp_D002a.yaml')
    prepare_parser.add_argument('--out', required=True)
    prepare_parser.add_argument('--folds', type=int, choices=[5], default=5)
    prepare_parser.add_argument('--inner-folds', type=int, choices=[3], default=3)
    prepare_parser.add_argument('--seed', type=int, choices=[2026], default=2026)
    prepare_parser.add_argument('--epochs', type=int, choices=[5], default=5)
    export_parser = sub.add_parser('export-last-ema')
    export_parser.add_argument('--checkpoint', required=True)
    export_parser.add_argument('--out', required=True)
    run_parser = sub.add_parser('run')
    location = run_parser.add_mutually_exclusive_group(required=True)
    location.add_argument('--root')
    location.add_argument('--manifest', help='prepare 生成的顶层 manifest.json')
    run_parser.add_argument('--stage', choices=('outer', 'inner'), default='outer')
    run_parser.add_argument('--outer-fold', type=int)
    run_parser.add_argument('--fold', type=int)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == 'prepare':
        print(prepare(args.config, args.out))
    elif args.command == 'export-last-ema':
        print(export_last_ema(args.checkpoint, args.out))
    else:
        if args.manifest and Path(args.manifest).name != 'manifest.json':
            raise ValueError('--manifest 必须指向顶层 manifest.json')
        root = args.root or str(Path(args.manifest).resolve().parent)
        run(root, args.stage, args.outer_fold, args.fold)


if __name__ == '__main__':
    main()
