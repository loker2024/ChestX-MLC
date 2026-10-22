"""将完成的实验注册为独立快照，以验证集 Macro-AUC 比较。"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
import math
import warnings
from uuid import uuid4
import pandas as pd
from .fingerprints import comparison_group

REQUIRED = ['best_model.pth', 'train_config.json', 'history.csv',
            'val_study_predictions.csv', 'validation_metrics.json']


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate(files: dict[str, Path]) -> tuple[dict,dict,pd.DataFrame]:
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    config = json.loads(files['train_config.json'].read_text(encoding='utf-8'))
    metrics = json.loads(files['validation_metrics.json'].read_text(encoding='utf-8'))
    history = pd.read_csv(files['history.csv'],encoding='utf-8')
    for key in ('size','batch_size'):
        if not isinstance(config[key],int) or config[key] <= 0:
            raise ValueError('实验尺寸/批量无效')
    if not math.isfinite(config['learning_rate']) or not 0 < config['learning_rate'] < 1:
        raise ValueError('实验学习率无效')
    for key in ('macro_auc','macro_f1'):
        value = metrics[key]
        if value is None and key == 'macro_auc':
            continue
        if not isinstance(value,(int,float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('实验指标无效')
    epochs = pd.to_numeric(history['epoch'],errors='raise')
    if epochs.empty or epochs.isna().any() or epochs.duplicated().any() or (epochs < 1).any() or (epochs % 1 != 0).any() or metrics['epoch'] not in set(epochs):
        raise ValueError('最佳轮次与历史不一致')
    predicted = pd.read_csv(files['val_study_predictions.csv'],dtype={'Subject_id':str,'Study_id':str},encoding='utf-8')
    if predicted.empty or predicted[['Subject_id','Study_id']].isna().any().any() or predicted.Study_id.duplicated().any():
        raise ValueError('验证预测ID缺失或重复')
    probabilities = predicted[[f'p{i}' for i in range(1,11)]].apply(pd.to_numeric,errors='raise')
    if probabilities.isna().any().any() or ((probabilities < 0) | (probabilities > 1)).any().any():
        raise ValueError('验证预测概率无效')
    if 'study_count' in metrics and metrics['study_count'] != len(predicted):
        raise ValueError('验证预测数量与指标不一致')
    return config,metrics,history


def register(source: Path, experiment_id: str, notes: str,
             validation_run: Path | None = None) -> Path:
    if not re.fullmatch(r'exp_[0-9]+', experiment_id):
        raise ValueError('实验编号应为 exp_001 等格式')
    destination = Path('outputs/experiments') / experiment_id
    if destination.exists():
        raise FileExistsError('实验已存在，拒绝覆盖')
    files = {name:source/name for name in REQUIRED}
    if validation_run is not None:
        report = json.loads((validation_run/'prediction_report.json').read_text(encoding='utf-8'))
        if report['model_sha256'] != sha256(files['best_model.pth']) or not report.get('model_unchanged'):
            raise ValueError('验证复测与来源模型的哈希不一致')
        for name in ('val_study_predictions.csv','validation_metrics.json'):
            files[name] = validation_run/name
        # 预测指标不含训练轮次，从原最佳模型指标补入，不改原始验证文件。
    _,metrics,_ = _validate({**files,'validation_metrics.json':source/'validation_metrics.json'}) if validation_run else _validate(files)
    if validation_run:
        fresh = json.loads(files['validation_metrics.json'].read_text(encoding='utf-8'))
        fresh['epoch'] = metrics['epoch']
        metrics = fresh
        if comparison_group(metrics) is None:
            raise ValueError('验证复测缺少可验证的划分和评估协议指纹')
    hashes = {name:sha256(path) for name,path in files.items()}
    root = destination.parent.resolve()
    root.mkdir(parents=True,exist_ok=True)
    # 临时目录仅位于本次明确的实验根目录，异常时只清理自身临时文件。
    with tempfile.TemporaryDirectory(prefix='.staging-',dir=root) as temporary:
        stage = Path(temporary).resolve()
        if stage.parent != root:
            raise RuntimeError('临时目录超出实验根目录')
        for name,path in files.items():
            shutil.copy2(path,stage/name)
            if sha256(stage/name) != hashes[name] or sha256(path) != hashes[name]:
                raise RuntimeError('实验快照校验失败或源文件同时发生变化')
        if validation_run:
            (stage/'validation_metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        _validate({name:stage/name for name in REQUIRED})
        snapshot_hashes = {name:sha256(stage/name) for name in REQUIRED}
        (stage/'experiment_notes.md').write_text(
            f'# {experiment_id}\n\n来源：{source.resolve()}\n\n{notes}\n\n'
            f'最佳 Epoch：{metrics["epoch"]}；Macro-AUC：{metrics["macro_auc"]}；Macro-F1：{metrics["macro_f1"]}。\n'
            '只有相同验证标签/划分与评估协议的实验可进入同组排名；旧指标指纹缺失时不自动排名。\n',encoding='utf-8')
        (stage/'provenance.json').write_text(json.dumps({'source':str(source.resolve()),'sha256':snapshot_hashes,
             'source_sha256':hashes,'validation_run':str(validation_run.resolve()) if validation_run else None},indent=2),encoding='utf-8')
        (stage/'registration_complete.json').write_text('{"schema":1}',encoding='utf-8')
        if destination.exists():
            raise FileExistsError('实验编号已被其他操作使用，拒绝覆盖')
        stage.rename(destination.resolve())  # 同一文件系统原子发布；目标存在时失败，不覆盖。
    rebuild_summary()
    return destination


def rebuild_summary() -> None:
    rows = []
    for folder in sorted(Path('outputs/experiments').glob('exp_*')):
        try:
            config,metrics,history = _validate({name:folder/name for name in REQUIRED})
            notes = (folder/'experiment_notes.md').read_text(encoding='utf-8').strip()
            provenance = json.loads((folder/'provenance.json').read_text(encoding='utf-8'))
            if any(sha256(folder/name) != provenance['sha256'][name] for name in REQUIRED):
                raise ValueError('登记后实验快照被修改')
        except (OSError,ValueError,KeyError,TypeError) as exc:
            warnings.warn(f'跳过不完整或损坏的实验 {folder.name}: {exc}',RuntimeWarning)
            continue
        group = comparison_group(metrics)
        rows.append({'experiment_id': folder.name, 'model_name': 'ResNet18',
                     'epochs': int(history.epoch.max()), 'best_epoch': metrics['epoch'],
                     'image_size': config['size'], 'learning_rate': config['learning_rate'],
                     'batch_size': config['batch_size'], 'macro_auc': metrics['macro_auc'],
                     'macro_f1': metrics['macro_f1'], 'model_path': str((folder / 'best_model.pth').resolve()),
                     'notes': notes, 'comparison_group':group or ('unverified:'+folder.name),
                     'comparison_status':'verified' if group else 'unverified_legacy',
                     'notes_path': str((folder / 'experiment_notes.md').resolve())})
    # 汇总表为可重建索引；实验目录本身不可覆盖。
    columns = ['experiment_id','model_name','epochs','best_epoch','image_size','learning_rate',
               'batch_size','macro_auc','macro_f1','model_path','notes','comparison_group','comparison_status','notes_path']
    summary = pd.DataFrame(rows,columns=columns).sort_values(['comparison_group','macro_auc'],ascending=[True,False],na_position='last')
    summary['rank_in_group'] = pd.array([None]*len(summary),dtype='Int64')
    eligible = summary.comparison_status.eq('verified') & summary.macro_auc.notna()
    summary.loc[eligible,'rank_in_group'] = summary.loc[eligible].groupby('comparison_group').cumcount()+1
    output = Path('outputs').resolve()
    output.mkdir(exist_ok=True)
    temporary = output/('.summary-'+uuid4().hex+'.tmp')
    try:
        summary.to_csv(temporary,index=False,encoding='utf-8',mode='x')
        temporary.replace(output/'experiment_summary.csv')
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--id', required=True)
    parser.add_argument('--notes', required=True)
    parser.add_argument('--validation-run',type=Path,help='用同一模型的新验证复测补充旧实验指纹')
    args = parser.parse_args()
    print(register(args.source, args.id, args.notes,args.validation_run))


if __name__ == '__main__':
    main()
