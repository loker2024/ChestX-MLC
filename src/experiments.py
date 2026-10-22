"""将完成的实验注册为独立快照，以验证集 Macro-AUC 比较。"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import pandas as pd

REQUIRED = ['best_model.pth', 'train_config.json', 'history.csv',
            'val_study_predictions.csv', 'validation_metrics.json']


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def register(source: Path, experiment_id: str, notes: str) -> Path:
    if not re.fullmatch(r'exp_[0-9]+', experiment_id):
        raise ValueError('实验编号应为 exp_001 等格式')
    destination = Path('outputs/experiments') / experiment_id
    if destination.exists():
        raise FileExistsError('实验已存在，拒绝覆盖')
    for name in REQUIRED:
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    destination.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name in REQUIRED:
        hashes[name] = sha256(source / name)
        shutil.copy2(source / name, destination / name)
        if sha256(destination / name) != hashes[name]:
            raise RuntimeError('实验快照校验失败')
    metrics = json.loads((source / 'validation_metrics.json').read_text(encoding='utf-8'))
    (destination / 'experiment_notes.md').write_text(
        f'# {experiment_id}\n\n来源：{source.resolve()}\n\n{notes}\n\n'
        f'最佳 Epoch：{metrics["epoch"]}；Macro-AUC：{metrics["macro_auc"]}；'
        f'Macro-F1：{metrics["macro_f1"]}。\n只比较同一官方验证集和相同 Study 级评估的实验。\n', encoding='utf-8')
    (destination / 'provenance.json').write_text(json.dumps({'source': str(source.resolve()), 'sha256': hashes}, indent=2), encoding='utf-8')
    rebuild_summary()
    return destination


def rebuild_summary() -> None:
    rows = []
    for folder in sorted(Path('outputs/experiments').glob('exp_*')):
        if not all((folder / name).exists() for name in REQUIRED):
            continue
        config = json.loads((folder / 'train_config.json').read_text(encoding='utf-8'))
        metrics = json.loads((folder / 'validation_metrics.json').read_text(encoding='utf-8'))
        history = pd.read_csv(folder / 'history.csv', encoding='utf-8')
        rows.append({'experiment_id': folder.name, 'model_name': 'ResNet18',
                     'epochs': int(history.epoch.max()), 'best_epoch': metrics['epoch'],
                     'image_size': config['size'], 'learning_rate': config['learning_rate'],
                     'batch_size': config['batch_size'], 'macro_auc': metrics['macro_auc'],
                     'macro_f1': metrics['macro_f1'], 'model_path': str((folder / 'best_model.pth').resolve()),
                     'notes': (folder / 'experiment_notes.md').read_text(encoding='utf-8').strip(),
                     'notes_path': str((folder / 'experiment_notes.md').resolve())})
    # 汇总表为可重建索引；实验目录本身不可覆盖。
    pd.DataFrame(rows).sort_values('macro_auc', ascending=False, na_position='last').to_csv(
        'outputs/experiment_summary.csv', index=False, encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--id', required=True)
    parser.add_argument('--notes', required=True)
    args = parser.parse_args()
    print(register(args.source, args.id, args.notes))


if __name__ == '__main__':
    main()
