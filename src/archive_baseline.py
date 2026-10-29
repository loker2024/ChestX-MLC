"""归档现有最佳模型与历史预测；不训练、不修改来源、不补造历史环境。"""
import argparse
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import platform
import tempfile

import torch
import torchvision

from .experiment_config import load_config
from .experiments import REQUIRED, register, sha256, _validate


def archive_baseline(source: Path, config_path: Path, prediction_runs: list[Path],
                     validation_run: Path | None = None) -> Path:
    specification = load_config(config_path)
    if specification['experiment_id'] != 'E00_baseline':
        raise ValueError('Baseline归档配置编号必须为E00_baseline')
    settings, metrics, history = _validate({name:source/name for name in REQUIRED})
    expected = {'image_size':settings['size'], 'batch_size':settings['batch_size'],
                'epochs':int(history.epoch.max()), 'random_seed':settings['seed'],
                'learning_rate':settings['learning_rate'], 'optimizer':settings['optimizer'],
                'weight_decay':settings['weight_decay']}
    if any(specification[key] != value for key,value in expected.items()):
        raise ValueError('归档参考配置与真实训练记录不一致')
    checkpoint = torch.load(source/'best_model.pth',map_location='cpu',weights_only=True)
    if checkpoint.get('architecture') != specification['backbone'] or checkpoint.get('num_classes') != 10:
        raise ValueError('归档模型结构不符')
    if checkpoint['metadata'].get('completed_epoch',checkpoint['metadata'].get('epochs')) != metrics['epoch']:
        raise ValueError('模型与最佳成绩轮次不一致')
    digest = sha256(source/'best_model.pth')
    source_hashes = {path.name:sha256(path) for path in source.iterdir() if path.is_file()}
    extras = {path.name:path for path in source.iterdir() if path.is_file() and path.name not in
              {*REQUIRED,'provenance.json','experiment_notes.md','registration_complete.json','experiment_config.json'}}
    extras['experiment_config.json'] = config_path
    extras['historical/validation_metrics.json'] = source/'validation_metrics.json'
    all_runs = list(dict.fromkeys([*prediction_runs, *([validation_run] if validation_run else [])]))
    for index, folder in enumerate(all_runs):
        report = json.loads((folder/'prediction_report.json').read_text(encoding='utf-8'))
        if report.get('model_sha256') != digest or not report.get('model_unchanged'):
            raise ValueError('已有预测与归档模型不匹配')
        for path in folder.rglob('*'):
            if path.is_file():
                extras[f'predictions/run_{index:02d}/'+path.relative_to(folder).as_posix()] = path
    if validation_run and (validation_run/'valid_image_predictions.csv').is_file():
        extras['valid_image_predictions.csv'] = validation_run/'valid_image_predictions.csv'
    for folder in prediction_runs:
        if (folder/'test_study_predictions.csv').is_file():
            extras['test_study_predictions.csv'] = folder/'test_study_predictions.csv'
    manifest = {'schema':1, 'experiment_id':'E00_baseline', 'source':str(source.resolve()),
                'archived_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),
                'model_sha256':digest, 'source_sha256':source_hashes,
                'best_epoch':metrics['epoch'], 'total_epochs':int(history.epoch.max()),
                'historical_macro_auc':metrics['macro_auc'], 'historical_macro_f1':metrics['macro_f1'],
                'historical_environment':{key:settings.get(key) for key in
                    ('python','torch','torchvision','device','gpu','gpu_total_bytes')},
                'historical_optimizer_state_restored':settings.get('optimizer_state_restored'),
                'archive_runtime':{'python':platform.python_version(),'torch':str(torch.__version__),
                    'torchvision':str(torchvision.__version__), 'cuda_available':torch.cuda.is_available()},
                'prediction_runs':[str(p.resolve()) for p in all_runs], 'new_training':False}
    root = Path('outputs/experiments').resolve()
    root.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.baseline-info-',dir=root) as temp:
        info = Path(temp)/'baseline_manifest.json'
        info.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        extras['baseline_manifest.json'] = info
        destination = register(source,'E00_baseline',
            '历史4轮最佳权重与预测归档；原始指标另存historical；未进行新训练。',
            validation_run=validation_run,extra_files=extras)
    if any(sha256(source/name) != value for name,value in source_hashes.items()):
        raise RuntimeError('归档期间来源发生变化')
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=Path('outputs/baseline_v2'))
    parser.add_argument('--config',type=Path,default=Path('configs/experiments/E00_baseline.json'))
    parser.add_argument('--prediction-run',type=Path,action='append',default=[])
    parser.add_argument('--validation-run',type=Path)
    args = parser.parse_args()
    print(archive_baseline(args.source,args.config,args.prediction_run,args.validation_run))


if __name__ == '__main__':
    main()
