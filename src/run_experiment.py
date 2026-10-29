"""配置驱动入口：默认仅检查，显式--execute才训练；复用现有训练函数。"""
import argparse
import json
from pathlib import Path

from .experiment_config import load_config, load_data_config
from .experiments import register


def plan(config_path: Path) -> dict:
    if json.loads(config_path.read_text(encoding='utf-8-sig')).get('experiment_id') in ('PT1','PT2_ES','PT2_WC_ES','B1_DenseNet121','B1_DenseNet121_ImageNet','B1_DenseNet121_320_Study','B1_DenseNet121_RA_320_Study','A1_DenseNet121_320_Study_LightAug','B2_EfficientNetB2_320_Study'):
        from .pt1 import plan as pt1_plan
        return pt1_plan(config_path)
    config = load_config(config_path)
    data = load_data_config(config)
    paths_exist = {split: {key: Path(value).exists() for key, value in data[split].items()}
                   for split in ('train', 'valid', 'test')}
    return {'experiment': config, 'data': data, 'paths_exist': paths_exist,
            'run_directory': str(Path('outputs/runs')/config['experiment_id']),
            'archive_directory': str(Path('outputs/experiments')/config['experiment_id']),
            'training_started': False}


def execute(config_path: Path, device: str = 'auto') -> Path:
    """供训练同学显式调用；本轮仅对调度进行mock接口测试。"""
    from .full_train import run as first_epoch
    from .continue_train import run as remaining_epochs
    if json.loads(config_path.read_text(encoding='utf-8-sig')).get('experiment_id') in ('PT1','PT2_ES','PT2_WC_ES','B1_DenseNet121','B1_DenseNet121_ImageNet','B1_DenseNet121_320_Study','B1_DenseNet121_RA_320_Study','A1_DenseNet121_320_Study_LightAug','B2_EfficientNetB2_320_Study'):
        from .pt1 import execute as pt1_execute
        return pt1_execute(config_path,device)
    prepared = plan(config_path)
    config = prepared['experiment']
    if not all(all(paths.values()) for split, paths in prepared['paths_exist'].items()
               if split in ('train','valid')):
        raise FileNotFoundError('训练/验证CSV或图片目录不存在，请修改data_config')
    root = Path(prepared['run_directory'])
    archive = Path(prepared['archive_directory'])
    if root.exists() or archive.exists():
        raise FileExistsError('实验编号已使用，请创建新编号，不覆盖旧实验')
    root.mkdir(parents=True, exist_ok=False)
    (root/'experiment_config.json').write_text(
        json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    try:
        first = root/'initial'
        first_epoch(device,Path(config['data_config']),first,config['random_seed'],experiment=config)
        result = first
        if config['epochs'] > 1:
            result = root/'result'
            remaining_epochs(first,result,config['epochs']-1,device_name=device,
                             resume_state=first/'training_state.pth')
        extras = {'experiment_config.json':root/'experiment_config.json'}
        if (result/'valid_image_predictions.csv').is_file():
            extras['valid_image_predictions.csv'] = result/'valid_image_predictions.csv'
        registered = register(result,config['experiment_id'],config['notes'],extra_files=extras)
        (root/'run_result.json').write_text(json.dumps({'result':str(result.resolve()),
            'archive':str(registered.resolve()),'epochs':config['epochs']},indent=2),encoding='utf-8')
        return registered
    except Exception as exc:
        (root/'failure.json').write_text(json.dumps({'type':type(exc).__name__,'error':str(exc)},
            ensure_ascii=False,indent=2),encoding='utf-8')
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--execute',action='store_true',help='显式开始训练；省略则仅检查配置')
    parser.add_argument('--device',choices=['auto','cuda','mps','cpu'],default='auto')
    args = parser.parse_args()
    if args.execute:
        print(execute(args.config,args.device))
    else:
        print(json.dumps(plan(args.config),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
