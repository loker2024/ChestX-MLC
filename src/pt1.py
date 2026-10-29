"""PT1配置、许可门禁和单批真实训练图片测试；完整训练仅显式--execute。"""
import argparse
from datetime import datetime
import json
import math
import re
from pathlib import Path
from uuid import uuid4

import torch

from .data import ChestXrayDataset, make_loader, IMAGENET_PREPROCESS
from .devices import select_device
from .model import build_model, load_model, classification_head
from .pretrained import require_approved_weight, file_sha256
from .train import seed_everything, make_optimizer, train_step, save_model


def validate_config(config: dict) -> dict:
    if not isinstance(config, dict) or config.get('experiment_id') not in ('PT1','PT2_ES','PT2_WC_ES','B1_DenseNet121','B1_DenseNet121_ImageNet','B1_DenseNet121_320_Study','B1_DenseNet121_RA_320_Study','A1_DenseNet121_320_Study_LightAug','B2_EfficientNetB2_320_Study'):
        raise ValueError('不支持的预训练实验编号')
    new_strategy = config['experiment_id'] != 'PT1'
    expected = {'experiment_id':'PT1','backbone':'resnet18','pretrained':True,
        'input_size':224,'num_classes':10,'optimizer':'AdamW','loss':'BCEWithLogitsLoss',
        'study_aggregation':'mean','validation_mode':'official_valid_eval_only',
        'normalization':'imagenet','normalization_mean':[.485,.456,.406],
        'normalization_std':[.229,.224,.225],'scheduler':'none','augmentation':'none','sampler':'shuffle'}
    required = {*expected,'pretrained_weight_path','pretrained_weight_sha256','pretrained_registry',
                'backbone_lr','head_lr','weight_decay','epochs','seed','batch_size','data_config'}
    if new_strategy:
        expected['experiment_id'] = config['experiment_id']
        expected['scheduler'] = config.get('scheduler')
        required.add('early_stopping')
    if config['experiment_id'] in ('B1_DenseNet121','B1_DenseNet121_ImageNet','B1_DenseNet121_320_Study','B1_DenseNet121_RA_320_Study','A1_DenseNet121_320_Study_LightAug','B2_EfficientNetB2_320_Study'):
        expected['backbone'] = 'densenet121'
    if config['experiment_id'] in ('B1_DenseNet121_320_Study','B1_DenseNet121_RA_320_Study','A1_DenseNet121_320_Study_LightAug','B2_EfficientNetB2_320_Study'):
        expected.update(input_size=320,sampler='study_batch')
        required.update({'image_size','loss_reduction','studies_per_batch','max_images_per_batch','num_workers','pin_memory'})
        if config.get('image_size')!=320 or type(config.get('image_size')) is not int or config.get('loss_reduction')!='study_balanced_image_bce':
            raise ValueError('320 Study配置必须使用320输入和均衡逐图BCE')
        for key in ('studies_per_batch','max_images_per_batch','num_workers'):
            if type(config.get(key)) is not int or config[key]<(0 if key=='num_workers' else 1):
                raise ValueError(key+'必须是有效整数')
        if type(config.get('pin_memory')) is not bool:
            raise ValueError('pin_memory必须为布尔值')
    if config['experiment_id']=='B2_EfficientNetB2_320_Study':
        expected['backbone']='efficientnet_b2'
    if config['experiment_id']=='A1_DenseNet121_320_Study_LightAug':
        from .augmentation import validate_augmentation
        validate_augmentation(config.get('augmentation'))
        expected['augmentation']=config['augmentation']
    if config['experiment_id']=='B1_DenseNet121_RA_320_Study':
        required.update({'model_backend','pretrained_model','pretrained_variant'})
        if (config.get('model_backend'),config.get('pretrained_model'),config.get('pretrained_variant'))!=('timm','densenet121.ra_in1k','densenet121_ra'):
            raise ValueError('RA实验必须显式绑定timm densenet121.ra_in1k，不能替换为A1/TV')
    if not isinstance(config,dict) or set(config) != required:
        raise ValueError('PT1配置字段缺失或存在未知选项')
    for key,value in expected.items():
        if config[key] != value or type(config[key]) is not type(value):
            raise ValueError(f'PT1当前仅支持{key}={value!r}')
    for key in ('epochs','seed','batch_size'):
        if type(config[key]) is not int or config[key] < (0 if key == 'seed' else 1):
            raise ValueError(f'{key}必须是合法整数')
    if config['seed'] >= 2**32 or config['batch_size'] < 2:
        raise ValueError('seed或batch_size范围错误')
    for key in ('backbone_lr','head_lr','weight_decay'):
        value = config[key]
        if type(value) not in (int,float) or not math.isfinite(value):
            raise ValueError(f'{key}不是有限数值')
    if not all(0 < config[key] < 1 for key in ('backbone_lr','head_lr')) or config['weight_decay'] < 0:
        raise ValueError('学习率或衰减范围错误')
    for key in ('pretrained_weight_path','pretrained_weight_sha256','pretrained_registry','data_config'):
        if not isinstance(config[key],str) or not config[key].strip():
            raise ValueError(f'{key}必须为非空字符串')
    if not re.fullmatch(r'[0-9a-f]{64}',config['pretrained_weight_sha256']):
        raise ValueError('预训练权重必须提供完整小写SHA-256')
    if new_strategy:
        from .early_stopping import EarlyStopping
        from .lr_schedule import validate_schedule
        stop = config['early_stopping']
        if not isinstance(stop, dict) or set(stop) != {'patience','min_delta'}:
            raise ValueError('早停需显式指定patience/min_delta')
        EarlyStopping(stop['patience'], stop['min_delta'])
        if config['experiment_id'] in ('PT2_WC_ES','B1_DenseNet121','B1_DenseNet121_ImageNet','B1_DenseNet121_320_Study','B1_DenseNet121_RA_320_Study','A1_DenseNet121_320_Study_LightAug','B2_EfficientNetB2_320_Study'):
            validate_schedule(config['scheduler'], config['epochs'])
        elif config['scheduler'] != 'none':
            raise ValueError('PT2_ES只改变早停，不启用调度')
    return dict(config)


def load_config(path: Path) -> dict:
    return validate_config(json.loads(path.read_text(encoding='utf-8-sig')))


def data_settings(config: dict) -> dict:
    data = json.loads(Path(config['data_config']).read_text(encoding='utf-8-sig'))
    for split in ('train','valid'):
        if not isinstance(data.get(split),dict) or set(data[split]) != {'csv','images'}:
            raise ValueError('PT1需要明确的train/valid CSV与图片路径')
    result={**data,'size':config['input_size'],'batch_size':config['batch_size']}
    if config.get('sampler')=='study_batch':
        result.update(num_workers=config['num_workers'],pin_memory=config['pin_memory'])
    return result


def approved_model(config: dict) -> torch.nn.Module:
    extra={key:config[key] for key in ('model_backend','pretrained_variant') if key in config}
    return build_model(config['pretrained_weight_path'],config['pretrained_weight_sha256'],
                       config['pretrained_registry'],backbone=config['backbone'],**extra)


def weight_review(config: dict) -> dict:
    args = (Path(config['pretrained_weight_path']),config['pretrained_weight_sha256'],Path(config['pretrained_registry']))
    extra={'backbone':config['backbone']} if config['backbone']!='resnet18' else {}
    if 'pretrained_variant' in config:extra['variant']=config['pretrained_variant']
    return require_approved_weight(*args,**extra)


def plan(config_path: Path) -> dict:
    config = load_config(config_path)
    try:
        provenance = weight_review(config)
        review = {'ready':True,'provenance':provenance}
    except (PermissionError,ValueError,FileNotFoundError,KeyError) as exc:
        review = {'ready':False,'blocking_type':type(exc).__name__,'blocking_reason':str(exc)}
    return {'experiment':config,'weight_review':review,'data':data_settings(config),
            'full_training_started':False,'normalization_mean':[.485,.456,.406],
            'normalization_std':[.229,.224,.225],'geometry':'RGB aspect-preserving letterbox, no center crop'}


def smoke(config_path: Path, device_name: str = 'auto', interface_only: bool = False,
          batch_size: int = 2) -> Path:
    config = load_config(config_path)
    if config.get('sampler')=='study_batch':
        raise ValueError('320 Study冒烟请用python -m src.study_smoke，不使用旧224图片级冒烟')
    seed_everything(config['seed'])
    torch.set_num_threads(4)
    device = select_device(device_name)
    # 接口模式明确从零构建；不是对pending权重的门禁绕过，也不作为PT1成绩。
    if type(batch_size) is not int or not 2 <= batch_size <= config['batch_size']:
        raise ValueError('单批测试批量必须在2到配置batch_size之间')
    model = (build_model(backbone=config['backbone']) if interface_only else approved_model(config)).to(device)
    data = data_settings(config)
    dataset = ChestXrayDataset(data['train']['csv'],data['train']['images'],size=224,normalization='imagenet')
    if (len(dataset),dataset.frame.Study_id.nunique()) != (30344,18513):
        raise ValueError('PT1真实冒烟测试要求官方训练集，不能冒充合成数据')
    workers = data.get('num_workers',4)
    batch = next(iter(make_loader(dataset,batch_size,shuffle=True,seed=config['seed'],num_workers=workers)))
    metadata = {'experiment_id':config['experiment_id']+('_interface_only' if interface_only else '_smoke'),
        'backbone':config['backbone'],'pretrained':not interface_only,'size':224,'batch_size':batch_size,'seed':config['seed'],'num_workers':workers,
        'data':data,'preprocess':IMAGENET_PREPROCESS,'normalization':'imagenet',
        'normalization_mean':config['normalization_mean'],'normalization_std':config['normalization_std'],
        'aggregation':'mean image probabilities per Study_id','threshold':.5,'epochs':0,
        'backbone_lr':config['backbone_lr'],'head_lr':config['head_lr'],
        'learning_rate':config['backbone_lr'],'weight_decay':config['weight_decay'],'optimizer':'AdamW',
        'validation_mode':config['validation_mode'],'device':device,'smoke_only':True,'steps':1,
        'pretrained_provenance':model.pretrained_provenance,
        'pretrained_backbone_verification':model.pretrained_backbone_verification}
    optimizer = make_optimizer(model,metadata)
    if isinstance(config['scheduler'], dict):
        from .lr_schedule import apply_epoch_lr
        metadata['lr_schedule'] = config['scheduler']
        apply_epoch_lr(optimizer, metadata, 1)
    head = classification_head(model)
    head_ids = {id(p) for p in head.parameters()}
    first_backbone = next(p for p in model.parameters() if id(p) not in head_ids)
    before_backbone,before_head = first_backbone.detach().clone(),head.weight.detach().clone()
    csv_hash = file_sha256(Path(data['train']['csv']))
    loss,shape = train_step(model,batch['image'].to(device),batch['label'].to(device),optimizer)
    deltas = {'backbone':float((before_backbone-first_backbone.detach()).abs().max()),
              'head':float((before_head-head.weight.detach()).abs().max())}
    if min(deltas.values()) <= 0:
        raise RuntimeError('骨干或分类头未更新')
    model.eval()
    with torch.no_grad():
        expected = model(batch['image'].to(device))
    name = config['experiment_id']+('_interface_smoke_' if interface_only else '_pretrained_smoke_')
    output = Path('outputs')/(name+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid4().hex[:8])
    output.mkdir(parents=True,exist_ok=False)
    save_model(model,output/'model.pth',metadata)
    restored = load_model(str(output/'model.pth'),device)
    with torch.no_grad():
        actual = restored(batch['image'].to(device))
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    if file_sha256(Path(data['train']['csv'])) != csv_hash:
        raise RuntimeError('官方训练CSV发生变化')
    report = {**metadata,'gpu':torch.cuda.get_device_name(0) if device == 'cuda' else None,
        'loss':loss,'output_shape':shape,'label_shape':list(batch['label'].shape),
        'image_shape':list(batch['image'].shape),'parameter_max_changes':deltas,
        'optimizer_groups':[{'name':g['name'],'lr':g['lr']} for g in optimizer.param_groups],
        'reload_equal':True,'official_training_csv_unchanged':True,
        'full_epoch_executed':False,'test_set_used':False,'candidate_weight_used':not interface_only,
        'probability_min':float(actual.sigmoid().min()),'probability_max':float(actual.sigmoid().max())}
    (output/'smoke_result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('experiment_id','candidate_weight_used','loss','output_shape',
        'label_shape','parameter_max_changes','optimizer_groups','reload_equal')},ensure_ascii=False,indent=2))
    print('COMPLETE '+str(output.resolve()))
    return output


def execute(config_path: Path, device: str = 'auto') -> Path:
    from .full_train import run as first_epoch
    from .continue_train import run as continuation
    from .experiments import register
    config = load_config(config_path)
    weight_review(config)
    experiment_id = config['experiment_id']
    root = Path('outputs')/experiment_id if experiment_id in ('A1_DenseNet121_320_Study_LightAug','B2_EfficientNetB2_320_Study') else Path('outputs/runs')/experiment_id
    if root.exists() or (Path('outputs/experiments')/experiment_id).exists():
        raise FileExistsError(f'{experiment_id}已存在，拒绝覆盖；请保留原实验并建立新版本')
    root.mkdir(parents=True,exist_ok=False)
    (root/'experiment_config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    try:
        if experiment_id=='A1_DenseNet121_320_Study_LightAug':
            from .augmentation_preview import create_preview
            create_preview(config,root/'augmentation_preview')
        first = root/'initial'
        first_epoch(device,output=first,pt1_config=config)
        result = first
        if config['epochs'] > 1:
            result = root/'result'
            stop = config.get('early_stopping', {'patience':0,'min_delta':0.0})
            continuation(first,result,config['epochs']-1,device_name=device,resume_state=first/'training_state.pth',
                early_stopping_patience=stop['patience'],early_stopping_min_delta=stop['min_delta'])
        extras = {'experiment_config.json':root/'experiment_config.json',
                  'valid_image_predictions.csv':result/'valid_image_predictions.csv'}
        if config.get('sampler')=='study_batch':
            extras['study_sampling_report.json']=result/'study_sampling_report.json'
        notes='用户确认本地ImageNet初始化；官方train监督微调、valid仅评估；G0。'
        if config.get('sampler')=='study_batch':
            notes+='320分辨率+Study均衡逐图BCE组合实验；记录实际Study/图片/Step预算，不能单因素归因。'
        if experiment_id=='A1_DenseNet121_320_Study_LightAug':
            notes='A1训练轻增强；相对核对的A0仅改变增强，原始TV ImageNet重新初始化，官方train微调/valid评估，G0。'
            extras['augmentation_preview.png']=root/'augmentation_preview/preview.png'
            extras['augmentation.json']=root/'augmentation_preview/augmentation.json'
        if experiment_id=='B2_EfficientNetB2_320_Study':
            notes='B2：仅更换TV EfficientNet-B2骨干/对应ImageNet权重，FP32、无新增增强、G0；固定A0的Study/优化设置，原生Dropout及随机深度为新架构自带。'
        archive = register(result,experiment_id,notes,extra_files=extras)
        if experiment_id=='A1_DenseNet121_320_Study_LightAug':
            for name in ('best_model.pth','train_config.json','history.csv','validation_metrics.json','valid_image_predictions.csv','val_study_predictions.csv','study_sampling_report.json','augmentation.json','augmentation_preview.png'):
                with (archive/name).open('rb') as source, (root/name).open('xb') as target:
                    import shutil
                    shutil.copyfileobj(source,target)
        if experiment_id=='B2_EfficientNetB2_320_Study':
            import shutil
            for name in ('best_model.pth','train_config.json','history.csv','validation_metrics.json','valid_image_predictions.csv','val_study_predictions.csv','study_sampling_report.json'):
                with (archive/name).open('rb') as source, (root/name).open('xb') as target:
                    shutil.copyfileobj(source,target)
        (root/'run_result.json').write_text(json.dumps({'result':str(result.resolve()),'archive':str(archive.resolve())},indent=2),encoding='utf-8')
        return archive
    except Exception as exc:
        (root/'failure.json').write_text(json.dumps({'type':type(exc).__name__,'error':str(exc)},ensure_ascii=False,indent=2),encoding='utf-8')
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=Path('configs/experiments/PT1.json'))
    parser.add_argument('--device',choices=['auto','cuda','mps','cpu'],default='auto')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--smoke',action='store_true',help='许可核实后才加载候选，仅更新一批')
    mode.add_argument('--interface-smoke',action='store_true',help='仅随机初始化的真实图片接口测试，不使用候选权重')
    mode.add_argument('--execute',action='store_true',help='许可通过后完整训练；本轮不得执行')
    args = parser.parse_args()
    if args.execute:
        print(execute(args.config,args.device))
    elif args.smoke or args.interface_smoke:
        smoke(args.config,args.device,args.interface_smoke)
    else:
        print(json.dumps(plan(args.config),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
