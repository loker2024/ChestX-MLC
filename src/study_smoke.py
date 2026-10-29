"""320 Study实验真实权重与少量批次验收，明确禁止完整Epoch。"""
import argparse
from datetime import datetime
import gc
import json
from pathlib import Path
import time
from uuid import uuid4

import numpy as np
from PIL import Image
import pandas as pd
import torch

from .data import ChestXrayDataset, make_loader, preprocess, IMAGENET_PREPROCESS
from .devices import select_device, synchronize, reset_peak, peak_memory
from .pt1 import load_config, data_settings, approved_model
from .train import seed_everything, make_optimizer, train_step, save_model
from .model import load_model, classification_head
from .pretrained import file_sha256
from .study_training import make_study_loader, sampling_summary, collate_studies
from .lr_schedule import apply_epoch_lr
from .evaluate import study_metrics


def run(config_path: Path, device_name: str='cuda', batches: int=3) -> Path:
    config=load_config(config_path)
    if config['experiment_id'] not in ('B1_DenseNet121_320_Study','B1_DenseNet121_RA_320_Study','A1_DenseNet121_320_Study_LightAug','B2_EfficientNetB2_320_Study') or not 1<=batches<=4:
        raise ValueError('仅允许320 Study配置及1到4个冒烟批次')
    seed_everything(config['seed']);torch.set_num_threads(4)
    device=select_device(device_name)
    data=data_settings(config)
    hashes={split:file_sha256(Path(data[split]['csv'])) for split in ('train','valid')}
    datasets={split:ChestXrayDataset(data[split]['csv'],data[split]['images'],size=config['image_size'],normalization='imagenet',training=split=='train',augmentation=config['augmentation'] if split=='train' else 'none') for split in ('train','valid')}
    if (len(datasets['train']),datasets['train'].frame.Study_id.nunique(),len(datasets['valid']),datasets['valid'].frame.Study_id.nunique())!=(30344,18513,1422,859):
        raise ValueError('真实冒烟要求同一官方train/valid，不能替换合成数据')
    for key in ('Subject_id','Study_id','Dicom_id'):
        if set(datasets['train'].frame[key])&set(datasets['valid'].frame[key]):
            raise ValueError('训练/验证存在交集：'+key)
    loader=make_study_loader(datasets['train'],config)
    report=sampling_summary(datasets['train'],loader)
    prefix='B2_320_Study_smoke_' if config['backbone']=='efficientnet_b2' else 'A1_LightAug_smoke_' if config['experiment_id']=='A1_DenseNet121_320_Study_LightAug' else 'B1_RA_320_Study_smoke_' if config.get('model_backend')=='timm' else 'B1_320_Study_smoke_'
    output=Path('outputs')/(prefix+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid4().hex[:8])
    output.mkdir(exist_ok=False)
    # 仅预览全部索引，证明全覆盖；不读取整轮图片或运行整轮优化。
    state=loader.batch_sampler.generator.get_state().clone()
    planned=list(loader.batch_sampler)
    flat=[index for batch in planned for index in batch]
    if sorted(flat)!=list(range(len(datasets['train']))):raise RuntimeError('预览索引覆盖错误')
    loader.batch_sampler.generator.set_state(state)
    report['planned_first_epoch_batches']=len(planned)
    report['planned_first_epoch_max_images']=max(map(len,planned))
    report['full_index_coverage_verified_without_training']=True
    model=approved_model(config).to(device)
    provenance=model.pretrained_provenance
    proof=model.pretrained_backbone_verification
    metadata={'backbone':config['backbone'],'size':320,'seed':config['seed'],'batch_size':config['batch_size'],
        'data':data,'preprocess':IMAGENET_PREPROCESS,'normalization':'imagenet','epochs':0,
        'experiment_id':config['experiment_id']+'_smoke','smoke_only':True,
        'backbone_lr':config['backbone_lr'],'head_lr':config['head_lr'],'weight_decay':config['weight_decay'],
        'learning_rate':config['backbone_lr'],'aggregation':'mean image probabilities per Study_id','threshold':.5,
        'augmentation':config['augmentation'],'device':device,'pretrained_provenance':provenance,'pretrained_backbone_verification':proof,
        'lr_schedule':config['scheduler'],**{key:config[key] for key in ('sampler','loss_reduction','studies_per_batch','max_images_per_batch','num_workers','pin_memory')}}
    optimizer=make_optimizer(model,metadata);rates=apply_epoch_lr(optimizer,metadata,1)
    if config.get('model_backend')=='timm':
        metadata.update(model_backend='timm',pretrained_model=config['pretrained_model'],pretrained_recipe='RA',
            pretrained_variant=config['pretrained_variant'],model_backend_version=model.cxr_backend_version)
    before_backbone=next(model.parameters()).detach().clone();before_head=classification_head(model).weight.detach().clone()
    reset_peak(device);timings=[];shapes=[]
    print(f'SMOKE backend={getattr(model,"cxr_backend","torchvision")} workers={config["num_workers"]} batches={batches}; starting DataLoader',flush=True)
    start=time.perf_counter();iterator=iter(loader)
    for step in range(batches):
        batch=next(iterator);synchronize(device);loaded=time.perf_counter()
        images=batch['image'].to(device,non_blocking=True);labels=batch['label'].to(device,non_blocking=True)
        if images.shape[1:]!=(3,320,320):raise RuntimeError('真实训练输入不是320')
        loss,shape=train_step(model,images,labels,optimizer,study_index=batch['study_index'])
        synchronize(device);end=time.perf_counter()
        shapes.append(list(images.shape))
        timings.append({'batch':step+1,'studies':len(set(batch['Study_id'])),'images':len(images),'loss':loss,
            'data_wait_seconds':loaded-start,'transfer_train_seconds':end-loaded,
            'images_per_compute_second':len(images)/(end-loaded),'output_shape':shape})
        print(f'SMOKE step={step+1} images={len(images)} loss={loss:.6f} data_wait={loaded-start:.3f}s compute={end-loaded:.3f}s',flush=True)
        start=time.perf_counter()
    del iterator
    # 单独核验实际配置允许的最大图片批次，不用验证或测试信息选择。
    worst=max(planned,key=len)
    worst_batch=collate_studies([datasets['train'][i] for i in worst])
    loss,shape=train_step(model,worst_batch['image'].to(device),worst_batch['label'].to(device),optimizer,study_index=worst_batch['study_index'])
    synchronize(device)
    changes={'backbone':float((before_backbone-next(model.parameters()).detach()).abs().max()),
        'head':float((before_head-classification_head(model).weight.detach()).abs().max())}
    if min(changes.values())<=0:raise RuntimeError('骨干或分类头未更新')
    model.eval()
    with torch.inference_mode():expected=model(images).cpu()
    save_model(model,output/'model.pth',metadata)
    restored=load_model(str(output/'model.pth'),device)
    with torch.inference_mode():actual=restored(images).cpu()
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    memory=peak_memory(device)
    del optimizer,restored,batch,worst_batch,images,labels
    gc.collect()
    if device=='cuda':torch.cuda.empty_cache()
    valid_loader=make_loader(datasets['valid'],config['batch_size'],num_workers=config['num_workers'],pin_memory=config['pin_memory'])
    valid=next(iter(valid_loader))
    if valid['image'].shape[1:]!=(3,320,320):raise RuntimeError('真实验证输入不是320')
    with torch.inference_mode():probs=model(valid['image'].to(device)).sigmoid().cpu().numpy()
    ids=set(valid['Dicom_id'])
    frame=datasets['valid'].frame
    complete=[study for study,group in frame.groupby('Study_id') if set(group.Dicom_id).issubset(ids)]
    subset=frame[frame.Study_id.isin(complete)]
    predictions=pd.DataFrame([{'Dicom_id':identifier,**{f'p{i+1}':float(value) for i,value in enumerate(row)}} for identifier,row in zip(valid['Dicom_id'],probs)])
    _,metrics=study_metrics(subset,predictions[predictions.Dicom_id.isin(subset.Dicom_id)])
    fixed=datasets['train'][0]['image'].unsqueeze(0).to(device)
    forward,whole=[],[]
    with torch.inference_mode():
        for _ in range(5):model(fixed).sigmoid()
        for i in range(10):
            synchronize(device);start=time.perf_counter();model(fixed).sigmoid();synchronize(device)
            forward.append((time.perf_counter()-start)*1000)
            start=time.perf_counter()
            with Image.open(datasets['train'].paths[0]) as image:one=preprocess(image,320,'imagenet').unsqueeze(0).to(device)
            probability=model(one).sigmoid().cpu().tolist()[0]
            with (output/f'timing_{i:02d}.json').open('x',encoding='utf-8') as handle:json.dump(probability,handle)
            synchronize(device);whole.append((time.perf_counter()-start)*1000)
    if any(file_sha256(Path(data[split]['csv']))!=digest for split,digest in hashes.items()):raise RuntimeError('官方CSV发生变化')
    if file_sha256(Path(config['pretrained_weight_path']))!=config['pretrained_weight_sha256']:raise RuntimeError('原始权重发生变化')
    import torchvision
    result={'torchvision':str(torchvision.__version__),'amp_enabled':False,'candidate_weight_used':True,'pretrained_backbone_verification':proof,'device':device,
        'model_backend':getattr(model,'cxr_backend','torchvision'),'pretrained_provenance':provenance,
        'model_backend_version':getattr(model,'cxr_backend_version',None),
        'num_workers':loader.num_workers,'pin_memory':loader.pin_memory,
        'gpu':torch.cuda.get_device_name(0) if device=='cuda' else None,'seed':config['seed'],
        'input_shapes':shapes,'validation_input_shape':list(valid['image'].shape),
        'steps':timings,'max_image_batch_probe':{'images':len(worst),'loss':loss,'output_shape':shape},
        'parameter_changes':changes,'reload_equal':True,'learning_rates_epoch1':rates,**memory,
        'parameters':sum(p.numel() for p in model.parameters()),'checkpoint_bytes':(output/'model.pth').stat().st_size,
        'inference_batch1':{'warmup':5,'repeats':10,'forward_mean_ms':float(np.mean(forward)),
            'read_to_json_mean_ms':float(np.mean(whole)),'scope':'same train image; warm cache; no Study aggregation'},
        'validation_interface_subset':metrics,'full_validation_performed':False,
        'official_T4_tested':False,'official_train_valid_csv_unchanged':True,'source_weight_unchanged':True,
        'full_epoch_executed':False,'test_used':False,'sampling':report}
    for name,value in [('smoke_result.json',result),('study_sampling_report.json',report)]:
        with (output/name).open('x',encoding='utf-8') as handle:json.dump(value,handle,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    print('COMPLETE '+str(output.resolve()),flush=True)
    return output


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=Path('configs/experiments/B1_DenseNet121_320_Study.json'))
    parser.add_argument('--device',default='cuda',choices=['auto','cuda','cpu','mps'])
    parser.add_argument('--batches',type=int,default=3)
    args=parser.parse_args();run(args.config,args.device,args.batches)


if __name__=='__main__':main()
