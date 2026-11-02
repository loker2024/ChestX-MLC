"""团队私有ENS02固定推理入口，不训练。"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .model import load_model
from .data import ChestXrayDataset,make_loader,normalization_for_settings
from .evaluate import evaluate
from .submission import make_submission,check_submission
from .devices import select_device
ROOT=Path(__file__).resolve().parents[1]
PROBS=[f'p{i}' for i in range(1,11)]
def average(a,b):
    keys=['Subject_id','Study_id'];m=a[keys+PROBS].merge(b[keys+PROBS],on=keys,how='outer',validate='one_to_one',indicator=True,suffixes=('_a','_b'))
    if not m['_merge'].eq('both').all():raise ValueError('患者/Study不一致')
    t=m[keys].copy();t[PROBS]=.5*m[[p+'_a' for p in PROBS]].to_numpy()+.5*m[[p+'_b' for p in PROBS]].to_numpy()
    return t.sort_values('Study_id').reset_index(drop=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--csv',type=Path,required=True);p.add_argument('--images',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda',choices=['cpu','cuda']);args=p.parse_args()
    if args.output.exists():raise FileExistsError('拒绝覆盖输出')
    cfg=json.loads((ROOT/'configs/ENS02.json').read_text(encoding='utf-8'))
    dataset=ChestXrayDataset(args.csv,args.images,labeled=True,size=384,normalization='imagenet')
    if len(dataset)!=1422 or dataset.frame.Study_id.nunique()!=859:raise ValueError('此交接入口仅限官方valid1422图/859 Study')
    device=select_device(args.device);torch.set_num_threads(4)
    tables={};metrics={}
    # 完整核对后才创建输出、执行前向。
    for name,entry in cfg['models'].items():
        path=ROOT/entry['path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:raise ValueError('权重SHA不符：'+name)
        payload=torch.load(path,map_location='cpu',weights_only=True);meta=payload['metadata']
        if payload['architecture']!='densenet121' or meta['size']!=384 or meta['completed_epoch']!=5 or normalization_for_settings(meta)!='imagenet':raise ValueError('Checkpoint配置不符')
        del payload
    args.output.mkdir(parents=True,exist_ok=False)
    for name,entry in cfg['models'].items():
        model=load_model(str(ROOT/entry['path']),device)
        loader=make_loader(dataset,16,num_workers=4,pin_memory=device=='cuda')
        table,metric=evaluate(model,loader,device,args.output/(name+'_image_predictions.csv'))
        tables[name]=table;metrics[name]=metric;table.to_csv(args.output/(name+'_study_predictions.csv'),index=False,encoding='utf-8')
        del model
        if device=='cuda':torch.cuda.empty_cache()
        print(name,metric['macro_auc'],flush=True)
    ensemble=average(tables['R1'],tables['R2'])
    from .evaluate import score_study_predictions
    ensemble,metrics['ENS02']=score_study_predictions(dataset.frame,ensemble)
    ensemble.to_csv(args.output/'ENS02_study_predictions.csv',index=False,encoding='utf-8')
    make_submission(ensemble,None).to_csv(args.output/'ENS02_official_valid_predictions.csv',index=False,encoding='utf-8')
    check=check_submission(args.output/'ENS02_official_valid_predictions.csv',dataset.frame,None,ensemble)
    if not check['passed']:raise ValueError(check)
    dataset.frame.groupby('Study_id',sort=True).first().reset_index()[['Subject_id','Study_id','Predict_class']].to_csv(args.output/'study_true_labels.csv',index=False,encoding='utf-8')
    (args.output/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
    print('COMPLETE',args.output.resolve(),metrics['ENS02']['macro_auc'],flush=True)
if __name__=='__main__':main()
