"""M1专用单步/整轮与预测；复用原Study采样、BCE归约和G0评价。"""
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from .metadata_fusion import MetadataDenseNet

from .evaluate import study_metrics
from .prediction_outputs import image_prediction_table
from .study_training import study_balanced_bce


def metadata_step(model: MetadataDenseNet, batch: dict, optimizer: torch.optim.Optimizer, device: str) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits=model(batch['image'].to(device),batch['metadata_ids'].to(device))
    loss=study_balanced_bce(logits,batch['label'].to(device),batch['study_index'].to(device))
    if not torch.isfinite(loss):raise ValueError('M1训练Loss非有限')
    loss.backward()
    if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
        raise ValueError('M1梯度非有限')
    optimizer.step()
    if not all(torch.isfinite(p).all() for p in model.parameters()):raise ValueError('M1参数非有限')
    return float(loss.detach())


def train_metadata_epoch(model: MetadataDenseNet, optimizer: torch.optim.Optimizer, loader: DataLoader, device: str) -> dict:
    seen_images,seen_studies=set(),set()
    total=0.;peak=0;steps=0
    positive_studies=np.zeros(10,dtype=int);positive_images=np.zeros(10,dtype=int)
    started=time.perf_counter()
    for steps,batch in enumerate(loader,1):
        ids=batch['Dicom_id'];studies=set(batch['Study_id'])
        if len(set(ids))!=len(ids) or seen_images.intersection(ids) or seen_studies.intersection(studies):
            raise RuntimeError('M1采样重复图片或Study跨批')
        loss=metadata_step(model,batch,optimizer,device)
        seen_images.update(ids);seen_studies.update(studies)
        total+=loss*len(studies);peak=max(peak,len(ids))
        positive_images+=batch['label'].sum(0).numpy().astype(int)
        for i in range(len(studies)):
            positive_studies+=batch['label'][batch['study_index']==i][0].numpy().astype(int)
        if steps==1 or steps%50==0:
            print(f'metadata_train step={steps} studies={len(seen_studies)} images={len(seen_images)} loss={total/len(seen_studies):.6f}',flush=True)
    frame=loader.dataset.frame
    if seen_images!=set(frame.Dicom_id) or seen_studies!=set(frame.Study_id):
        raise RuntimeError('M1训练覆盖不完整')
    return {'train_loss':total/len(seen_studies),'training_images':len(seen_images),'training_studies':len(seen_studies),
        'optimizer_steps':steps,'max_batch_images':peak,'coverage_complete':True,
        'positive_studies_per_class':positive_studies.tolist(),'positive_images_per_class':positive_images.tolist(),
        'epoch_seconds_including_data':time.perf_counter()-started}


def predict_metadata(model: MetadataDenseNet, loader: DataLoader, device: str) -> tuple[pd.DataFrame,dict]:
    """有标签/无标签均可；映射从模型Checkpoint取得，不重新拟合。"""
    if loader.dataset.encoder.mapping!=model.encoder.mapping:
        raise ValueError('Dataset编码映射与Checkpoint不一致')
    model.eval();records=[];total_bce=0.;count=0;per_image_losses=[]
    with torch.no_grad():
        for batch in loader:
            logits=model(batch['image'].to(device),batch['metadata_ids'].to(device))
            if logits.shape!=(len(batch['image']),10) or not torch.isfinite(logits).all():
                raise ValueError('M1预测输出必须为有限[B,10]logits')
            if 'label' in batch:
                losses=torch.nn.functional.binary_cross_entropy_with_logits(logits,batch['label'].to(device),reduction='none').mean(1)
                total_bce+=float(losses.sum());count+=len(losses)
                per_image_losses.extend(zip(batch['Dicom_id'],losses.cpu().tolist()))
            for image_id,probs in zip(batch['Dicom_id'],logits.sigmoid().cpu().numpy()):
                records.append({'Dicom_id':image_id,**{f'p{i+1}':float(p) for i,p in enumerate(probs)}})
    table=image_prediction_table(loader.dataset.frame,pd.DataFrame(records))
    info={'images':len(table),'loss_definition':'image mean BCEWithLogitsLoss, same as A',
          'validation_loss':total_bce/count if count else None}
    if per_image_losses:
        losses=pd.DataFrame(per_image_losses,columns=['Dicom_id','bce'])
        joined=table[['Dicom_id','Study_id']].merge(losses,on='Dicom_id',validate='one_to_one')
        info['validation_study_balanced_bce']=float(joined.groupby('Study_id').bce.mean().mean())
    return table,info


def evaluate_metadata(model: MetadataDenseNet, loader: DataLoader, device: str) -> tuple[pd.DataFrame,pd.DataFrame,dict]:
    images,info=predict_metadata(model,loader,device)
    if info['validation_loss'] is None:raise ValueError('评价需要真实标签；test只允许预测')
    studies,metrics=study_metrics(loader.dataset.frame,images)
    metrics.update(validation_loss=info['validation_loss'],validation_images=info['images'],
        validation_study_balanced_bce=info['validation_study_balanced_bce'],
        validation_loss_definition=info['loss_definition'])
    return images,studies,metrics
