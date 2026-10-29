"""Study组批及均衡逐图BCE；只使用训练集分组，不聚合概率后算Loss。"""
import math
import time
from pathlib import Path
import json
from collections.abc import Iterator
import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
from torch.utils.data._utils.collate import default_collate
from .data import ChestXrayDataset, parse_labels


class StudyBatchSampler(Sampler):
    def __init__(self, dataset: ChestXrayDataset, studies_per_batch: int = 8,
                 max_images_per_batch: int = 16, seed: int = 42) -> None:
        for value in (studies_per_batch,max_images_per_batch):
            if type(value) is not int or value < 1:
                raise ValueError('Study数与图片上限必须为正整数')
        if dataset.labels is None or not len(dataset):
            raise ValueError('Study训练需要非空真实标签')
        self.groups=[group.index.tolist() for _,group in dataset.frame.groupby('Study_id',sort=False)]
        if max(map(len,self.groups))>max_images_per_batch:
            raise ValueError('单个Study超过图片上限；不能截断或丢图，请显式调整配置')
        self.studies_per_batch,self.max_images_per_batch=studies_per_batch,max_images_per_batch
        self.generator=torch.Generator().manual_seed(seed)

    def _batches(self, generator):
        batch,studies=[],0
        for index in torch.randperm(len(self.groups),generator=generator).tolist():
            group=self.groups[index]
            if batch and (studies==self.studies_per_batch or len(batch)+len(group)>self.max_images_per_batch):
                yield batch
                batch,studies=[],0
            batch.extend(group);studies+=1
        if batch:yield batch

    def __iter__(self) -> Iterator[list[int]]:
        return self._batches(self.generator)

    def __len__(self) -> int:
        # 单独预览RNG，不改变真实采样状态。
        generator=torch.Generator();generator.set_state(self.generator.get_state())
        return sum(1 for _ in self._batches(generator))


def collate_studies(samples: list[dict]) -> dict:
    batch=default_collate(samples)
    mapping={study:i for i,study in enumerate(dict.fromkeys(batch['Study_id']))}
    batch['study_index']=torch.tensor([mapping[s] for s in batch['Study_id']],dtype=torch.long)
    for index in mapping.values():
        labels=batch['label'][batch['study_index']==index]
        if not torch.equal(labels,labels[0].expand_as(labels)):
            raise ValueError('同一Study标签不一致，拒绝训练')
    return batch


def study_balanced_bce(logits: torch.Tensor, labels: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if logits.ndim!=2 or logits.shape!=labels.shape or logits.shape[1]!=10:
        raise ValueError('逐图输出/标签必须为[B,10]')
    if indices.shape!=(len(logits),) or indices.dtype!=torch.long or not len(indices):
        raise ValueError('Study索引必须为每图对应的long向量')
    indices=indices.to(logits.device)
    groups=torch.unique(indices,sorted=True)
    if not torch.equal(groups,torch.arange(len(groups),device=indices.device)):
        raise ValueError('Study索引须连续从0开始')
    if not torch.isfinite(logits).all() or not torch.isfinite(labels).all() or not ((labels==0)|(labels==1)).all():
        raise ValueError('logits须有限，标签须为0/1')
    losses=torch.nn.functional.binary_cross_entropy_with_logits(logits,labels,reduction='none').mean(1)
    sums=losses.new_zeros(len(groups)).scatter_add(0,indices,losses)
    counts=torch.bincount(indices,minlength=len(groups)).to(losses.dtype)
    return (sums/counts).mean()


def make_study_loader(dataset: ChestXrayDataset, settings: dict) -> DataLoader:
    workers=settings.get('num_workers',4)
    if type(workers) is not int or workers<0 or type(settings.get('pin_memory',False)) is not bool:
        raise ValueError('num_workers/pin_memory不合法')
    sampler=StudyBatchSampler(dataset,settings['studies_per_batch'],settings['max_images_per_batch'],settings['seed'])
    return DataLoader(dataset,batch_sampler=sampler,num_workers=workers,
        generator=torch.Generator().manual_seed(settings['seed']),collate_fn=collate_studies,
        pin_memory=settings.get('pin_memory',False))


def sampling_summary(dataset: ChestXrayDataset, loader) -> dict:
    groups=dataset.frame.groupby('Study_id',sort=False)
    sizes=groups.size()
    labels=np.stack(groups.first()['Predict_class'].map(parse_labels))
    return {'training_images':len(dataset),'training_studies':len(sizes),
        'images_per_study_histogram':{str(k):int(v) for k,v in sizes.value_counts().sort_index().items()},
        'positive_studies_per_class':labels.sum(0).astype(int).tolist(),
        'positive_images_per_class':dataset.labels.sum(0).astype(int).tolist(),
        'planned_next_epoch_batches':len(loader),'old_image_batch16_steps':math.ceil(len(dataset)/16),
        'scheduler_unit':'epoch','loss_reduction':'study_balanced_image_bce',
        'max_images_per_batch':loader.batch_sampler.max_images_per_batch,
        'studies_per_batch':loader.batch_sampler.studies_per_batch,
        'batch_limit_policy':'close batch before image cap; never split Study',
        'test_or_valid_used_for_sampling':False,'epochs':[]}


def run_study_epoch(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    loader: DataLoader, device: str) -> dict:
    from .train import train_step
    image_ids,study_ids=set(),set()
    total_loss=0.;steps=0;peak_images=0
    positive_studies=np.zeros(10,dtype=int);positive_images=np.zeros(10,dtype=int)
    start=time.perf_counter()
    for steps,batch in enumerate(loader,1):
        ids=batch['Dicom_id'];studies=set(batch['Study_id'])
        if len(set(ids))!=len(ids) or image_ids.intersection(ids) or study_ids.intersection(studies):
            raise RuntimeError('Study被截断到多个批次或图片重复')
        image_ids.update(ids);study_ids.update(studies)
        indices=batch['study_index']
        loss,_=train_step(model,batch['image'].to(device,non_blocking=True),
            batch['label'].to(device,non_blocking=True),optimizer,study_index=indices)
        total_loss+=loss*len(studies);peak_images=max(peak_images,len(ids))
        positive_images+=batch['label'].sum(0).numpy().astype(int)
        for i in range(len(studies)):
            positive_studies+=batch['label'][indices==i][0].numpy().astype(int)
        if steps==1 or steps%50==0:
            print(f'study_train step={steps} studies={len(study_ids)} images={len(image_ids)} loss={total_loss/len(study_ids):.6f}',flush=True)
    frame=loader.dataset.frame
    if image_ids!=set(frame.Dicom_id) or study_ids!=set(frame.Study_id):
        raise RuntimeError('训练Study/图片覆盖不完整')
    return {'train_loss':total_loss/len(study_ids),'training_images':len(image_ids),
        'training_studies':len(study_ids),'optimizer_steps':steps,'max_batch_images':peak_images,
        'positive_studies_per_class':positive_studies.tolist(),'positive_images_per_class':positive_images.tolist(),
        'coverage_complete':True,'epoch_seconds_including_data':time.perf_counter()-start}


def write_sampling_epoch(output: Path, dataset: ChestXrayDataset, loader: DataLoader, epoch: int, result: dict,
                         prior: dict | None = None) -> dict:
    report=prior if prior is not None else sampling_summary(dataset,loader)
    report['epochs'].append({'epoch':epoch,**result})
    report['planned_next_epoch_batches']=len(loader)
    (output/'study_sampling_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report
