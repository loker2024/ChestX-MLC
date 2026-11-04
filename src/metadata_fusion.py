"""M1独立数据/模型接口：只编码训练视角与体位，不改旧图像模型入口。"""
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from .data import ChestXrayDataset, normalization_for_settings
from .model import build_model

FIELDS = ('ViewPosition', 'PatientOrientation')
FUSION_SPEC = {'kind':'concat_v1', 'fields':list(FIELDS), 'encoding':'one_hot',
               'hidden_dim':32, 'metadata_dim':16, 'image_dim':1024,
               'normalization':'strip_outer_whitespace_preserve_case', 'new_branch_lr':'head_lr'}
ARCHITECTURE = 'densenet121_metadata_concat_v1'


def category(value: object) -> str:
    return '' if pd.isna(value) else str(value).strip()


class MetadataEncoder:
    def __init__(self, mapping: dict) -> None:
        if set(mapping) != {'schema','fields','missing_id','unknown_id','known_values'}:
            raise ValueError('元数据映射字段异常')
        if mapping['schema'] != 1 or mapping['fields'] != list(FIELDS) or mapping['missing_id'] != 0 or mapping['unknown_id'] != 1:
            raise ValueError('仅支持M1固定两字段/MISSING=0/UNK=1')
        if set(mapping['known_values']) != set(FIELDS):
            raise ValueError('严禁增加ID、标签、Rows或Columns作为特征')
        for field, values in mapping['known_values'].items():
            if (not isinstance(values,dict) or len(values)>32
                    or any(not isinstance(k,str) or not k or category(k)!=k for k in values)
                    or any(type(v) is not int for v in values.values())
                    or sorted(values.values())!=list(range(2,len(values)+2))):
                raise ValueError('元数据映射高基数/非规范字符串/编码不连续：'+field)
        self.mapping=copy.deepcopy(mapping)

    @classmethod
    def fit(cls, train_frame: pd.DataFrame) -> 'MetadataEncoder':
        if train_frame.empty or not set(FIELDS).issubset(train_frame):
            raise ValueError('编码仅接受非空内部训练字段')
        known={field:{value:i+2 for i,value in enumerate(sorted(set(train_frame[field].map(category))-{''}))} for field in FIELDS}
        return cls({'schema':1,'fields':list(FIELDS),'missing_id':0,'unknown_id':1,'known_values':known})

    @property
    def cardinalities(self) -> tuple[int,int]:
        return tuple(len(self.mapping['known_values'][field])+2 for field in FIELDS)

    def encode(self, row: dict) -> torch.Tensor:
        values=[]
        for field in FIELDS:
            value=category(row.get(field,''))
            values.append(0 if not value else self.mapping['known_values'][field].get(value,1))
        return torch.tensor(values,dtype=torch.long)


class MetadataDataset(Dataset):
    """复用原图像Dataset；按其返回的Dicom_id查元数据，绝不按外部CSV行号拼接。"""
    def __init__(self, base: ChestXrayDataset, encoder: MetadataEncoder) -> None:
        self.base=base
        self.frame=base.frame
        self.labels=base.labels
        if self.frame.Dicom_id.duplicated().any():
            raise ValueError('元数据索引Dicom_id重复')
        self.encoder=MetadataEncoder(encoder.mapping)
        self.by_id={row['Dicom_id']:{k:row[k] for k in FIELDS} for row in self.frame.to_dict('records')}

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict:
        sample=self.base[index]
        row=self.by_id[sample['Dicom_id']]
        if any(category(sample[field])!=category(row[field]) for field in FIELDS):
            raise ValueError('图片返回元数据与Dicom_id索引不一致')
        sample['metadata_ids']=self.encoder.encode(row)
        return sample


class MetadataDenseNet(nn.Module):
    def __init__(self, image_model: nn.Module, encoder: MetadataEncoder) -> None:
        super().__init__()
        self.features=image_model.features
        self.encoder=MetadataEncoder(encoder.mapping)
        self.cardinalities=self.encoder.cardinalities
        self.metadata_mlp=nn.Sequential(nn.Linear(sum(self.cardinalities),32),nn.ReLU(),nn.Linear(32,16),nn.ReLU())
        self.classifier=nn.Linear(1024+16,10)
        self.pretrained_provenance=getattr(image_model,'pretrained_provenance',None)
        self.pretrained_backbone_verification=getattr(image_model,'pretrained_backbone_verification',None)

    def forward(self, images: torch.Tensor, metadata_ids: torch.Tensor) -> torch.Tensor:
        if images.ndim!=4 or images.shape[1]!=3 or metadata_ids.shape!=(len(images),2) or metadata_ids.dtype!=torch.long:
            raise ValueError('M1要求图像[B,3,H,W]及元数据long[B,2]')
        if metadata_ids.device!=images.device:
            raise ValueError('元数据与图像设备不一致')
        for i,n in enumerate(self.cardinalities):
            if (metadata_ids[:,i]<0).any() or (metadata_ids[:,i]>=n).any():
                raise ValueError('元数据编码越界，不能静默裁剪')
        features=F.adaptive_avg_pool2d(F.relu(self.features(images),inplace=False),(1,1)).flatten(1)
        if features.shape!=(len(images),1024):
            raise ValueError('DenseNet图像特征须为1024维')
        one_hot=torch.cat([F.one_hot(metadata_ids[:,i],n) for i,n in enumerate(self.cardinalities)],dim=1).to(features.dtype)
        metadata=self.metadata_mlp(one_hot)
        return self.classifier(torch.cat([features,metadata],dim=1))


def build_metadata_model(encoder: MetadataEncoder, weight_path: Path | None = None,
                         weight_sha: str | None = None, registry: str | None = None) -> MetadataDenseNet:
    image=build_model(weight_path,weight_sha,registry,backbone='densenet121') if weight_path is not None else build_model(backbone='densenet121')
    return MetadataDenseNet(image,encoder)


def metadata_optimizer(model: MetadataDenseNet, settings: dict) -> torch.optim.AdamW:
    """新MLP与融合头共用A的head_lr，避免被误分到backbone_lr。"""
    backbone=list(model.features.parameters())
    head=[*model.metadata_mlp.parameters(),*model.classifier.parameters()]
    if len({id(p) for p in backbone+head})!=len(list(model.parameters())):
        raise ValueError('M1参数分组有遗漏或重复')
    return torch.optim.AdamW([{'params':backbone,'lr':settings['backbone_lr'],'name':'backbone'},
        {'params':head,'lr':settings['head_lr'],'name':'head'}],weight_decay=settings['weight_decay'])


def save_metadata_model(model: MetadataDenseNet, path: Path, settings: dict) -> None:
    if normalization_for_settings(settings)!='imagenet' or settings.get('size')!=320:
        raise ValueError('M1必须保存320/ImageNet预处理')
    if settings.get('fusion')!=FUSION_SPEC or settings.get('metadata_mapping')!=model.encoder.mapping:
        raise ValueError('保存配置不匹配M1结构')
    with path.open('xb') as handle:
        torch.save({'schema':1,'architecture':ARCHITECTURE,'num_classes':10,'fusion':FUSION_SPEC,
            'metadata_mapping':model.encoder.mapping,'state_dict':model.state_dict(),'metadata':settings},handle)


def load_metadata_model(path: Path, device: str = 'cpu') -> tuple[MetadataDenseNet,dict]:
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    if checkpoint.get('schema')!=1 or checkpoint.get('architecture')!=ARCHITECTURE or checkpoint.get('num_classes')!=10 or checkpoint.get('fusion')!=FUSION_SPEC:
        raise ValueError('不是M1固定Concat结构Checkpoint')
    settings=checkpoint['metadata']
    if settings.get('fusion')!=FUSION_SPEC or settings.get('size')!=320:
        raise ValueError('Checkpoint结构/输入尺寸冲突')
    if normalization_for_settings(settings)!='imagenet' or settings.get('metadata_mapping')!=checkpoint['metadata_mapping']:
        raise ValueError('M1归一化或元数据映射与Checkpoint冲突')
    model=build_metadata_model(MetadataEncoder(checkpoint['metadata_mapping']))
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    return model.to(device).eval(),settings


def metadata_statistics(frame: pd.DataFrame, encoder: MetadataEncoder) -> dict:
    result={}
    for field in FIELDS:
        values=frame[field].map(category)
        result[field]={'counts':{str(k):int(v) for k,v in values.value_counts().items()},
            'missing_images':int(values.eq('').sum()),'missing_fraction':float(values.eq('').mean()),
            'nonempty_categories':int(values[values.ne('')].nunique()),
            'unknown_images':int((values.ne('')&~values.isin(encoder.mapping['known_values'][field])).sum())}
    return result
