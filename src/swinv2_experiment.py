"""X-SWINV2-01独立实现；不改动既有DenseNet入口。"""
import copy,json,math,time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image,ImageOps
import torch
from torch import nn
from torchvision.models import swin_v2_t
from torchvision.transforms import RandomAffine,InterpolationMode
from torchvision.transforms import functional as TF
from .data import ChestXrayDataset,IMAGENET_MEAN,IMAGENET_STD,make_loader,preprocess
from .study_training import make_study_loader,study_balanced_bce
from .analysis_common import sha256,dump,save_csv
from .prediction_outputs import image_prediction_table
from .evaluate import study_metrics
SHA='b137f0e2d744bbe1e82ad10ddb8dc938676cf04a4332eae66e45c2048ac09711'

def load_config(path):
    c=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if c['experiment_id'] not in ('X-SWINV2-01','SW-02') or c['effective_batch_unit']!='approximately_32_images_complete_studies' or c['image_size']!=384 or c['pretrained_weight_sha256']!=SHA:raise ValueError('实验身份/累积单位冲突')
    if c['effective_batch_size']!=32 or type(c['epochs']) is not int or c['epochs'] not in (12,20) or c['head_dropout']!=.1 or c['ema']['decay']!=.999:raise ValueError('核心参数不能静默修改；仅支持原12轮及独立20轮实验')
    fixed={'backbone':'swin_v2_t','full_finetune':True,'loss':'BCEWithLogitsLoss','pos_weight':None,'sampler':'study_batch','optimizer':'AdamW','backbone_lr':2e-5,'head_lr':2e-4,'weight_decay':.03,'warmup_epochs':2,'scheduler':'cosine','minimum_lr_absolute':1e-6,'gradient_clip_norm':1.,'amp':True,'seed':42,'study_aggregation':'G0','normalization':'imagenet','native_stochastic_depth':.2,'early_stopping':{'patience':3,'min_delta':.0001},'augmentation':{'affine_probability':.5,'degrees':3,'translate':[.02,.02],'brightness_contrast_probability':.3,'brightness':[.9,1.1],'contrast':[.9,1.1],'horizontal_flip':False}}
    if c['experiment_id']=='SW-02':
        from .asl import ASL_PARAMETERS
        fixed['loss']='ASL'
        if c.get('asl')!=ASL_PARAMETERS or c['epochs']!=20 or c['validation_mode']!='official_valid':raise ValueError('SW02固定ASL参数/20轮/官方验证不符')
    if any(c.get(k)!=v for k,v in fixed.items()):raise ValueError('配置与本次预定配方不符，禁止静默忽略')
    for k in ('max_images_per_batch','studies_per_batch','validation_batch_size'):
        if type(c.get(k)) is not int or c[k]<1:raise ValueError('物理批量非法')
    if type(c.get('num_workers')) is not int or c['num_workers']<0:raise ValueError('num_workers必须为非负整数')
    if c.get('validation_mode') not in ('fixed_v2_internal','official_valid'):raise ValueError('未知validation_mode')
    if c.get('amp_init_scale')!=1024. or c.get('amp_max_consecutive_skips')!=8:raise ValueError('AMP缩放策略配置不符')
    if c.get('amp_dtype') not in ('float16','bfloat16'):raise ValueError('必须显式指定AMP精度，不能静默回退')
    return c

def license_check(c):
    review=json.loads(Path(c['review_file']).read_text(encoding='utf-8'))
    if review['status']!='approved' or review['weight_sha256']!=SHA or not review.get('competition_use') or not review.get('redistribution') or review.get('nc') or review.get('gated'):raise PermissionError('Swin具体权重许可范围待核实；不能绕过门禁')
    if not review.get('evidence'):raise PermissionError('缺少核准依据')
    for row in review['evidence']:
        if sha256(row['path'])!=row['sha256']:raise ValueError('许可依据哈希改变')
    return review

def build(c,pretrained=True):
    m=swin_v2_t(weights=None)
    if pretrained:
        license_check(c)
        if sha256(c['pretrained_weight_path'])!=SHA:raise ValueError('原始权重SHA错误')
        state=torch.load(c['pretrained_weight_path'],map_location='cpu',weights_only=True);m.load_state_dict(state,strict=True)
        if not all(torch.equal(v.cpu(),state[k].cpu()) for k,v in m.state_dict().items()):raise ValueError('预训练张量不符')
    m.head=nn.Sequential(nn.Dropout(c['head_dropout']),nn.Linear(m.head.in_features,10))
    return m

def optimizer_for(m,c):
    head=list(m.head.parameters());ids={id(p) for p in head}
    return torch.optim.AdamW([{'params':[p for p in m.parameters() if id(p) not in ids],'name':'backbone','lr':c['backbone_lr']},{'params':head,'name':'head','lr':c['head_lr']}],weight_decay=c['weight_decay'])

def rates(c,epoch):
    if epoch<=2:return {n:c[n+'_lr']*(.1+.9*(epoch-1)) for n in ('backbone','head')}
    f=(1+math.cos(math.pi*(epoch-2)/(c['epochs']-2)))/2
    return {n:c['minimum_lr_absolute']+(c[n+'_lr']-c['minimum_lr_absolute'])*f for n in ('backbone','head')}

class EMA:
    def __init__(self,model,decay=.999):
        self.model=copy.deepcopy(model).eval();self.model.requires_grad_(False);self.decay=decay;self.updates=0
    @torch.no_grad()
    def update(self,model):
        source=model.state_dict()
        for key,value in self.model.state_dict().items():
            if value.is_floating_point():value.mul_(self.decay).add_(source[key],alpha=1-self.decay)
            else:value.copy_(source[key])
        self.updates+=1

class SwinDataset(ChestXrayDataset):
    """只覆盖在线训练像素变换，ID/标签/Study索引沿用旧Dataset。"""
    def __init__(self,*args,augment=False,**kwargs):super().__init__(*args,**kwargs);self.x_augment=augment
    def __getitem__(self,index):
        if not self.x_augment:return super().__getitem__(index)
        row=self.frame.iloc[index]
        try:
            with Image.open(self.paths[index]) as raw:im=ImageOps.contain(raw.convert('L'),(384,384),Image.Resampling.BILINEAR)
            canvas=Image.new('L',(384,384),0);canvas.paste(im,((384-im.width)//2,(384-im.height)//2))
            if torch.rand(()).item()<.5:
                angle,translation,scale,shear=RandomAffine.get_params([-3,3],[.02,.02],[1.,1.],None,[384,384])
                canvas=TF.affine(canvas,angle,translation,scale,shear,interpolation=InterpolationMode.BILINEAR,fill=0)
            if torch.rand(()).item()<.3:
                canvas=TF.adjust_brightness(canvas,float(torch.empty(()).uniform_(.9,1.1)))
                canvas=TF.adjust_contrast(canvas,float(torch.empty(()).uniform_(.9,1.1)))
            tensor=TF.pil_to_tensor(canvas.convert('RGB')).float()/255
            tensor=TF.normalize(tensor,IMAGENET_MEAN,IMAGENET_STD)
        except Exception as exc:raise RuntimeError('读取增强失败：'+row.Dicom_id) from exc
        sample={k:row[k] for k in ('Subject_id','Study_id','Dicom_id')};sample.update(image=tensor,label=torch.from_numpy(self.labels[index].copy()))
        return sample

def accumulation_groups(loader,target=32):
    pending=[];images=0
    for batch in loader:
        pending.append(batch);images+=len(batch['image'])
        if images>=target:yield pending;pending=[];images=0
    if pending:yield pending

def amp_components(c,device='cuda'):
    name=c['amp_dtype'];dtype={'float16':torch.float16,'bfloat16':torch.bfloat16}[name]
    if device=='cuda' and name=='bfloat16':
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
            raise RuntimeError('GPU不支持原生BF16，停止；不静默回退为已失败的FP16')
    scaler=torch.amp.GradScaler(device,enabled=name=='float16',init_scale=c['amp_init_scale'])
    return dtype,scaler


def accumulated_update(model,optimizer,ema,batches,device,scaler,amp=True,clip=1.,amp_dtype=torch.float16,loss_function=None):
    """各微批均值乘Study数/整次更新Study数；尾组亦按实际总数归一化。"""
    studies=[len(set(b['Study_id'])) for b in batches];total=sum(studies)
    optimizer.zero_grad(set_to_none=True);weighted_loss=0.
    for batch,n in zip(batches,studies):
        with torch.autocast(device_type=device,enabled=amp,dtype=amp_dtype):
            logits=model(batch['image'].to(device))
        # BCE和Study归约明确在autocast之外使用FP32。
        loss=(study_balanced_bce if loss_function is None else loss_function)(logits.float(),batch['label'].to(device),batch['study_index'])
        if not torch.isfinite(loss):raise FloatingPointError('Loss非有限，停止')
        scaler.scale(loss*(n/total)).backward();weighted_loss+=loss.item()*n
    scaler.unscale_(optimizer)
    enabled=scaler.is_enabled() if hasattr(scaler,'is_enabled') else True
    bad=[name for name,p in model.named_parameters() if p.grad is not None and not bool(torch.isfinite(p.grad).all())]
    finite=not bad
    norm=None
    if finite:
        value=torch.nn.utils.clip_grad_norm_(model.parameters(),clip,error_if_nonfinite=True)
        norm=float(value)
    elif not enabled:
        raise FloatingPointError('未启用缩放时梯度非有限；异常参数：'+','.join(bad[:8]))
    before=float(scaler.get_scale());scaler.step(optimizer);scaler.update();after=float(scaler.get_scale());success=after>=before
    if success and not finite:raise FloatingPointError('非有限梯度却未被Scaler跳过，停止')
    if success:ema.update(model)
    return {'loss':weighted_loss/total,'studies':total,'images':sum(len(b['image']) for b in batches),'stepped':success,
            'gradient_norm':norm,'gradients_finite':finite,'nonfinite_gradient_parameters':bad[:8],'scale_before':before,'scale_after':after,
            'skip_reason':None if success else 'AMP_overflow_scale_backoff'}

def dump(path,value):
    """先完整序列化后打开新文件，拒绝NaN/Inf且不遗留半截JSON。"""
    text=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)
    with Path(path).open('x',encoding='utf-8') as handle:handle.write(text)

def smoke_attempts(model,optimizer,ema,batches,device,scaler,max_attempts=8,update_function=None,amp_dtype=torch.float16,loss_function=None):
    update=accumulated_update if update_function is None else update_function
    history=[];initial_updates=ema.updates
    for attempt in range(1,max_attempts+1):
        if update_function is None:result=update(model,optimizer,ema,batches,device,scaler,True,1.,amp_dtype=amp_dtype,loss_function=loss_function)
        else:result=update(model,optimizer,ema,batches,device,scaler,True,1.)
        history.append({'attempt':attempt,**result})
        if result['stepped']:
            if ema.updates!=initial_updates+1:raise RuntimeError('EMA更新次数异常')
            return {'passed':True,'attempts':history,'successful_updates':1,'EMA_updates':ema.updates,'result':result}
    return {'passed':False,'attempts':history,'successful_updates':0,'EMA_updates':ema.updates,'reason':'AMP连续溢出，不能启动完整训练'}

@torch.no_grad()
def forward_logits(model,images):
    model.eval();z=model(images)
    if z.ndim!=2 or z.shape[1]!=10 or not torch.isfinite(z).all():raise ValueError('输出非法')
    return z

@torch.no_grad()
def predict_batch(model,images):
    return forward_logits(model,images).float().sigmoid()

@torch.no_grad()
def predict_one(model,path,device):
    with Image.open(path) as im:x=preprocess(im,384,'imagenet').unsqueeze(0).to(device)
    return predict_batch(model,x).cpu().numpy()[0]

@torch.no_grad()
def evaluate_model(model,dataset,device,output,prefix,batch_size=4):
    model.eval();records=[];loss_sum=0.;count=0
    for batch in make_loader(dataset,batch_size,num_workers=4):
        # 同一概率函数用于批量验证与batch=1测速；验证BCE由未裁剪logits直接计算。
        x=batch['image'].to(device);z=forward_logits(model,x)
        if z.shape!=(len(x),10) or not torch.isfinite(z).all():raise ValueError('输出非法')
        probabilities=z.float().sigmoid().cpu().numpy()
        loss_sum+=torch.nn.functional.binary_cross_entropy_with_logits(z.float(),batch['label'].to(device)).item()*len(x);count+=len(x)
        for dicom,prob in zip(batch['Dicom_id'],probabilities):records.append({'Dicom_id':dicom,**{f'p{i+1}':float(v) for i,v in enumerate(prob)}})
    raw=pd.DataFrame(records);studies,metrics=study_metrics(dataset.frame,raw);metrics['validation_image_bce']=loss_sum/count
    save_csv(image_prediction_table(dataset.frame,raw),output/(prefix+'_image_predictions.csv'));save_csv(studies,output/(prefix+'_study_predictions.csv'));dump(output/(prefix+'_metrics.json'),metrics)
    return metrics

def save_checkpoint(path,model,c,audit,epoch,variant,metric):
    with Path(path).open('xb') as f:torch.save({'architecture':'swin_v2_t','num_classes':10,'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'config':c,'split_audit':audit,'epoch':epoch,'variant':variant,'metrics':metric},f)
    return sha256(path)

def load_checkpoint(path,device):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload['architecture']!='swin_v2_t' or payload['num_classes']!=10:raise ValueError('Checkpoint身份错误')
    m=build(payload['config'],False);m.load_state_dict(payload['state_dict'],strict=True)
    return m.to(device).eval(),payload
