"""X-SWINV2-01：默认预检；显式train/valid/benchmark才执行对应操作。"""
import argparse,json,copy,math,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .swinv2_experiment import *
from .internal_split import read_paths,verify_split,assert_sources_unchanged
from .early_stopping import EarlyStopping
from .train import seed_everything
from .devices import synchronize
from .submission import make_submission,check_submission
from .analysis_common import finish_manifest

CONFIG=Path('configs/experiments/X_SWINV2_01.json')
def prepare(c,path):
    if c.get('validation_mode')=='official_valid':
        from .swinv2_official_data import prepare_official
        return prepare_official(c,path)
    if c.get('validation_mode')!='fixed_v2_internal':raise ValueError('未知validation_mode')
    paths=read_paths(path);_,audit=verify_split(paths)
    # 权重路径与五路径配置同源，支持Windows/Linux迁移；SHA和许可门禁不变。
    c['pretrained_weight_path']=str(paths['pretrained_weight_path'])
    if audit['split_manifest_sha256']!='5c8cc5a132454472254973f1ad3b22d800e05c0215660f6f41e4ddd1f05a6b91':raise ValueError('不是固定v2')
    ds={s:SwinDataset(paths['split_directory']/('internal_'+s+'.csv'),paths['image_root'],size=384,normalization='imagenet',augment=s=='train') for s in ('train','valid')}
    loader=make_study_loader(ds['train'],c)
    return paths,audit,ds,loader

def precheck(c,args):
    paths,audit,ds,loader=prepare(c,args.paths)
    # 不加载模型、不运行前向，源文件结构核对不代表许可通过。
    if sha256(c['pretrained_weight_path'])!=SHA:raise ValueError('SHA不符')
    state=torch.load(c['pretrained_weight_path'],map_location='cpu',weights_only=True)
    model=swin_v2_t(weights=None);model.load_state_dict(state,strict=True)
    backbone_verified=all(torch.equal(v.cpu(),state[k].cpu()) for k,v in model.state_dict().items())
    # 使用独立workers=0的小批读取，正式配置仍4；不改变正式sampler RNG。
    sample=ds['train'][0];valid=ds['valid'][0]
    batches=list(loader.batch_sampler);images=0;studies=0;updates=[]
    for indices in batches:
        images+=len(indices);studies+=ds['train'].frame.iloc[indices].Study_id.nunique()
        if images>=32:updates.append({'images':images,'studies':studies});images=0;studies=0
    if images:updates.append({'images':images,'studies':studies})
    status='approved'
    try:license_check(c)
    except PermissionError:status='pending'
    report={'pretrained_sha256':SHA,'backbone_all_tensors_equal':backbone_verified,'CPU_identity_load_only':True,'forward':False,'GPU':False,'license':status,'train_studies':ds['train'].frame.Study_id.nunique(),'valid_studies':ds['valid'].frame.Study_id.nunique(),'train_image_shape':list(sample['image'].shape),'valid_image_shape':list(valid['image'].shape),'physical_max_images':c['max_images_per_batch'],'physical_max_studies':c['studies_per_batch'],'update_groups':len(updates),'effective_image_range':[min(x['images'] for x in updates),max(x['images'] for x in updates)],'tail_update':updates[-1],'ready_for_formal_training':status=='approved','GPU_AMP_memory_tested':False,'audit':audit}
    if c['experiment_id']=='SW-02':
        report['SW01_reference']=args.reference_report
        report['ready_for_formal_training']=report['ready_for_formal_training'] and args.reference_report['verified']
    args.output.mkdir(parents=True,exist_ok=False);dump(args.output/'precheck.json',report);print(json.dumps(report,ensure_ascii=False),flush=True)

def train(c,args):
    if args.output.exists():raise FileExistsError('拒绝覆盖训练输出；当前不支持自动续训')
    license_check(c);paths,audit,ds,loader=prepare(c,args.paths)
    if not torch.cuda.is_available():raise RuntimeError('CUDA不可用')
    seed_everything(c['seed']);torch.set_num_threads(4);m=build(c).cuda();opt=optimizer_for(m,c);ema=EMA(m,c['ema']['decay']);amp_dtype,scaler=amp_components(c,'cuda')
    args.output.mkdir(parents=True,exist_ok=False);dump(args.output/'experiment_config.json',c);dump(args.output/'train_config.json',c);dump(args.output/'split_audit.json',audit)
    dump(args.output/'pretrain_identity.json',{'source_sha256':SHA,'strict_all_tensors_verified':True})
    if c['experiment_id']=='SW-02':dump(args.output/'SW01_reference_audit.json',args.reference_report)
    loss_function=None
    if c['loss']=='ASL':
        from .sw02 import study_asl
        loss_function=study_asl
    best={'normal':-1.,'ema':-1.};stops={k:EarlyStopping(c['early_stopping']['patience'],c['early_stopping']['min_delta']) for k in best};best_paths={};rows=[];started=time.perf_counter()
    try:
      for epoch in range(1,c['epochs']+1):
        epochdir=args.output/f'epoch_{epoch}';epochdir.mkdir();lr=rates(c,epoch)
        for group in opt.param_groups:group['lr']=lr[group['name']]
        m.train();seen=set();seen_studies=set();total=0.;n=0;steps=0;skips=0;consecutive_skips=0;groups=[];torch.cuda.reset_peak_memory_stats()
        for batches in accumulation_groups(loader,32):
            for batch in batches:
                ids=set(batch['Dicom_id']);s=set(batch['Study_id'])
                if len(ids)!=len(batch['Dicom_id']) or seen&ids or seen_studies&s:raise ValueError('图片重复或Study被截断')
                seen|=ids;seen_studies|=s
            result=accumulated_update(m,opt,ema,batches,'cuda',scaler,True,1.,amp_dtype=amp_dtype,loss_function=loss_function)
            consecutive_skips=0 if result['stepped'] else consecutive_skips+1
            if consecutive_skips>=c['amp_max_consecutive_skips']:raise FloatingPointError('连续8次AMP更新被跳过，停止训练，检查数值问题')
            total+=result['loss']*result['studies'];n+=result['studies'];steps+=int(result['stepped']);skips+=int(not result['stepped']);groups.append(result)
            if len(groups)==1 or len(groups)%25==0:print(json.dumps({'epoch':epoch,'update_groups':len(groups),'images':len(seen),'loss':total/n}),flush=True)
        if seen!=set(ds['train'].frame.Dicom_id) or seen_studies!=set(ds['train'].frame.Study_id):raise ValueError('Epoch覆盖不完整')
        dump(epochdir/'study_sampling_report.json',{'studies':n,'images':len(seen),'successful_updates':steps,'AMP_skipped_updates':skips,'groups':groups})
        if steps==0:raise FloatingPointError('整轮没有成功更新，停止')
        for variant,model in [('normal',m),('ema',ema.model)]:
            metric=evaluate_model(model,ds['valid'],'cuda',epochdir,variant,c['validation_batch_size']);score=metric['macro_auc']
            if score is None:raise ValueError('AUC未定义')
            rows.append({'epoch':epoch,'variant':variant,('train_study_asl' if c['loss']=='ASL' else 'train_study_bce'):total/n,'validation_image_bce':metric['validation_image_bce'],'macro_auc':score,'macro_f1':metric['macro_f1'],'backbone_lr':lr['backbone'],'head_lr':lr['head'],'successful_updates':steps,'AMP_skipped_updates':skips,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),**{'class_'+k+'_auc':v for k,v in metric['class_auc'].items()}})
            if score>best[variant]:
                path=epochdir/(variant+'_best.pth');digest=save_checkpoint(path,model,c,audit,epoch,variant,metric);best[variant]=score;best_paths[variant]={'path':str(path.resolve()),'sha256':digest,'epoch':epoch,'metrics':metric}
            stops[variant].step(score,epoch)
        # 仅本次新实验中的活动history允许更新，不修改任何历史实验。
        pd.DataFrame(rows).to_csv(args.output/'history.csv',index=False,encoding='utf-8')
        if all(s.should_stop for s in stops.values()):break
      import shutil
      for variant,record in best_paths.items():shutil.copy2(record['path'],args.output/('best_'+variant+'.pth'))
      selected=max(('normal','ema'),key=lambda k:best[k]);dump(args.output/'validation_metrics.json',best_paths)
      dump(args.output/'selected_checkpoint.json',{'variant':selected,'path':str((args.output/('best_'+selected+'.pth')).resolve()),'sha256':best_paths[selected]['sha256'],'selection_data':c['validation_mode'],'selection_rule':'highest Macro-AUC across ordinary and EMA; normal on exact tie'})
      dump(args.output/'run_summary.json',{'completed_epochs':epoch,'seconds':time.perf_counter()-started,'early_stopping':{k:s.as_dict() for k,s in stops.items()},'best':best_paths})
      assert_sources_unchanged(audit['source_hashes']);finish_manifest(args.output)
    except Exception as exc:dump(args.output/'failure.json',{'type':type(exc).__name__,'error':str(exc)});raise

def valid(args):
    if args.output.exists():raise FileExistsError('拒绝覆盖')
    config=json.loads((args.run/'train_config.json').read_text(encoding='utf-8'))
    if config.get('validation_mode')=='official_valid':
        from .swinv2_official_data import read_official_paths,audit_official
        paths=read_official_paths(args.paths);audit_official(paths)
        if args.valid_images.resolve()!=paths['valid_images']:raise ValueError('valid图片路径与配置不符')
    else:paths=read_paths(args.paths)
    selection=json.loads((args.run/'selected_checkpoint.json').read_text(encoding='utf-8'));p=Path(selection['path'])
    if sha256(p)!=selection['sha256']:raise ValueError('所选权重SHA不符')
    model,payload=load_checkpoint(p,'cuda')
    source=str(paths['official_valid_csv'])
    if sha256(paths['official_valid_csv'])!=payload['split_audit']['source_hashes'][source]:raise ValueError('官方valid来源已变化')
    ds=SwinDataset(paths['official_valid_csv'],args.valid_images,size=384,normalization='imagenet')
    if len(ds)!=1422 or ds.frame.Study_id.nunique()!=859:raise ValueError('不是官方valid')
    train_csv=paths['official_train_csv'] if config.get('validation_mode')=='official_valid' else paths['split_directory']/'internal_train.csv'
    if set(ds.frame.Subject_id)&set(pd.read_csv(train_csv,dtype=str).Subject_id):raise ValueError('患者交叉')
    args.output.mkdir(parents=True,exist_ok=False);metric=evaluate_model(model,ds,'cuda',args.output,'official_valid',4)
    studies=pd.read_csv(args.output/'official_valid_study_predictions.csv',dtype={'Subject_id':str,'Study_id':str});save_csv(make_submission(studies,None),args.output/'official_valid_predictions.csv')
    check=check_submission(args.output/'official_valid_predictions.csv',ds.frame,None,studies)
    if not check['passed']:raise ValueError(check)
    dump(args.output/'inference_report.json',{'checkpoint_sha256':selection['sha256'],'variant':payload['variant'],'epoch':payload['epoch'],'metrics':metric,'format_check':check,'test_accessed':False})

def benchmark(args):
    if args.output.exists():raise FileExistsError('拒绝覆盖')
    files=sorted(p for p in args.pictures.rglob('*') if p.suffix.lower() in ('.jpg','.png','.jpeg'));warm=sorted(p for p in args.warmup.rglob('*') if p.suffix.lower() in ('.jpg','.png','.jpeg'))
    if len(files)!=100 or len(warm)!=5 or set(files)&set(warm):raise ValueError('需要互不重叠100张及5张预热图')
    model,payload=load_checkpoint(args.checkpoint,'cuda');args.output.mkdir(parents=True,exist_ok=False)
    for p in warm:predict_one(model,p,'cuda')
    synchronize('cuda');times=[];reference={}
    for repeat in range(3):
      row=[]
      for path in files:
        synchronize('cuda');start=time.perf_counter();values=predict_one(model,path,'cuda');synchronize('cuda')
        with (args.output/f'{repeat}_{path.stem}.json').open('x',encoding='utf-8') as f:json.dump(values.tolist(),f)
        row.append(time.perf_counter()-start)
        if repeat==0:reference[str(path)]=values
        else:np.testing.assert_allclose(values,reference[str(path)],rtol=1e-5,atol=1e-6)
      times.append(row)
    mean=float(np.mean(np.mean(times,axis=0)))*1000
    dump(args.output/'timing.json',{'mean_ms':mean,'per_round_seconds':times,'GPU':torch.cuda.get_device_name(0),'precision':'FP32','checkpoint_sha256':sha256(args.checkpoint),'single_image_output_only':True,'Study_G0_not_timed':True,'official_certification':False,'note':'替换成员的单图筛选计时；尚非最终Study提交/cost.py一致性验收，不能声称整套集成满足100ms'})

def smoke(c,args):
    if args.output.exists():raise FileExistsError('拒绝覆盖')
    license_check(c);paths,audit,ds,loader=prepare(c,args.paths)
    if not torch.cuda.is_available():raise RuntimeError('CUDA不可用')
    seed_everything(42);m=build(c).cuda();opt=optimizer_for(m,c);ema=EMA(m);amp_dtype,scaler=amp_components(c,'cuda');m.train();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    batches=next(accumulation_groups(loader,32))
    args.output.mkdir(parents=True,exist_ok=False)
    try:
        loss_function=None
        if c['loss']=='ASL':
            from .sw02 import study_asl
            loss_function=study_asl
        result=smoke_attempts(m,opt,ema,batches,'cuda',scaler,c['amp_max_consecutive_skips'],amp_dtype=amp_dtype,loss_function=loss_function)
        synchronize('cuda')
        result.update(seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),GPU=torch.cuda.get_device_name(0),physical_max_images=c['max_images_per_batch'],full_epoch=False,amp_init_scale=c['amp_init_scale'],amp_dtype=c['amp_dtype'],scaler_enabled=scaler.is_enabled())
        dump(args.output/'gpu_smoke.json',result)
        if not result['passed']:raise FloatingPointError(result['reason'])
        print('GPU_SMOKE_PASS '+str(args.output.resolve()),flush=True)
    except Exception as exc:
        dump(args.output/'failure.json',{'type':type(exc).__name__,'error':str(exc)})
        raise



def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=CONFIG);p.add_argument('--paths',type=Path,default=Path('configs/internal_ab/paths.swinv2.local.json'));p.add_argument('--mode',choices=['check','smoke','train','valid','benchmark'],default='check');p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true');p.add_argument('--run',type=Path);p.add_argument('--valid-images',type=Path);p.add_argument('--checkpoint',type=Path);p.add_argument('--pictures',type=Path);p.add_argument('--warmup',type=Path);p.add_argument('--reference-run',type=Path);a=p.parse_args()
    if a.mode!='check' and not a.execute:p.error('需要用户显式--execute')
    if a.mode in ('check','smoke','train') and load_config(a.config)['experiment_id']=='SW-02':
        from .sw02 import check_reference
        a.reference_report=check_reference(load_config(a.config),a.paths,a.reference_run)
        if a.mode!='check' and not a.reference_report['verified']:p.error('必须通过--reference-run核验服务器SW01，禁止比较错误划分')
    if a.mode=='check':precheck(load_config(a.config),a)
    elif a.mode=='smoke':smoke(load_config(a.config),a)
    elif a.mode=='train':train(load_config(a.config),a)
    elif a.mode=='valid':
        if not a.run or not a.valid_images:p.error('valid需要--run及--valid-images')
        valid(a)
    else:
        if not a.checkpoint or not a.pictures or not a.warmup:p.error('benchmark需要checkpoint/pictures/warmup')
        benchmark(a)
if __name__=='__main__':main()
