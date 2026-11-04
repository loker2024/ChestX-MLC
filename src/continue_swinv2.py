"""用户确认的3轮追加微调：EMA12初始化、重置AdamW，不是无缝续训。"""
import argparse,copy,json,random,shutil,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .swinv2_experiment import *
from .run_swinv2 import prepare
from .early_stopping import EarlyStopping
from .train import seed_everything
from .internal_split import assert_sources_unchanged
from .analysis_common import finish_manifest
SOURCE=Path('outputs/X_SWINV2_BF16_run_20261103_094042')
SOURCE_SHA='2640baa432a77ba34f8e0d1e6133c2b552de59e041749ebee6de34315a85775b'

def continuation_settings():
    return {'stage':'extra_finetune_3_epochs','global_epochs':[13,14,15],'optimizer_reset':True,'scheduler_reset':False,'scheduler':'constant_1e-6_both_groups_no_warmup','seed_reset':42,'sampling_RNG_reset':True,'EMA_reset_from_source_EMA':True,'not_uninterrupted_resume':True}

def fixed_lr(optimizer):
    for group in optimizer.param_groups:group['lr']=1e-6
    return {group['name']:group['lr'] for group in optimizer.param_groups}

def source_check(folder,pathsfile):
    manifest=json.loads((folder/'output_manifest.json').read_text(encoding='utf-8'))['sha256']
    for name,h in manifest.items():
        if Path(name).name!=name or sha256(folder/name)!=h:raise ValueError('原实验清单不符：'+name)
    chosen=json.loads((folder/'selected_checkpoint.json').read_text(encoding='utf-8'))
    path=folder/'best_ema.pth'
    if chosen['variant']!='ema' or chosen['sha256']!=SOURCE_SHA or sha256(path)!=SOURCE_SHA:raise ValueError('不是已确认的Epoch12最佳EMA')
    model,payload=load_checkpoint(path,'cpu');c=copy.deepcopy(payload['config'])
    if payload['epoch']!=12 or payload['variant']!='ema' or c.get('amp_dtype')!='bfloat16':raise ValueError('源Checkpoint身份不符')
    license_check(c);paths,audit,ds,loader=prepare(c,pathsfile)
    for key in ('split_manifest_sha256','train_fingerprint','validation_fingerprint'):
        if audit[key]!=payload['split_audit'][key]:raise ValueError('v2划分改变：'+key)
    metrics=json.loads((folder/'validation_metrics.json').read_text(encoding='utf-8'))['ema']['metrics']
    if metrics!=payload['metrics']:raise ValueError('源EMA指标与Checkpoint不符')
    sourcepaths=[folder/k for k in manifest]+[folder/'output_manifest.json']
    hashes={str(p.resolve()):sha256(p) for p in sourcepaths}
    return model,c,audit,ds,loader,metrics,hashes

def execute(args):
    if args.output.exists():raise FileExistsError('拒绝覆盖新旧结果')
    m,c,audit,ds,loader,baseline,protected=source_check(args.source,args.paths)
    stage=continuation_settings()
    if not args.execute:
        args.output.mkdir(parents=True,exist_ok=False)
        dump(args.output/'precheck.json',{'source_sha256':SOURCE_SHA,'source_epoch':12,'source_variant':'ema','source_macro_auc':baseline['macro_auc'],'continuation':stage,'data_audit':audit,'GPU_started':False,'training_started':False});print('PRECHECK_PASS',args.output.resolve());return
    if not torch.cuda.is_available():raise RuntimeError('CUDA不可用')
    seed_everything(42);torch.set_num_threads(4);m=m.cuda();opt=optimizer_for(m,c);fixed_lr(opt);ema=EMA(m,c['ema']['decay']);amp_dtype,scaler=amp_components(c,'cuda')
    c['continuation']={**stage,'source_sha256':SOURCE_SHA,'source_epoch':12,'source_best_auc':baseline['macro_auc']}
    args.output.mkdir(parents=True,exist_ok=False);dump(args.output/'experiment_config.json',c);dump(args.output/'train_config.json',c);dump(args.output/'split_audit.json',audit);dump(args.output/'source_binding.json',{'source':str(args.source.resolve()),'sha256':SOURCE_SHA,'protected':protected,'baseline':baseline})
    # 追加阶段未超过原EMA时，保留原EMA作为最终选择，不用变差的新模型覆盖它。
    best={key:baseline['macro_auc'] for key in ('normal','ema')};best_paths={}
    for variant,model in [('normal',m),('ema',ema.model)]:
        path=args.output/('initial_'+variant+'.pth');digest=save_checkpoint(path,model,c,audit,12,variant,baseline);best_paths[variant]={'path':str(path.resolve()),'sha256':digest,'epoch':12,'metrics':baseline,'initial_copy_of_source':True}
    stops={k:EarlyStopping(3,.0001) for k in best}
    for stop in stops.values():stop.step(baseline['macro_auc'],12)
    rows=[];started=time.perf_counter()
    try:
      for epoch in (13,14,15):
        epochdir=args.output/f'epoch_{epoch}';epochdir.mkdir();lr=fixed_lr(opt);m.train();seen=set();seen_studies=set();total=0.;n=0;steps=0;skips=0;consecutive_skips=0;groups=[];torch.cuda.reset_peak_memory_stats()
        for batches in accumulation_groups(loader,32):
            for batch in batches:
                ids=set(batch['Dicom_id']);s=set(batch['Study_id'])
                if len(ids)!=len(batch['Dicom_id']) or seen&ids or seen_studies&s:raise ValueError('Study截断或重复')
                seen|=ids;seen_studies|=s
            result=accumulated_update(m,opt,ema,batches,'cuda',scaler,True,1.,amp_dtype=amp_dtype)
            consecutive_skips=0 if result['stepped'] else consecutive_skips+1
            if consecutive_skips>=8:raise FloatingPointError('连续8次跳步')
            total+=result['loss']*result['studies'];n+=result['studies'];steps+=int(result['stepped']);skips+=int(not result['stepped']);groups.append(result)
            if len(groups)==1 or len(groups)%25==0:print(json.dumps({'epoch':epoch,'update_groups':len(groups),'images':len(seen),'loss':total/n}),flush=True)
        if seen!=set(ds['train'].frame.Dicom_id) or seen_studies!=set(ds['train'].frame.Study_id) or not steps:raise ValueError('整轮覆盖或有效更新异常')
        dump(epochdir/'study_sampling_report.json',{'studies':n,'images':len(seen),'successful_updates':steps,'skips':skips,'groups':groups})
        for variant,model in [('normal',m),('ema',ema.model)]:
            metric=evaluate_model(model,ds['valid'],'cuda',epochdir,variant,c['validation_batch_size']);score=metric['macro_auc']
            if score is None:raise ValueError('AUC未定义')
            rows.append({'epoch':epoch,'variant':variant,'train_study_bce':total/n,'validation_image_bce':metric['validation_image_bce'],'macro_auc':score,'macro_f1':metric['macro_f1'],'backbone_lr':lr['backbone'],'head_lr':lr['head'],'successful_updates':steps,'AMP_skipped_updates':skips,**{'class_'+k+'_auc':v for k,v in metric['class_auc'].items()}})
            print(json.dumps({'epoch':epoch,'variant':variant,'macro_auc':score,'macro_f1':metric['macro_f1'],'delta_vs_source':score-baseline['macro_auc']}),flush=True)
            if score>best[variant]:
                path=epochdir/(variant+'_best.pth');digest=save_checkpoint(path,model,c,audit,epoch,variant,metric);best[variant]=score;best_paths[variant]={'path':str(path.resolve()),'sha256':digest,'epoch':epoch,'metrics':metric,'initial_copy_of_source':False}
            stops[variant].step(score,epoch)
        pd.DataFrame(rows).to_csv(args.output/'history.csv',index=False,encoding='utf-8')
        # 为此后真正恢复保存训练状态；当前不宣称已提供自动恢复入口。
        with (epochdir/'training_state.pth').open('xb') as f:
            torch.save({'normal':m.state_dict(),'ema':ema.model.state_dict(),'EMA_updates':ema.updates,'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate(),'numpy_rng':{'kind':np.random.get_state()[0],'keys':np.random.get_state()[1].tolist(),'pos':np.random.get_state()[2],'gauss':np.random.get_state()[3:]},'sampler_rng':loader.batch_sampler.generator.get_state(),'loader_rng':loader.generator.get_state(),'config':c,'epoch':epoch},f)
        if all(s.should_stop for s in stops.values()):break
      for variant,record in best_paths.items():shutil.copy2(record['path'],args.output/('best_'+variant+'.pth'))
      selected=max(('normal','ema'),key=lambda k:best[k]);improved=best[selected]>baseline['macro_auc']
      # 无改善时直接绑定原EMA，明确保留旧结果。
      target=args.output/('best_'+selected+'.pth') if improved else args.source/'best_ema.pth'
      dump(args.output/'selected_checkpoint.json',{'variant':selected if improved else 'ema','path':str(target.resolve()),'sha256':sha256(target),'improved_over_source':improved,'selection_data':'v2 only'})
      dump(args.output/'validation_metrics.json',best_paths);dump(args.output/'run_summary.json',{'completed_additional_epochs':epoch-12,'source_macro_auc':baseline['macro_auc'],'best_macro_auc_including_source':max(best.values()),'improved':improved,'seconds':time.perf_counter()-started,'continuation':stage,'early_stopping':{k:s.as_dict() for k,s in stops.items()}})
      assert_sources_unchanged(audit['source_hashes']);assert_sources_unchanged(protected);finish_manifest(args.output)
    except Exception as exc:dump(args.output/'failure.json',{'type':type(exc).__name__,'error':str(exc)});raise

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=SOURCE);p.add_argument('--paths',type=Path,default=Path('configs/internal_ab/paths.swinv2.local.json'));p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true');args=p.parse_args();execute(args)
if __name__=='__main__':main()
