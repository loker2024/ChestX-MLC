"""M1独立入口，默认只显示配置；--check预检，--execute由用户明确启动训练。"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import time
import torch
import pandas as pd

from .analysis_common import dump, save_csv, sha256, provenance, finish_manifest
from .data import make_loader, IMAGENET_PREPROCESS, IMAGENET_MEAN, IMAGENET_STD
from .devices import select_device, synchronize, reset_peak, peak_memory
from .early_stopping import EarlyStopping
from .internal_ab import validate_config, prepare, state_digest
from .internal_split import read_paths, assert_sources_unchanged
from .lr_schedule import apply_epoch_lr
from .metadata_fusion import MetadataEncoder, MetadataDataset, FUSION_SPEC, build_metadata_model, metadata_optimizer, save_metadata_model, load_metadata_model, metadata_statistics
from .metadata_training import train_metadata_epoch, evaluate_metadata
from .study_training import make_study_loader, write_sampling_epoch
from .train import seed_everything

EXPERIMENT='M1_metadata_fusion_v1'
V2_SHA='5c8cc5a132454472254973f1ad3b22d800e05c0215660f6f41e4ddd1f05a6b91'


def configuration(reference: Path) -> dict:
    settings=json.loads((reference/'train_config.json').read_text(encoding='utf-8'))
    base=validate_config(settings['experiment_config'])
    if base['experiment_id']!='Internal_A_DenseNet121_320_Study':raise ValueError('M1唯一对照必须是A的BCE配方')
    config=copy.deepcopy(base)
    config.update(experiment_id=EXPERIMENT,fusion=copy.deepcopy(FUSION_SPEC))
    return config


def validate_m1(config: dict, reference: Path) -> dict:
    if config!=configuration(reference):raise ValueError('M1除结构与编号外必须逐项匹配A实际配置')
    return copy.deepcopy(config)


def prepare_m1(config: dict, paths: dict, reference: Path) -> tuple:
    validate_m1(config,reference)
    base=copy.deepcopy(config);base.pop('fusion');base['experiment_id']='Internal_A_DenseNet121_320_Study'
    datasets,audit,approval=prepare(base,paths)
    ref_audit=json.loads((reference/'split_audit.json').read_text(encoding='utf-8'))
    if audit['split_manifest_sha256']!=V2_SHA or any(audit[k]!=ref_audit[k] for k in ('split_manifest_sha256','train_fingerprint','validation_fingerprint')):
        raise ValueError('M1必须保持A同一固定v2划分')
    for s in ('train','valid'):
        if sha256(reference/'split_snapshot'/f'internal_{s}.csv')!=sha256(paths['split_directory']/f'internal_{s}.csv'):
            raise ValueError('A划分快照不一致')
    encoder=MetadataEncoder.fit(datasets['train'].frame)
    result={s:MetadataDataset(d,encoder) for s,d in datasets.items()}
    audit['source_hashes'][str((reference/'train_config.json').resolve())]=sha256(reference/'train_config.json')
    stats={s:metadata_statistics(d.frame,encoder) for s,d in datasets.items()}
    return result,encoder,audit,approval,stats


def execute(config: dict, paths: dict, reference: Path, output: Path, device_name: str) -> Path:
    if output.exists():raise FileExistsError('拒绝覆盖已有M1实验')
    datasets,encoder,audit,approval,stats=prepare_m1(config,paths,reference)
    if shutil.disk_usage('.').free<2*1024**3:raise RuntimeError('磁盘不足2GiB')
    device=select_device(device_name)
    seed_everything(config['seed']);torch.set_num_threads(4)
    model=build_metadata_model(encoder,paths['pretrained_weight_path'],config['pretrained_weight_sha256'],config['pretrained_registry']).to(device)
    if not model.pretrained_backbone_verification or not model.pretrained_backbone_verification['all_equal']:
        raise RuntimeError('原始ImageNet骨干核验失败，不允许随机初始化回退')
    if not all(p.requires_grad for p in model.parameters()):
        raise RuntimeError('M1与A都要求完整微调，不能静默冻结骨干')
    settings={**config,'experiment_config':copy.deepcopy(config),'size':320,'preprocess':IMAGENET_PREPROCESS,
        'normalization_mean':list(IMAGENET_MEAN),'normalization_std':list(IMAGENET_STD),
        'aggregation':'mean image probabilities per Study_id','threshold':.5,'device':device,
        'lr_schedule':config['scheduler'],'scheduler_unit':'epoch','data':{
            s:{'csv':str((paths['split_directory']/f'internal_{s}.csv').resolve()),'images':str(paths['image_root'].resolve())} for s in ('train','valid')},
        'metadata_mapping':encoder.mapping,'mapping_fit_source':'internal_train only, no labels',
        'pretrained_provenance':approval,'pretrained_backbone_verification':model.pretrained_backbone_verification,
        'initial_state_sha256':state_digest(model),'initial_features_sha256':state_digest(model.features),
        'initialization':'original approved ImageNet features; new fusion head, never A fine-tuned checkpoint',
        'split_audit':audit,'reference_config_sha256':sha256(reference/'train_config.json'),
        'parameter_count':sum(p.numel() for p in model.parameters()),
        'loss_definition':'per-image per-class BCE mean -> per-Study image mean -> batch Study mean',
        'runtime':{'torch':str(torch.__version__),'device':device},'resume_supported':False}
    optimizer=metadata_optimizer(model,settings)
    train=make_study_loader(datasets['train'],config)
    valid=make_loader(datasets['valid'],config['batch_size'],num_workers=config['num_workers'],seed=config['seed'],pin_memory=config['pin_memory'])
    output.mkdir(parents=True,exist_ok=False)
    origin=provenance({str(i):Path(p) for i,p in enumerate(audit['source_hashes'])},config);origin['training_started']=True
    dump(output/'provenance.json',origin);dump(output/'experiment_config.json',config)
    dump(output/'metadata_mapping.json',encoder.mapping);dump(output/'metadata_statistics.json',stats);dump(output/'split_audit.json',audit)
    snapshot=output/'split_snapshot';shutil.copytree(paths['split_directory'],snapshot)
    best=-1.;history=[];sampling=None;started=time.perf_counter();stopper=EarlyStopping(**config['early_stopping'])
    try:
        for epoch in range(1,config['epochs']+1):
            epoch_dir=output/'epochs'/f'epoch_{epoch}';epoch_dir.mkdir(parents=True)
            assert_sources_unchanged(audit['source_hashes']);lrs=apply_epoch_lr(optimizer,settings,epoch);reset_peak(device)
            t=time.perf_counter();trained=train_metadata_epoch(model,optimizer,train,device);synchronize(device)
            train_seconds=time.perf_counter()-t
            sampling=write_sampling_epoch(output,datasets['train'],train,epoch,trained,sampling)
            t=time.perf_counter();images,studies,metrics=evaluate_metadata(model,valid,device);synchronize(device)
            metrics.update(epoch=epoch,train_loss=trained['train_loss'],training_images=trained['training_images'],
                training_studies=trained['training_studies'],optimizer_steps=trained['optimizer_steps'],
                train_seconds=train_seconds,validation_seconds=time.perf_counter()-t,learning_rates=lrs,
                split_manifest_sha256=audit['split_manifest_sha256'],validation_mode='internal_subject_holdout',**peak_memory(device))
            if metrics['macro_auc'] is None:raise RuntimeError('M1完整Macro-AUC未定义，停止')
            stopper.step(metrics['macro_auc'],epoch);metrics['early_stopping']=stopper.as_dict()
            saved_settings={**settings,'completed_epoch':epoch}
            save_metadata_model(model,epoch_dir/'checkpoint.pth',saved_settings)
            restored,_=load_metadata_model(epoch_dir/'checkpoint.pth','cpu')
            if state_digest(restored)!=state_digest(model):raise RuntimeError('M1保存/重载张量不一致')
            del restored
            metrics.update(checkpoint_sha256=sha256(epoch_dir/'checkpoint.pth'),reload_state_exact=True)
            save_csv(images,epoch_dir/'valid_image_predictions.csv');save_csv(studies,epoch_dir/'val_study_predictions.csv')
            save_csv(pd.DataFrame([{'class_id':int(k),'auc':v} for k,v in metrics['class_auc'].items()]),epoch_dir/'class_auc.csv')
            dump(epoch_dir/'train_config.json',saved_settings);dump(epoch_dir/'validation_metrics.json',metrics)
            if metrics['macro_auc']>best:
                best=metrics['macro_auc'];best_epoch=epoch
                for source,dest in [('checkpoint.pth','best_checkpoint.pth'),('valid_image_predictions.csv','valid_image_predictions.csv'),
                    ('valid_image_predictions.csv','predictions.csv'),('val_study_predictions.csv','val_study_predictions.csv'),
                    ('validation_metrics.json','validation_metrics.json'),('train_config.json','train_config.json'),('class_auc.csv','class_auc.csv')]:
                    shutil.copy2(epoch_dir/source,output/dest)
            row={k:metrics[k] for k in ('epoch','train_loss','validation_loss','validation_study_balanced_bce','macro_auc','macro_f1',
                'training_images','training_studies','optimizer_steps','train_seconds','validation_seconds')}
            row.update({f'class_{k}_auc':v for k,v in metrics['class_auc'].items()});row.update({k+'_lr':v for k,v in lrs.items()})
            history.append(row);pd.DataFrame(history).to_csv(output/'history.csv',index=False,encoding='utf-8')
            print(json.dumps({'epoch':epoch,'macro_auc':metrics['macro_auc'],'best_epoch':best_epoch}),flush=True)
            if stopper.should_stop:break
        assert_sources_unchanged(audit['source_hashes'])
        dump(output/'run_summary.json',{'best_epoch':best_epoch,'best_macro_auc':best,'completed_epochs':len(history),
            'elapsed_seconds':time.perf_counter()-started,'stop_reason':'early_stopping' if stopper.should_stop else 'max_epochs',
            'initial_features_sha256':settings['initial_features_sha256'],'test_accessed':False,'official_valid_used_for_selection':False})
        finish_manifest(output)
        return output
    except Exception as exc:
        dump(output/'failure.json',{'type':type(exc).__name__,'error':str(exc),'parameters_automatically_changed':False})
        raise


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/experiments/M1_metadata_fusion_v1.json'))
    parser.add_argument('--reference',type=Path,default=Path('outputs/Internal_AB_local_v2/A'))
    parser.add_argument('--paths',type=Path,default=Path('configs/internal_ab/paths.v2.local.json'))
    parser.add_argument('--output',type=Path,default=Path('outputs/M1_metadata_fusion_v1'))
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    actions=parser.add_mutually_exclusive_group();actions.add_argument('--check',action='store_true');actions.add_argument('--execute',action='store_true')
    args=parser.parse_args();config=validate_m1(json.loads(args.config.read_text(encoding='utf-8-sig')),args.reference)
    if not args.check and not args.execute:
        print(json.dumps({'config':config,'training_started':False},ensure_ascii=False,indent=2));return
    paths=read_paths(args.paths)
    if args.check:
        datasets,encoder,audit,approval,stats=prepare_m1(config,paths,args.reference)
        sample={s:{'image_shape':list(d[0]['image'].shape),'metadata_shape':list(d[0]['metadata_ids'].shape)} for s,d in datasets.items()}
        print(json.dumps({'metadata':stats,'mapping':encoder.mapping,'sample_shapes':sample,'split':audit,
                          'weight':approval,'model_loaded':False,'training_started':False},ensure_ascii=False,indent=2));return
    result=execute(config,paths,args.reference,args.output,args.device)
    print('COMPLETE '+str(result.resolve()),flush=True)


if __name__=='__main__':main()
