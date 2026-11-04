"""A模型固定内部v2验证的T0/T1水平翻转对照；无训练、无官方valid/test入口。"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from PIL import Image
import torch

from .analysis_common import PROBS, dump, finish_manifest, provenance, save_csv, sha256
from .compare_internal_ab import read_run, paired_bootstrap
from .data import ChestXrayDataset, make_loader, normalization_for_settings, preprocess
from .devices import select_device, synchronize
from .evaluate import study_metrics
from .internal_ab import state_digest
from .model import load_model
from .prediction_outputs import image_prediction_table
from .train import seed_everything

A_SHA = 'cc98accc86f38b396b5900efc0500a01b0c9ee24be25557906082802efdfff91'
LABELS = ['Enlarged_Cardiomediastinum','Pneumothorax','Consolidation','Pneumonia','Edema',
          'Cardiomegaly','Atelectasis','Lung_Opacity','Pleural_Effusion','No_Finding']


def probabilities(model: torch.nn.Module, images: torch.Tensor, flipped: bool = False) -> torch.Tensor:
    logits = model(images.flip(-1) if flipped else images)
    if logits.shape != (len(images), 10) or not torch.isfinite(logits).all():
        raise ValueError('输出必须为有限[B,10]logits')
    return logits.sigmoid()


def predict_pair(model: torch.nn.Module, loader, device: str) -> tuple[dict, dict]:
    model.eval(); records = {'T0': [], 'T1': []}
    base_seconds = flip_seconds = wait_seconds = 0.; seen = 0
    wall_start = previous = time.perf_counter()
    with torch.inference_mode():
        for step, batch in enumerate(loader, 1):
            wait_seconds += time.perf_counter()-previous
            synchronize(device); start = time.perf_counter()
            images = batch['image'].to(device)
            normal = probabilities(model, images)
            p0 = normal.cpu().numpy(); synchronize(device)
            base_seconds += time.perf_counter()-start
            start = time.perf_counter()
            mirrored = probabilities(model, images, True)
            p1 = ((normal+mirrored)/2).cpu().numpy(); synchronize(device)
            flip_seconds += time.perf_counter()-start
            for key, values in [('T0',p0),('T1',p1)]:
                records[key].extend({'Dicom_id': image_id, **dict(zip(PROBS,map(float,values[i])))}
                                    for i,image_id in enumerate(batch['Dicom_id']))
            seen += len(images)
            if step == 1 or step % 25 == 0 or step == len(loader):
                print(json.dumps({'step':step,'images':seen,'elapsed_seconds':time.perf_counter()-wall_start}),flush=True)
            previous = time.perf_counter()
    return {k:pd.DataFrame(v) for k,v in records.items()}, {
        'images':seen,'joint_prediction_wall_seconds':time.perf_counter()-wall_start,
        'loader_wait_seconds_including_worker_start_decode_preprocess':wait_seconds,
        'T0_transfer_forward_sigmoid_cpu_seconds':base_seconds,
        'additional_hflip_forward_mean_cpu_seconds':flip_seconds,
        'T1_compute_to_T0_ratio':(base_seconds+flip_seconds)/base_seconds,
        'T0_forwards_per_image':1,'T1_forwards_per_image':2,
        'experiment_reuses_T0_forward':True,'theoretical_extra_forward_compute_percent':100,
        'joint_wall_is_not_two_separate_pipeline_end_to_end_timings':True}


def local_benchmark(model, dataset, device: str, samples: int = 32) -> dict:
    indices = np.random.default_rng(42).choice(len(dataset),min(samples,len(dataset)),replace=False).tolist()
    times = {'T0':[],'T1':[]}
    def one(index, mode):
        synchronize(device); start=time.perf_counter()
        with Image.open(dataset.paths[index]) as image:
            tensor=preprocess(image,dataset.size,dataset.normalization).unsqueeze(0).to(device)
        normal=probabilities(model,tensor)
        result=(normal+probabilities(model,tensor,True))/2 if mode=='T1' else normal
        json.dumps(result.cpu().numpy().tolist()); synchronize(device)
        return time.perf_counter()-start
    with torch.inference_mode():
        for mode in times:
            for i in indices[:5]: one(i,mode)
        for j,index in enumerate(indices):
            for mode in (('T0','T1') if j%2==0 else ('T1','T0')):
                times[mode].append(one(index,mode))
    means={k:float(np.mean(v)*1000) for k,v in times.items()}
    return {'batch_size':1,'samples_per_method':len(indices),'warmup_images_per_method':min(5,len(indices)),
        'mean_ms':means,'T1_to_T0_ratio':means['T1']/means['T0'],
        'scope':'read/decode/preprocess/transfer/forward/sigmoid/mean/CPU/JSON serialization; no disk write',
        'sample_source':'fixed seed42 internal valid only; alternating method order; warm file cache',
        'official_T4_tested':False,'not_official_100_image_3_round_protocol':True}


def run(folder: Path, output: Path, device_name: str = 'cuda', workers: int = 4) -> Path:
    if output.exists(): raise FileExistsError('拒绝覆盖已有TTA结果')
    baseline=read_run(folder); settings=baseline['metadata']; model_path=folder/'best_model.pth'
    if sha256(model_path)!=A_SHA or settings['completed_epoch']!=5 or settings['size']!=320:
        raise ValueError('不是已指定的A第5轮最佳模型/320输入')
    snapshot=folder/'split_snapshot'; csv=snapshot/'internal_valid.csv'
    split=json.loads((snapshot/'split_config.json').read_text(encoding='utf-8'))
    if split['algorithm']!='structure_multilabel_swap_v2' or settings['validation_mode']!='internal_subject_holdout':
        raise ValueError('只能使用A已冻结的v2内部验证')
    normalization=normalization_for_settings(settings)
    dataset=ChestXrayDataset(csv,settings['data']['valid']['images'],size=320,normalization=normalization)
    if (len(dataset),dataset.frame.Study_id.nunique(),dataset.frame.Subject_id.nunique())!=(4554,2779,1659):
        raise ValueError('固定v2内部验证规模错误')
    if set(dataset.frame.Subject_id)&set(pd.read_csv(snapshot/'internal_train.csv',dtype=str).Subject_id):
        raise ValueError('训练患者混入内部验证')
    inputs={'model':model_path,'valid_csv':csv,'source_metrics':folder/'validation_metrics.json',
        'source_study_predictions':folder/'val_study_predictions.csv','source_image_predictions':folder/'valid_image_predictions.csv',
        'split_manifest':snapshot/'output_manifest.json','source_history':folder/'history.csv'}
    origin=provenance(inputs,{'workers':workers,'T0':'identity','T1':'mean(sigmoid(identity),sigmoid(hflip))',
        'size':320,'aggregation':'G0 after image TTA','seed':42,'bootstrap_replicates':1000})
    output.mkdir(parents=True,exist_ok=False)
    dump(output/'provenance.json',origin)
    dump(output/'label_review.json',{'class_mapping':{str(i+1):name for i,name in enumerate(LABELS)},
        'mapping_source':'user confirmed official mapping in this chat on 2026-10-31; original MD mapping was empty',
        'explicit_left_right_specific_classes':[],
        'risks':['horizontal flip reverses anatomical position and any embedded L/R/text markers',
                 'labels without laterality do not imply model invariance or clinically natural mirrored anatomy'],
        'marker_pixels_preserved_but_mirrored':True,'marker_masking_or_OCR_not_applied':True,
        'scope':'internal validation experiment only; not automatically approved for final inference'})
    try:
        seed_everything(settings['seed']);torch.set_num_threads(4);device=select_device(device_name)
        model=load_model(str(model_path),device);model.requires_grad_(False);model.eval()
        digest_before=state_digest(model)
        loader=make_loader(dataset,settings['batch_size'],num_workers=workers,seed=settings['seed'],pin_memory=True)
        images,timing=predict_pair(model,loader,device)
        results={};metric_rows=[]
        for mode in ('T0','T1'):
            image_table=image_prediction_table(dataset.frame,images[mode])
            studies,metrics=study_metrics(dataset.frame,images[mode])
            save_csv(image_table,output/(mode+'_image_predictions.csv'))
            save_csv(studies,output/(mode+'_study_predictions.csv'))
            dump(output/(mode+'_metrics.json'),metrics)
            results[mode]=(studies,metrics)
            metric_rows.append({'method':mode,'macro_auc':metrics['macro_auc'],'macro_f1':metrics['macro_f1'],
                **{'class_'+k+'_auc':v for k,v in metrics['class_auc'].items()}})
        # 必须先复现已有T0，不能在更换了基准后判断TTA收益。
        source=baseline['studies'].set_index('Study_id').loc[results['T0'][0].Study_id,PROBS].to_numpy(float)
        actual=results['T0'][0][PROBS].to_numpy(float)
        difference=float(np.max(np.abs(source-actual)))
        if difference>1e-4 or abs(results['T0'][1]['macro_auc']-baseline['metrics']['macro_auc'])>1e-6:
            raise RuntimeError('T0与历史预测不一致，停止TTA选型')
        keys=['Dicom_id','Subject_id','Study_id']
        reread={m:pd.read_csv(output/(m+'_image_predictions.csv'),dtype=str,keep_default_na=False) for m in results}
        if not reread['T0'][keys].equals(reread['T1'][keys]):raise RuntimeError('逐图ID错位')
        if not results['T0'][0][['Subject_id','Study_id']].equals(results['T1'][0][['Subject_id','Study_id']]):
            raise RuntimeError('Study ID/行序错位')
        for mode in results:
            assert list(reread[mode].columns[-10:])==PROBS
            rebuilt,_=study_metrics(dataset.frame,reread[mode][['Dicom_id',*PROBS]].assign(
                **{p:pd.to_numeric(reread[mode][p]) for p in PROBS}))
            np.testing.assert_allclose(rebuilt[PROBS],results[mode][0][PROBS],rtol=0,atol=1e-12)
        save_csv(pd.DataFrame(metric_rows),output/'comparison.csv')
        save_csv(pd.DataFrame([{'class':i,'name':LABELS[i-1],
            'auc_T0':results['T0'][1]['class_auc'][str(i)],'auc_T1':results['T1'][1]['class_auc'][str(i)],
            'delta_T1_minus_T0':results['T1'][1]['class_auc'][str(i)]-results['T0'][1]['class_auc'][str(i)]}
            for i in range(1,11)]),output/'class_auc_comparison.csv')
        print('BOOTSTRAP_START',flush=True)
        interval=paired_bootstrap(dataset.frame,results['T0'][0],results['T1'][0],1000,42)
        interval['method']='paired Subject_id cluster percentile bootstrap; T1 minus T0'
        dump(output/'paired_bootstrap.json',interval)
        timing['local_batch1_benchmark']=local_benchmark(model,dataset,device)
        timing.update(device=device,gpu=torch.cuda.get_device_name(0) if device=='cuda' else None,
                      model_bytes=model_path.stat().st_size,official_T4_tested=False)
        dump(output/'timing.json',timing)
        for item in origin['inputs'].values():
            if sha256(item['path'])!=item['sha256']:raise RuntimeError('源文件发生变化')
        if state_digest(model)!=digest_before:raise RuntimeError('模型参数或BN状态发生变化')
        delta=results['T1'][1]['macro_auc']-results['T0'][1]['macro_auc']
        ci=interval['macro_auc_delta']['percentile_95']
        dump(output/'checks.json',{'images':len(dataset),'studies':len(results['T0'][0]),'subjects':1659,
            'same_image_and_study_ids_order':True,'probability_columns':PROBS,'csv_roundtrip_and_G0_verified':True,
            'historical_T0_max_probability_difference':difference,'model_and_BN_unchanged':True,
            'source_files_unchanged':True,'official_valid_or_test_accessed':False,'training_started':False,
            'delta_macro_auc_T1_minus_T0':delta,'delta_95_ci':ci,
            'positive_evidence':bool(ci is not None and ci[0]>0 and delta>0),
            'automatic_deployment':False,'selection_bias_corrected':False})
        finish_manifest(output);print('COMPLETE '+str(output.resolve()),flush=True)
        return output
    except Exception as exc:
        dump(output/'failure.json',{'type':type(exc).__name__,'error':str(exc)});raise


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-run',type=Path,default=Path('outputs/Internal_AB_local_v2/A'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--num-workers',type=int,default=4)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    a=p.parse_args();run(a.model_run,a.output,a.device,a.num_workers)


if __name__=='__main__':main()
