"""固定C1与A的结果核验、患者配对Bootstrap和预定分组，不自动扩展实验。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis_common import dump, finish_manifest, save_csv, sha256
from .compare_internal_ab import read_run, paired_bootstrap
from .evaluate import score_study_predictions


def compare_asl(a_path: Path, c_path: Path, output: Path) -> Path:
    if output.exists():raise FileExistsError('拒绝覆盖已有ASL对照')
    a,c=read_run(a_path),read_run(c_path)
    keys={k for k in c['config'] if c['config'][k]!=a['config'].get(k)}
    if keys!={'experiment_id','loss','loss_reduction','asl'}:raise ValueError('C1和A不止Loss不同')
    for k in ('split_manifest_sha256','train_fingerprint','validation_fingerprint'):
        if a['audit'][k]!=c['audit'][k]:raise ValueError('C1和A划分/标签不一致')
    if a['summary']['initial_state_sha256']!=c['summary']['initial_state_sha256']:
        raise ValueError('初始ImageNet及随机分类头不同')
    if a['metadata']['runtime']!=c['metadata']['runtime'] or a['metadata']['data']!=c['metadata']['data']:
        raise ValueError('环境或实际数据路径不同')
    score_study_predictions(a['frame'],c['studies'])
    output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for name,run,path in [('A',a,a_path),('C1_ASL',c,c_path)]:
        m=run['metrics'];s=run['summary'];peaks=[]
        for p in (path/'epochs').glob('epoch_*/validation_metrics.json'):
            peaks.append(json.loads(p.read_text(encoding='utf-8')))
        rows.append({'experiment':name,'macro_auc':m['macro_auc'],'macro_f1':m['macro_f1'],
            'best_epoch':s['best_epoch'],'completed_epochs':s['completed_epochs'],
            'best_train_loss':m['train_loss'],'train_loss_name':'ASL' if name=='C1_ASL' else 'BCE',
            'validation_bce':m['validation_loss'],'weak_1_4_8_auc':float(np.mean([m['class_auc'][str(i)] for i in (1,4,8)])),
            'elapsed_seconds':s['elapsed_seconds'],'optimizer_steps_total':int(run['history'].optimizer_steps.sum()),
            'peak_allocated_bytes':max([x.get('peak_allocated_bytes') or 0 for x in peaks],default=0),
            'peak_reserved_bytes':max([x.get('peak_reserved_bytes') or 0 for x in peaks],default=0),
            'model_path':str((path/'best_model.pth').resolve()),'checkpoint_sha256':m['checkpoint_sha256']})
    save_csv(pd.DataFrame(rows),output/'comparison.csv')
    per_class=pd.DataFrame([{'class':i,'auc_A':a['metrics']['class_auc'][str(i)],'auc_ASL':c['metrics']['class_auc'][str(i)],
        'delta_ASL_minus_A':c['metrics']['class_auc'][str(i)]-a['metrics']['class_auc'][str(i)]} for i in range(1,11)])
    save_csv(per_class,output/'class_auc_comparison.csv')
    ci=paired_bootstrap(a['frame'],a['studies'],c['studies'],1000,42)
    ci['method']='paired Subject_id cluster percentile bootstrap; ASL minus A'
    dump(output/'paired_bootstrap.json',ci)
    frame=a['frame'];study_table=frame.groupby('Study_id').first()
    image_counts=frame.groupby('Study_id').size()
    patient_counts=study_table.groupby('Subject_id').size()
    counts=study_table.Subject_id.map(patient_counts)
    masks={'single_image':image_counts.eq(1),'multi_image':image_counts.gt(1),
        'subject_1_study':counts.eq(1),'subject_2_to_3_studies':counts.between(2,3),'subject_4plus_studies':counts.ge(4)}
    groups=[]
    for group,mask in masks.items():
        ids=set(mask[mask].index); part=frame[frame.Study_id.isin(ids)]
        for name,run in [('A',a),('C1_ASL',c)]:
            predictions=run['studies'][run['studies'].Study_id.isin(ids)]
            _,m=score_study_predictions(part,predictions)
            groups.append({'group':group,'experiment':name,'studies':part.Study_id.nunique(),'patients':part.Subject_id.nunique(),
                'macro_auc':m['macro_auc'],'macro_f1':m['macro_f1'],'undefined_auc_classes':json.dumps(m['undefined_auc_classes']),
                **{'class_'+k+'_auc':v for k,v in m['class_auc'].items()}})
    save_csv(pd.DataFrame(groups),output/'subgroup_metrics.csv')
    curves=pd.concat([a['history'].assign(experiment='A'),c['history'].assign(experiment='C1_ASL')],ignore_index=True)
    save_csv(curves,output/'history_comparison.csv')
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4))
    for name,run in [('A',a),('C1 ASL',c)]:
        ax.plot(run['history'].epoch,run['history'].macro_auc,marker='o',label=name)
    ax.set(xlabel='Epoch',ylabel='Internal Study Macro-AUC');ax.legend();ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(output/'auc_curve.png',dpi=160);plt.close(fig)
    delta=c['metrics']['macro_auc']-a['metrics']['macro_auc'];interval=ci['macro_auc_delta']['percentile_95']
    improved=int(per_class.delta_ASL_minus_A.gt(0).sum());declined=int(per_class.delta_ASL_minus_A.lt(0).sum())
    verdict=('no_benefit' if delta<=0 else 'valuable_candidate' if delta>=.005 and interval is not None
             and interval[0]>0 and declined<5 else 'uncertain')
    dump(output/'decision.json',{'delta_macro_auc_ASL_minus_A':delta,'delta_95_ci':interval,'improved_classes':improved,
        'declined_classes':declined,'verdict':verdict,'automatic_followup_search':False,'minimum_practical_delta':.005,
        'same_initial_model':True,'same_frozen_split':True,'changed_configuration_keys':sorted(keys),
        'loss_scale_note':'ASL and BCE train Loss not directly comparable; mean over labels/Study preserved, no extra rescaling',
        'validation_loss':'image-mean BCE for both; not Study-balanced training loss',
        'selection_bias_corrected':False,'training_seed_variability_estimated':False,
        'reference_sha256':sha256(a_path/'best_model.pth'),'ASL_sha256':sha256(c_path/'best_model.pth')})
    finish_manifest(output);return output
