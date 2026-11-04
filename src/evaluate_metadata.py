"""M1训练完成后离线比较A，仅使用固定v2缓存；不加载模型或访问test。"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from .analysis_common import read_predictions, save_csv, dump, provenance, finish_manifest, sha256
from .compare_internal_ab import paired_bootstrap
from .compare_view_position import load_v2
from .data import read_csv
from .evaluate import score_study_predictions
from .metadata_fusion import MetadataEncoder, FIELDS, category
from .run_metadata import validate_m1
from .submission import PROB_COLUMNS


def compare_tables(frame: pd.DataFrame, a: pd.DataFrame, m1: pd.DataFrame, output: Path, replicates: int = 1000) -> dict:
    if output.exists():raise FileExistsError('拒绝覆盖已有M1评价')
    a,am=score_study_predictions(frame,a);m1,mm=score_study_predictions(frame,m1)
    output.mkdir(parents=True,exist_ok=False)
    save_csv(pd.DataFrame([{'model':name,'macro_auc':m['macro_auc'],'macro_f1':m['macro_f1'],
        'weak_1_4_8_mean_auc':None if any(m['class_auc'][str(i)] is None for i in (1,4,8)) else float(np.mean([m['class_auc'][str(i)] for i in (1,4,8)])),
        **{f'class_{k}_auc':v for k,v in m['class_auc'].items()}} for name,m in [('A',am),('M1',mm)]]),output/'comparison.csv')
    perclass=pd.DataFrame([{'class_id':i,'A_auc':am['class_auc'][str(i)],'M1_auc':mm['class_auc'][str(i)],
        'delta':None if am['class_auc'][str(i)] is None else mm['class_auc'][str(i)]-am['class_auc'][str(i)]} for i in range(1,11)])
    save_csv(perclass,output/'class_auc.csv')
    ci=paired_bootstrap(frame,a,m1,replicates,42);ci['method']='paired Subject_id cluster percentile bootstrap; M1 minus A'
    dump(output/'bootstrap_results.json',ci)
    counts=frame.groupby('Study_id').size()
    groups={'single_image':set(counts[counts==1].index),'multi_image':set(counts[counts>1].index)}
    for field in FIELDS:
        values=frame[field].map(category)
        for value in sorted(set(values)-{''}):
            groups['contains_'+field+'_'+value]=set(frame.loc[values==value,'Study_id'])
    any_missing=frame[list(FIELDS)].apply(lambda s:s.map(category)).eq('').any(axis=1)
    missing_ids=set(frame.loc[any_missing,'Study_id'])
    groups['any_metadata_missing']=missing_ids
    groups['all_metadata_present']=set(frame.Study_id)-missing_ids
    rows=[]
    for group,ids in groups.items():
        if not ids:continue
        part=frame[frame.Study_id.isin(ids)]
        for name,table in [('A',a),('M1',m1)]:
            _,metrics=score_study_predictions(part,table[table.Study_id.isin(ids)])
            rows.append({'group':group,'model':name,'studies':len(ids),'patients':part.Subject_id.nunique(),
                'macro_auc':metrics['macro_auc'],'macro_f1':metrics['macro_f1'],
                'undefined_auc_classes':json.dumps(metrics['undefined_auc_classes']),
                **{f'class_{k}_auc':v for k,v in metrics['class_auc'].items()}})
    save_csv(pd.DataFrame(rows),output/'subgroup_metrics.csv')
    save_csv(m1,output/'M1_study_predictions.csv')
    report={'A':am,'M1':mm,'delta_macro_auc':None if am['macro_auc'] is None else mm['macro_auc']-am['macro_auc'],
        'improved_classes':int(perclass.delta.gt(0).sum()),'declined_classes':int(perclass.delta.lt(0).sum()),
        'paired_interval':ci['macro_auc_delta']['percentile_95'],'automatic_promotion':False,
        'group_definition':'Study contains view/posture; overlapping groups; all its images used in prior G0',
        'test_accessed':False,'model_inference':False,'weight_search':False}
    dump(output/'comparison.json',report)
    return report


def evaluate_saved(run: Path, reference: Path, output: Path, replicates: int = 1000) -> Path:
    if output.exists():raise FileExistsError('评价目录已存在')
    if (run/'failure.json').exists():raise ValueError('M1存在失败记录，不能当作完成实验')
    config=json.loads((run/'experiment_config.json').read_text(encoding='utf-8'));validate_m1(config,reference)
    frame,_,a,am=load_v2(reference,run/'split_snapshot')
    metrics=json.loads((run/'validation_metrics.json').read_text(encoding='utf-8'))
    settings=json.loads((run/'train_config.json').read_text(encoding='utf-8'))
    summary=json.loads((run/'run_summary.json').read_text(encoding='utf-8'))
    audit=json.loads((run/'split_audit.json').read_text(encoding='utf-8'))
    history=pd.read_csv(run/'history.csv',encoding='utf-8')
    epoch=int(history.loc[history.macro_auc.idxmax(),'epoch'])
    if epoch!=summary['best_epoch'] or epoch!=metrics['epoch'] or epoch!=settings['completed_epoch']:
        raise ValueError('M1最佳轮权重/指标/训练历史不一致')
    if (history.epoch.tolist()!=list(range(1,summary['completed_epochs']+1))
            or settings['experiment_config']!=config or settings['split_audit']!=audit
            or metrics['validation_fingerprint']!=am['validation_fingerprint']):
        raise ValueError('M1配置、轮次或数据身份不匹配')
    train_frame=read_csv(run/'split_snapshot/internal_train.csv',True)
    mapping=json.loads((run/'metadata_mapping.json').read_text(encoding='utf-8'))
    if MetadataEncoder.fit(train_frame).mapping!=mapping or settings['metadata_mapping']!=mapping:
        raise ValueError('M1映射并非仅由冻结内部训练集生成')
    if sha256(run/'best_checkpoint.pth')!=metrics['checkpoint_sha256']:
        raise ValueError('M1最佳权重SHA不匹配')
    for filename in ('valid_image_predictions.csv','val_study_predictions.csv','validation_metrics.json'):
        if sha256(run/filename)!=sha256(run/f'epochs/epoch_{epoch}'/filename):
            raise ValueError('M1根结果不是最佳轮副本')
    images=read_predictions(run/'valid_image_predictions.csv','image')
    # 两个字段都核对原始字符串（不重解释体位），ID/概率由G0接口进一步校验。
    from .aggregation import aggregate_predictions
    joined=frame[['Dicom_id',*FIELDS]].merge(images[['Dicom_id',*FIELDS]],on='Dicom_id',validate='one_to_one',suffixes=('_truth','_pred'),how='outer',indicator=True)
    if not joined['_merge'].eq('both').all() or any(not joined[k+'_truth'].map(category).equals(joined[k+'_pred'].map(category)) for k in FIELDS):
        raise ValueError('M1图片元数据与Dicom_id不匹配')
    reconstructed=aggregate_predictions(frame,images)
    m1,actual=score_study_predictions(frame,read_predictions(run/'val_study_predictions.csv','study'))
    merged=m1.merge(reconstructed,on=['Subject_id','Study_id'],validate='one_to_one',suffixes=('_saved','_new'))
    np.testing.assert_allclose(merged[[p+'_saved' for p in PROB_COLUMNS]],merged[[p+'_new' for p in PROB_COLUMNS]],rtol=0,atol=1e-12)
    for key in ('macro_auc','macro_f1'):
        np.testing.assert_allclose(actual[key],metrics[key],rtol=0,atol=1e-12)
    for i in range(1,11):np.testing.assert_allclose(actual['class_auc'][str(i)],metrics['class_auc'][str(i)],rtol=0,atol=1e-12)
    inputs={'M1_predictions':run/'val_study_predictions.csv','M1_images':run/'valid_image_predictions.csv',
        'M1_config':run/'train_config.json','M1_metrics':run/'validation_metrics.json','A_predictions':reference/'val_study_predictions.csv',
        'v2_valid':run/'split_snapshot/internal_valid.csv','metadata_mapping':run/'metadata_mapping.json'}
    origin=provenance(inputs,{'replicates':replicates,'seed':42,'checkpoint_loaded':False})
    report=compare_tables(frame,a,m1,output,replicates)
    history['validation_study_BCE_minus_online_train_BCE']=history.validation_study_balanced_bce-history.train_loss
    save_csv(history,output/'training_curve.csv')
    interval=report['paired_interval']
    epochs=[json.loads(p.read_text(encoding='utf-8')) for p in (run/'epochs').glob('epoch_*/validation_metrics.json')]
    costs={'elapsed_seconds':summary['elapsed_seconds'],
        'peak_allocated_bytes':max([r.get('peak_allocated_bytes') or 0 for r in epochs],default=0),
        'peak_reserved_bytes':max([r.get('peak_reserved_bytes') or 0 for r in epochs],default=0),
        'parameter_count':settings['parameter_count'],'official_T4_tested':False}
    dump(output/'training_cost.json',costs)
    text=(f'# M1与A内部v2对照\n\nA Macro-AUC={am["macro_auc"]:.6f}；M1={actual["macro_auc"]:.6f}；差值={report["delta_macro_auc"]:+.6f}；患者配对95%区间={interval}。\n\n'
        f'最佳Epoch={epoch}，实际{summary["completed_epochs"]}轮；训练总耗时={summary["elapsed_seconds"]:.1f}秒。\n\n'
        f'A Macro-F1={am["macro_f1"]:.6f}；M1={actual["macro_f1"]:.6f}；改善{report["improved_classes"]}类、退步{report["declined_classes"]}类。\n\n'
        f'显存峰值allocated={costs["peak_allocated_bytes"]}字节、reserved={costs["peak_reserved_bytes"]}字节；仅记录实际训练设备，不代表T4验收。\n\n'
        '十类差值见class_auc.csv，单/多图、含指定视角/体位与缺失分组见subgroup_metrics.csv。视角组可重叠，按完整Study G0评分，不丢弃其他视角。\n\n'
        'training_curve.csv同时记录在线训练Study均衡BCE与验证Study均衡BCE差值；训练为逐步更新/BN训练模式，验证是轮末eval，因此只作趋势诊断，不是同一冻结模型的精确训练-验证泛化差距。未额外完整推理训练集。\n\n'
        'M1改变了结构和新头初始化；不自动认定元数据必有收益，不搜索参数、不根据test选型、不自动集成。Bootstrap不校正最佳轮/重复留出选型偏差或训练seed变化。\n')
    (output/'experiment_report.md').write_text(text,encoding='utf-8')
    if any(sha256(item['path'])!=item['sha256'] for item in origin['inputs'].values()):raise RuntimeError('评价源文件变化')
    dump(output/'provenance.json',origin);finish_manifest(output)
    return output


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=Path('outputs/M1_metadata_fusion_v1'))
    parser.add_argument('--reference',type=Path,default=Path('outputs/Internal_AB_local_v2/A'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--bootstrap',type=int,default=1000)
    args=parser.parse_args();print('COMPLETE '+str(evaluate_saved(args.run_dir,args.reference,args.output,args.bootstrap).resolve()))


if __name__=='__main__':main()
