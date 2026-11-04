"""复用逐标签ASL和Study归约，及SW01对照门禁。"""
import json
from pathlib import Path
import pandas as pd
from .asl import AsymmetricLabelLoss
from .study_training import reduce_study_label_losses
from .analysis_common import sha256
from .swinv2_official_data import audit_official, read_official_paths


def study_asl(logits, labels, indices):
    return reduce_study_label_losses(AsymmetricLabelLoss()(logits.float(), labels), indices)


def check_reference(c, paths_file, folder=None):
    if folder is None:
        return {'verified': False, 'reason': '缺少服务器SW01实际结果；check可执行，但禁止正式训练'}
    folder = Path(folder)
    old = json.loads((folder/'train_config.json').read_text(encoding='utf-8'))
    if old['loss']!='BCEWithLogitsLoss' or old.get('validation_mode')!='official_valid' or old['epochs']!=20:
        raise ValueError('SW01不是官方train/valid的20轮BCE对照，不得替换成v2或12轮结果')
    ignore = {'experiment_id','status','run_label','loss','asl','pretrained_weight_path','review_file'}
    differences = {k: [old.get(k), c.get(k)] for k in set(old)|set(c)
                   if k not in ignore and old.get(k)!=c.get(k)}
    if differences: raise ValueError('除ASL外配置不同：'+str(differences))
    _, current = audit_official(read_official_paths(paths_file))
    prior = json.loads((folder/'split_audit.json').read_text(encoding='utf-8'))
    for key in ('train_fingerprint','validation_fingerprint','statistics','official_valid_in_training'):
        if prior.get(key)!=current[key]:raise ValueError('SW01划分/数据范围不符：'+key)
    summary = json.loads((folder/'run_summary.json').read_text(encoding='utf-8'))
    history = pd.read_csv(folder/'history.csv')
    if set(summary['best'])!={'normal','ema'} or history.empty or history.epoch.max()!=summary['completed_epochs']:
        raise ValueError('SW01普通/EMA结果或完整轮次记录缺失')
    if summary['completed_epochs']<20 and not all(v['should_stop'] for v in summary['early_stopping'].values()):
        raise ValueError('SW01未正常完成或早停，不能把中断当作完整对照')
    best = {}
    for variant, record in summary['best'].items():
        path = folder/('best_'+variant+'.pth')
        if sha256(path)!=record['sha256']: raise ValueError('SW01最佳权重SHA不符：'+variant)
        row = history[(history.epoch==record['epoch']) & (history.variant==variant)]
        if len(row)!=1 or abs(float(row.iloc[0].macro_auc)-record['metrics']['macro_auc'])>1e-12:
            raise ValueError('SW01历史与最佳指标不符')
        best[variant] = {'epoch':record['epoch'], 'macro_auc':record['metrics']['macro_auc'], 'sha256':record['sha256']}
    return {'verified':True,'reference_run':str(folder.resolve()),'best':best,
            'last_epochs':history.tail(8).to_dict('records'),
            'completed_epochs':summary['completed_epochs'],'early_stopping':summary['early_stopping'],
            'stop_reason':'max_epochs' if summary['completed_epochs']==20 else 'early_stopping',
            'differences':{'loss':['BCEWithLogitsLoss','ASL']}}
