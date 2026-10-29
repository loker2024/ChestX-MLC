"""仅使用完整验证集逐图CSV比较G0/G1/G2；无需加载模型或读取图片。"""
import argparse
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd

from .aggregation import METHODS, method_info
from .data import read_csv
from .evaluate import study_metrics, score_study_predictions
from .experiments import sha256
from .fingerprints import dataset_fingerprint
from .submission import PROB_COLUMNS


def compare(valid_csv: Path, image_predictions: Path,
            output: Path = Path('outputs/experiments/G_aggregation_v1'),
            methods: tuple[str,...] = METHODS, expected_g0_auc: float | None = None,
            reference_studies: Path | None = None, prediction_report: Path | None = None,
            allow_nonofficial_counts: bool = False) -> Path:
    if output.exists():
        raise FileExistsError(f'拒绝覆盖已有聚合实验: {output}')
    if any(part.lower() == 'e00_baseline' for part in output.resolve().parts):
        raise ValueError('禁止将聚合输出写入E00_baseline归档')
    if not methods or set(methods)-set(METHODS):
        raise ValueError('聚合方法只能选择G0/G1/G2')
    methods = tuple(method for method in METHODS if method in {*methods,'G0'})
    if expected_g0_auc is not None and (not np.isfinite(expected_g0_auc) or not 0 <= expected_g0_auc <= 1):
        raise ValueError('参考G0 AUC不合法')
    started = time.perf_counter()
    input_paths = {'valid_csv':valid_csv,'image_predictions':image_predictions}
    if reference_studies is not None:
        input_paths['reference_studies'] = reference_studies
    if prediction_report is not None:
        input_paths['prediction_report'] = prediction_report
    hashes = {key:sha256(path) for key,path in input_paths.items()}
    frame = read_csv(valid_csv,True)
    predictions = pd.read_csv(image_predictions,dtype=str,encoding='utf-8-sig',keep_default_na=False)
    required = {'Dicom_id','Subject_id','Study_id','ViewPosition',*PROB_COLUMNS}
    if not required.issubset(predictions):
        raise ValueError(f'逐图CSV缺列: {sorted(required-set(predictions.columns))}')
    if not allow_nonofficial_counts and (len(frame),frame.Study_id.nunique()) != (1422,859):
        raise ValueError('必须使用完整官方valid.csv：1422张图片、859次Study')
    # 图片集合、归属与概率在study_metrics中严格核对，视位仅核对保存、不参与聚合。
    views = frame[['Dicom_id','ViewPosition']].merge(predictions[['Dicom_id','ViewPosition']],
        on='Dicom_id',how='outer',validate='one_to_one',suffixes=('_official','_prediction'))
    if not views.ViewPosition_official.eq(views.ViewPosition_prediction).all():
        raise ValueError('逐图CSV视位或图片ID与官方验证CSV不符')
    counts = frame.groupby('Study_id').size()
    multi_ids = set(counts[counts > 1].index)
    multi_frame = frame.loc[frame.Study_id.isin(multi_ids)].copy()
    single_ids = set(counts[counts == 1].index)
    records, tables, results = [], {}, {}
    checks = {'official_counts_checked':not allow_nonofficial_counts}
    for method in methods:
        table,metrics = study_metrics(frame,predictions,method)
        tables[method] = table
        info = method_info(predictions,method)
        subset = None
        if not multi_frame.empty:
            _,subset = score_study_predictions(multi_frame,table.loc[table.Study_id.isin(multi_ids)],
                                               metrics['evaluation_protocol'])
        results[method] = {'aggregation':info,'all_studies':metrics,'multi_image_studies':subset}
        if method == 'G0':
            if expected_g0_auc is not None and (metrics['macro_auc'] is None or abs(metrics['macro_auc']-expected_g0_auc) > 1e-10):
                raise ValueError(f'G0未复现指定历史AUC：actual={metrics["macro_auc"]} expected={expected_g0_auc}')
            checks['g0_expected_auc_matches'] = True if expected_g0_auc is not None else None
            if reference_studies is not None:
                old = pd.read_csv(reference_studies,dtype={'Subject_id':str,'Study_id':str},encoding='utf-8-sig')
                joined = table.merge(old[['Subject_id','Study_id',*PROB_COLUMNS]],on=['Subject_id','Study_id'],
                    how='outer',validate='one_to_one',indicator=True,suffixes=('_new','_old'))
                if not joined['_merge'].eq('both').all():
                    raise ValueError('历史Study预测ID不一致')
                difference = np.abs(joined[[p+'_new' for p in PROB_COLUMNS]].to_numpy(dtype=float)-
                                    joined[[p+'_old' for p in PROB_COLUMNS]].to_numpy(dtype=float))
                if not np.isfinite(difference).all() or difference.max() > 1e-12:
                    raise ValueError('G0与已有平均概率Study预测不一致')
                checks.update(g0_reference_studies_match=True,g0_max_probability_difference=float(difference.max()))
        else:
            single_new = table.loc[table.Study_id.isin(single_ids)].set_index('Study_id').sort_index()
            single_old = tables['G0'].loc[tables['G0'].Study_id.isin(single_ids)].set_index('Study_id').sort_index()
            np.testing.assert_allclose(single_new[PROB_COLUMNS],single_old[PROB_COLUMNS],rtol=0,atol=1e-15)
        base = results['G0']['all_studies']
        base_multi = results['G0']['multi_image_studies']
        row = {'method':method,'study_count':len(table),'macro_auc':metrics['macro_auc'],
               'macro_f1':metrics['macro_f1'],'threshold':0.5,
               'auc_delta_vs_G0':None if metrics['macro_auc'] is None or base['macro_auc'] is None else metrics['macro_auc']-base['macro_auc'],
               'multi_study_count':len(multi_ids),
               'multi_macro_auc':subset['macro_auc'] if subset else None,
               'multi_macro_f1':subset['macro_f1'] if subset else None,
               'multi_auc_delta_vs_G0':None if not subset or subset['macro_auc'] is None or base_multi['macro_auc'] is None else subset['macro_auc']-base_multi['macro_auc']}
        for label,auc in metrics['class_auc'].items():
            baseline = base['class_auc'][label]
            row[f'class_{label}_auc'] = auc
            row[f'class_{label}_auc_delta_vs_G0'] = None if auc is None or baseline is None else auc-baseline
        records.append(row)
    checks.update(single_image_methods_equivalent=True,ids_and_probabilities_valid=True)
    if any(sha256(path) != hashes[key] for key,path in input_paths.items()):
        raise RuntimeError('运行期间输入文件发生变化')
    report = {'created_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'image_count':len(frame),'study_count':int(frame.Study_id.nunique()),
        'single_image_study_count':len(single_ids),'multi_image_study_count':len(multi_ids),
        'study_image_count_distribution':{str(n):int(v) for n,v in counts.value_counts().sort_index().items()},
        'validation_fingerprint':dataset_fingerprint(frame),'methods':results,
        'inputs':{key:{'path':str(path.resolve()),'sha256':hashes[key]} for key,path in input_paths.items()},
        'expected_g0_auc':expected_g0_auc,'checks':checks,'threshold':0.5,
        'training_performed':False,'model_loaded':False,'test_set_used':False,
        'elapsed_seconds':time.perf_counter()-started}
    if prediction_report is not None:
        supplied = json.loads(prediction_report.read_text(encoding='utf-8'))
        report['reported_model_sha256'] = supplied.get('model_sha256')
        report['model_identity_note'] = 'report supplied by user; image CSV hash recorded separately'
    # 同盘暂存后发布；已有实验目录拒绝覆盖，不向E00或其他历史实验写文件。
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.aggregation-',dir=output.parent.resolve()) as temporary:
        stage = Path(temporary)
        for method,table in tables.items():
            path = stage/f'{method}_study_predictions.csv'
            table.to_csv(path,index=False,encoding='utf-8',mode='x')
            reread = pd.read_csv(path,dtype={'Subject_id':str,'Study_id':str},encoding='utf-8')
            if not reread[['Subject_id','Study_id']].equals(table[['Subject_id','Study_id']]):
                raise ValueError('Study预测CSV回读ID不一致')
            np.testing.assert_allclose(reread[PROB_COLUMNS],table[PROB_COLUMNS],rtol=0,atol=1e-15)
        comparison = pd.DataFrame(records)
        comparison.to_csv(stage/'aggregation_comparison.csv',index=False,encoding='utf-8',mode='x')
        reread = pd.read_csv(stage/'aggregation_comparison.csv',encoding='utf-8')
        if reread.method.tolist() != list(methods):
            raise ValueError('比较CSV回读方法不一致')
        report['checks']['all_output_csv_reread_passed'] = True
        (stage/'aggregation_metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        (stage/'experiment_notes.md').write_text(
            '# Study聚合对照\n\n同一份验证集逐图预测；未训练、未加载模型、未使用正式测试集。\n'
            f'图片{len(frame)}张，Study{frame.Study_id.nunique()}次，其中多图{len(multi_ids)}次。\n'
            'G0算术平均，G1逐类最大，G2平均logits后sigmoid。G2优先logit1..logit10，'
            '无logits时以1e-7裁剪后反变换；单图保留原概率。\n'
            '整体859次Study的Macro-AUC为主，F1固定0.5；多图子集仅辅助分析。\n'
            'AUC缺少正负样本时为null；JSON记录具体类别。对同一验证集选择方法不构成独立泛化证明。\n'
            '此目录含真实Study预测，属于本地结果，不应上传公共仓库或放入代码交接包。\n',encoding='utf-8')
        if output.exists():
            raise FileExistsError('输出目录被其他进程使用，拒绝覆盖')
        stage.rename(output.resolve())
    print(comparison[['method','macro_auc','macro_f1','auc_delta_vs_G0','multi_macro_auc','multi_auc_delta_vs_G0']].to_string(index=False))
    print('COMPLETE '+str(output.resolve()))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--valid-csv',type=Path,required=True)
    parser.add_argument('--image-predictions',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('outputs/experiments/G_aggregation_v1'))
    parser.add_argument('--methods',choices=METHODS,nargs='+',default=list(METHODS))
    parser.add_argument('--expected-g0-auc',type=float)
    parser.add_argument('--reference-studies',type=Path)
    parser.add_argument('--prediction-report',type=Path)
    parser.add_argument('--allow-nonofficial-counts',action='store_true',help='仅用于合成数据单元测试')
    args = parser.parse_args()
    compare(args.valid_csv,args.image_predictions,args.output,tuple(args.methods),args.expected_g0_auc,
            args.reference_studies,args.prediction_report,args.allow_nonofficial_counts)


if __name__ == '__main__':
    main()
