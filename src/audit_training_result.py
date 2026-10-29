"""只读核对训练交付：history、最佳权重、预测与指标，不修改源实验。"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import read_csv, ChestXrayDataset, make_loader, normalization_for_settings
from .evaluate import score_study_predictions, study_metrics
from .fingerprints import dataset_fingerprint
from .model import load_model
from .pretrained import file_sha256


def audit(source: Path, valid_csv: Path, image_root: Path | None = None,
          device: str = 'cpu') -> dict:
    hashes = {p.name: file_sha256(p) for p in source.iterdir() if p.is_file()}
    config = json.loads((source/'train_config.json').read_text(encoding='utf-8'))
    metrics = json.loads((source/'validation_metrics.json').read_text(encoding='utf-8'))
    history = pd.read_csv(source/'history.csv', encoding='utf-8')
    checkpoint = torch.load(source/'best_model.pth', map_location='cpu', weights_only=True)
    metadata = checkpoint['metadata']
    for name in ('size','seed','batch_size','backbone_lr','head_lr','pretrained_weight_sha256',
                 'preprocess','validation_mode'):
        if metadata.get(name) != config.get(name):
            raise ValueError('权重内嵌配置与交付配置不符：'+name)
    architecture = checkpoint['architecture']
    if architecture not in ('resnet18','densenet121') or checkpoint['num_classes'] != 10:
        raise ValueError('模型架构/输出类别不符')
    head_key,width = ('fc.weight',512) if architecture == 'resnet18' else ('classifier.weight',1024)
    if tuple(checkpoint['state_dict'][head_key].shape) != (10,width):
        raise ValueError('分类头形状不符')
    if history.epoch.duplicated().any() or not np.isfinite(history.macro_auc).all():
        raise ValueError('history轮次重复或AUC无效')
    best = history.loc[history.macro_auc.idxmax()]
    epoch = int(best.epoch)
    if metadata.get('completed_epoch', metadata.get('epochs')) != epoch or metrics['epoch'] != epoch:
        raise ValueError('最佳Checkpoint轮次与history/metrics不一致')
    for name in ('macro_auc','macro_f1','validation_loss','train_loss'):
        if not np.isclose(float(metrics[name]), float(best[name]), rtol=0, atol=1e-12):
            raise ValueError('最佳模型指标与history不一致：'+name)
    model_hash = hashes['best_model.pth']
    if metrics.get('checkpoint_sha256', model_hash) != model_hash:
        raise ValueError('指标绑定的Checkpoint SHA错误')
    frame = read_csv(valid_csv, labeled=True)
    if metrics['validation_fingerprint'] != dataset_fingerprint(frame):
        raise ValueError('实际验证标签/划分与原指标不一致')
    studies = pd.read_csv(source/'val_study_predictions.csv', dtype=str, encoding='utf-8')
    _, recomputed = score_study_predictions(frame, studies)
    images = pd.read_csv(source/'valid_image_predictions.csv', dtype=str, encoding='utf-8')
    aggregated, image_metrics = study_metrics(frame, images)
    for check in (recomputed, image_metrics):
        for name in ('macro_auc','macro_f1'):
            if not np.isclose(check[name],metrics[name],rtol=0,atol=1e-12):
                raise ValueError('保存的预测CSV与最佳指标不一致：'+name)
        for i in range(1,11):
            actual,expected = check['class_auc'][str(i)],metrics['class_auc'][str(i)]
            if actual is None or expected is None:
                if actual is not expected:
                    raise ValueError('未定义类别AUC标记不一致')
            elif not np.isclose(actual,expected,rtol=0,atol=1e-12):
                raise ValueError('类别AUC与指标不一致')
    # 不只看AUC：还要检查两份CSV的每个Study概率实际一致。
    cols = [f'p{i}' for i in range(1,11)]
    np.testing.assert_allclose(aggregated.set_index('Study_id').sort_index()[cols].to_numpy(float),
        studies.set_index('Study_id').sort_index()[cols].to_numpy(float),rtol=0,atol=1e-12)
    sample_check = {'executed':False,'full_validation_rerun':False}
    if image_root is not None:
        torch.set_num_threads(4)
        dataset = ChestXrayDataset(valid_csv,image_root,size=metadata['size'],
                                  normalization=normalization_for_settings(metadata))
        indices = np.random.default_rng(42).choice(len(dataset),size=min(8,len(dataset)),replace=False).tolist()
        loader = make_loader(torch.utils.data.Subset(dataset,indices),2,num_workers=0)
        model = load_model(str(source/'best_model.pth'),device)
        table = images.set_index('Dicom_id')
        error = 0.0
        with torch.no_grad():
            for batch in loader:
                actual = model(batch['image'].to(device)).sigmoid().cpu().numpy()
                expected = table.loc[batch['Dicom_id'],cols].to_numpy(float)
                error = max(error,float(np.abs(actual-expected).max()))
        sample_check = {'executed':True,'images':len(indices),'device':device,'max_probability_error':error,
                        'tolerance':1e-4,'within_tolerance':error <= 1e-4,
                        'status':'within_tolerance' if error <= 1e-4 else 'warning_requires_original_environment_recheck',
                        'full_validation_rerun':False}
    if any(file_sha256(source/name) != sha for name,sha in hashes.items()):
        raise RuntimeError('源实验文件发生变化')
    return {'source':str(source.resolve()),'source_sha256':hashes,'architecture':architecture,
        'num_classes':10,'checkpoint_completed_epoch':epoch,'last_history_epoch':int(history.epoch.max()),
        'metrics':metrics,'config':config,'csv_metrics_recomputed':True,
        'checkpoint_sample_check':sample_check,'source_unchanged':True,
        'optimizer_state_file_present':any(source.rglob('training_state.pth')),
        'checkpoint_identity_limit':'历史模型无指标SHA绑定；通过内嵌轮次、CSV复算及抽样前向核对，不冒充全量重新推理。'}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--valid-csv',type=Path,required=True)
    parser.add_argument('--valid-images',type=Path)
    parser.add_argument('--device',default='cpu',choices=['cpu','cuda','mps'])
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('拒绝覆盖已有审计结果')
    report = audit(args.source,args.valid_csv,args.valid_images,args.device)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as handle:
        json.dump(report,handle,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({'checkpoint_epoch':report['checkpoint_completed_epoch'],
        'macro_auc':report['metrics']['macro_auc'],'macro_f1':report['metrics']['macro_f1'],
        'sample':report['checkpoint_sample_check'],'source_unchanged':True},ensure_ascii=False))


if __name__ == '__main__':
    main()
