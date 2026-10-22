"""按 ID 对齐真实标签，先平均逐图概率，再计算检查级指标。"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score
import torch

from .data import parse_labels
from .fingerprints import dataset_fingerprint, EVALUATION_PROTOCOL


def study_metrics(frame: pd.DataFrame, predictions: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    columns = [f'p{i}' for i in range(1, 11)]
    if predictions['Dicom_id'].duplicated().any() or set(predictions['Dicom_id']) != set(frame['Dicom_id']):
        raise ValueError('验证预测 ID 缺失、重复或多余')
    merged = frame.merge(predictions, on='Dicom_id', validate='one_to_one')
    probabilities = merged[columns].to_numpy()
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError('预测概率无效')
    rows, truths = [], []
    for study_id, group in merged.groupby('Study_id', sort=True):
        labels = np.stack(group['Predict_class'].map(parse_labels))
        if not (labels == labels[0]).all() or group['Subject_id'].nunique() != 1:
            raise ValueError('检查标签或患者归属冲突')
        row = {'Subject_id': group['Subject_id'].iloc[0], 'Study_id': study_id}
        row.update(dict(zip(columns, group[columns].mean().to_numpy())))
        rows.append(row)
        truths.append(labels[0])
    result = pd.DataFrame(rows)
    truth = np.stack(truths)
    probs = result[columns].to_numpy()
    aucs = [float(roc_auc_score(truth[:, i], probs[:, i])) if len(np.unique(truth[:, i])) == 2 else None for i in range(10)]
    undefined = [i + 1 for i, auc in enumerate(aucs) if auc is None]
    metrics = {'study_count': len(result), 'threshold': 0.5,
               'class_auc': {str(i + 1): auc for i, auc in enumerate(aucs)},
               'undefined_auc_classes': undefined,
               'macro_auc': float(np.mean(aucs)) if not undefined else None,
               'macro_f1': float(f1_score(truth, probs >= 0.5, average='macro', zero_division=0)),
               'f1_zero_division': 0,
               'positive_studies': truth.sum(axis=0).astype(int).tolist(),
               'validation_fingerprint': dataset_fingerprint(frame),
               'evaluation_protocol': dict(EVALUATION_PROTOCOL)}
    return result, metrics


def evaluate(model: torch.nn.Module, loader, device: str) -> tuple[pd.DataFrame, dict]:
    model.eval()
    records, total_loss, count = [], 0.0, 0
    with torch.no_grad():
        for batch in loader:
            images, labels = batch['image'].to(device), batch['label'].to(device)
            logits = model(images)
            if logits.shape != labels.shape:
                raise ValueError('验证输出形状错误')
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError('验证损失非有限')
            total_loss += loss.item() * len(images)
            count += len(images)
            for image_id, probs in zip(batch['Dicom_id'], logits.sigmoid().cpu().numpy()):
                records.append({'Dicom_id': image_id, **{f'p{i+1}': float(p) for i, p in enumerate(probs)}})
    result, metrics = study_metrics(loader.dataset.frame, pd.DataFrame(records))
    metrics.update(validation_loss=total_loss / count, validation_images=count)
    return result, metrics
