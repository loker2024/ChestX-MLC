"""为图片预测附加元数据，不将ID、标签或元数据输入模型。"""
import numpy as np
import pandas as pd

from .data import METADATA_COLUMNS, with_metadata
from .submission import PROB_COLUMNS


def image_prediction_table(frame: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """按Dicom_id一对一回连，保持官方CSV行序并检查概率。"""
    if frame.Dicom_id.duplicated().any() or predictions.Dicom_id.duplicated().any():
        raise ValueError('图片ID重复')
    if set(frame.Dicom_id) != set(predictions.Dicom_id):
        raise ValueError('图片预测与原CSV的ID不一致')
    columns = ['Dicom_id', 'Subject_id', 'Study_id', *METADATA_COLUMNS,
               *(key + '_missing' for key in METADATA_COLUMNS)]
    metadata = with_metadata(frame)[columns]
    result = metadata.merge(predictions[['Dicom_id', *PROB_COLUMNS]],
                            on='Dicom_id', validate='one_to_one', sort=False)
    values = result[PROB_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('图片概率不合法')
    return result
