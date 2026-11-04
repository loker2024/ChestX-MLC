"""诊断/划分共用的可追踪私有输出；不访问正式提交或test。"""
import hashlib
import json
import platform
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROBS = [f'p{i}' for i in range(1, 11)]
P5_SHA = '0771ffd0932eec25e53881aee648e5a37898c81c199844e452544027cad96fb9'
TRAIN_FINGERPRINT = 'e27256661aff4e30e66c1081ba87c2b137983e777baff472127a3ae236b43220'
VALID_FINGERPRINT = '191c3054c6925fec37a3efa1f89acce7a8eb3fc04a52ec6d4d2445769b9739c9'


def sha256(path: str | Path) -> str:
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while chunk := handle.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def dump(path: Path, value: dict) -> None:
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding='utf-8', mode='x')


def read_predictions(path: Path, unit: str) -> pd.DataFrame:
    """显式按p1..p10列名重排，拒绝缺列、重复表头、错误ID及非法概率。"""
    import csv
    with path.open(encoding='utf-8-sig', newline='') as handle:
        header = next(csv.reader(handle))
    if len(header) != len(set(header)):
        raise ValueError('预测CSV表头重复')
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    keys = ['Subject_id', 'Study_id'] if unit == 'study' else ['Dicom_id']
    if not set(keys + PROBS).issubset(frame.columns) or frame.empty:
        raise ValueError('预测CSV缺少必要ID或完整p1..p10列')
    if frame[keys].apply(lambda col: col.str.strip().eq('')).any().any():
        raise ValueError('预测ID为空')
    if frame['Study_id' if unit == 'study' else 'Dicom_id'].duplicated().any():
        raise ValueError('预测ID重复')
    frame[PROBS] = frame[PROBS].apply(pd.to_numeric, errors='raise')
    values = frame[PROBS].to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('概率必须有限且在[0,1]，不能传入logits')
    return frame


def provenance(inputs: dict[str, Path], arguments: dict) -> dict:
    from datetime import datetime
    from zoneinfo import ZoneInfo
    import sklearn
    import torch
    import torchvision
    folder = Path(__file__).resolve().parent
    code = [path.name for path in sorted(folder.glob('*.py'))]
    return {
        'created_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'inputs': {name: {'path': str(path.resolve()), 'sha256': sha256(path)}
                   for name, path in inputs.items()},
        'code_sha256': {name: sha256(folder/name) for name in code if (folder/name).is_file()},
        'arguments': arguments, 'command': [sys.executable, *sys.argv],
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__,
                    'pandas': pd.__version__, 'sklearn': sklearn.__version__,
                    'torch': str(torch.__version__), 'torchvision': str(torchvision.__version__)},
        'test_accessed': False, 'training_started': False,
        'privacy': '含病例ID的输出仅限团队私有回传，不进入公共仓库或代码ZIP',
    }


def finish_manifest(output: Path) -> None:
    dump(output/'output_manifest.json', {
        'sha256': {path.name: sha256(path) for path in sorted(output.iterdir()) if path.is_file()},
        'excludes_self': True,
    })
