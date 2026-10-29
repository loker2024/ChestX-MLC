"""官方逐图数据读取；保留检查 ID，供后续 Study 级评估使用。"""
from pathlib import Path
import re

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
from torch.utils.data import DataLoader, Dataset

METADATA_COLUMNS = ('ViewPosition', 'PatientOrientation', 'Rows', 'Columns')
LEGACY_PREPROCESS = 'RGB letterbox /255'
IMAGENET_PREPROCESS = 'RGB letterbox /255 ImageNet normalization'
IMAGENET_MEAN = (0.485,0.456,0.406)
IMAGENET_STD = (0.229,0.224,0.225)


def normalization_for_preprocess(name: str) -> str:
    if name == LEGACY_PREPROCESS:
        return 'none'
    if name == IMAGENET_PREPROCESS:
        return 'imagenet'
    raise ValueError('不支持的检查点预处理配置')


def normalization_for_settings(settings: dict) -> str:
    mode = normalization_for_preprocess(settings.get('preprocess'))
    if settings.get('normalization',mode) != mode:
        raise ValueError('归一化名称与预处理配置冲突')
    if mode == 'imagenet' and (settings.get('normalization_mean',list(IMAGENET_MEAN)) != list(IMAGENET_MEAN)
                              or settings.get('normalization_std',list(IMAGENET_STD)) != list(IMAGENET_STD)):
        raise ValueError('检查点归一化常量与已实现的ImageNet预处理不一致')
    return mode


def with_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    """保留原始字符串；空白和缺列显式标记，不猜测方向或实际尺寸。"""
    result = frame.copy()
    for key in METADATA_COLUMNS:
        if key not in result:
            result[key] = ''
        result[key] = result[key].fillna('').astype(str)
        result[key + '_missing'] = result[key].str.strip().eq('')
    return result


def parse_labels(value: str) -> np.ndarray:
    """类别 1..10 对应下标 0..9；未知标签不能作为阴性。"""
    if not isinstance(value, str) or not re.fullmatch(r"(?:10|[1-9])(?:;(?:10|[1-9]))*", value):
        raise ValueError(f"无法解释的 Predict_class: {value!r}")
    ids = [int(token) for token in value.split(';')]
    if len(ids) != len(set(ids)):
        raise ValueError(f"重复的类别: {value!r}")
    result = np.zeros(10, dtype=np.float32)
    result[np.array(ids) - 1] = 1
    return result


def read_csv(path: str | Path, labeled: bool) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, encoding='utf-8-sig', keep_default_na=False)
    required = {'Subject_id', 'Study_id', 'Dicom_id'}
    if labeled:
        required.add('Predict_class')
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError(f"CSV 为空或缺少字段: {path}")
    for column in required:
        if frame[column].str.strip().eq('').any():
            raise ValueError(f"字段存在空值: {column}")
    if frame['Dicom_id'].duplicated().any():
        raise ValueError('Dicom_id 重复，不能唯一匹配图片')
    if frame.groupby('Study_id')['Subject_id'].nunique().gt(1).any():
        raise ValueError('同一 Study_id 属于多个患者')
    if labeled:
        labels = frame['Predict_class'].map(lambda value: tuple(parse_labels(value)))
        if labels.groupby(frame['Study_id']).nunique().gt(1).any():
            raise ValueError('同一 Study_id 标签不一致，停止自动聚合')
    return with_metadata(frame)


def index_images(root: str | Path) -> dict[str, Path]:
    """递归索引实际图片，不假设患者/检查文件夹布局。"""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"图片目录不存在: {root}")
    index = {}
    for path in root.rglob('*'):
        if path.is_file() and path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
            if path.stem in index:
                raise ValueError(f"图片 ID 对应多个文件: {path.stem}")
            index[path.stem] = path
    return index


def preprocess(image: Image.Image, size: int = 224, normalization: str = 'none', augmentation='none') -> torch.Tensor:
    if size <= 0:
        raise ValueError('图片尺寸必须大于零')
    # 先保持纵横比缩放，再居中黑色补边；不做左右翻转。
    image = ImageOps.contain(image.convert('RGB'), (size, size), Image.Resampling.BILINEAR)
    canvas = Image.new('RGB', (size, size), (0, 0, 0))
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    if augmentation != 'none':
        from .augmentation import apply_light_aug
        canvas = apply_light_aug(canvas, augmentation)
    tensor = torch.from_numpy(np.array(canvas, dtype=np.float32)).permute(2, 0, 1).contiguous() / 255.0
    if normalization == 'imagenet':
        tensor = (tensor-torch.tensor(IMAGENET_MEAN)[:,None,None])/torch.tensor(IMAGENET_STD)[:,None,None]
    elif normalization != 'none':
        raise ValueError('归一化仅支持none/imagenet，不从测试集估计统计量')
    return tensor


class ChestXrayDataset(Dataset):
    def __init__(self, csv_path: str | Path, image_root: str | Path,
                 labeled: bool = True, size: int = 224, normalization: str = 'none', *, training: bool = False, augmentation='none') -> None:
        if size <= 0:
            raise ValueError('图片尺寸必须大于零')
        from .augmentation import validate_augmentation
        self.augmentation = validate_augmentation(augmentation)
        if self.augmentation != 'none' and (not training or not labeled):
            raise ValueError('随机增强仅允许显式training=True的有标签训练Dataset')
        self.frame = read_csv(csv_path, labeled)
        self.labeled, self.size = labeled, size
        if normalization not in ('none','imagenet'):
            raise ValueError('归一化模式不支持')
        self.normalization = normalization
        index = index_images(image_root)
        missing = [value for value in self.frame['Dicom_id'] if value not in index]
        if missing:
            raise FileNotFoundError(f"缺少 {len(missing)} 张图片，首个 Dicom_id={missing[0]}")
        self.paths = [index[value] for value in self.frame['Dicom_id']]
        self.labels = np.stack(self.frame['Predict_class'].map(parse_labels)) if labeled else None

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        try:
            with Image.open(self.paths[index]) as image:
                tensor = preprocess(image, self.size, self.normalization, self.augmentation)
        except Exception as exc:
            raise RuntimeError(f"图片读取失败，Dicom_id={row['Dicom_id']}") from exc
        sample = {key: row[key] for key in ('Subject_id', 'Study_id', 'Dicom_id')}
        for key in METADATA_COLUMNS:
            sample[key] = row[key]
            sample[key + '_missing'] = bool(row[key + '_missing'])
        sample['image'] = tensor
        if self.labels is not None:
            sample['label'] = torch.from_numpy(self.labels[index].copy())
        return sample


def make_loader(dataset: Dataset, batch_size: int = 8, shuffle: bool = False,
                num_workers: int = 4, seed: int = 42, pin_memory: bool = False) -> DataLoader:
    if type(num_workers) is not int or num_workers < 0:
        raise ValueError('num_workers必须为非负整数')
    # 保持有序返回和每轮重建worker；PyTorch按generator的base seed设置各worker RNG；增强只使用该RNG，支持整轮边界恢复。
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, generator=torch.Generator().manual_seed(seed),pin_memory=pin_memory)
