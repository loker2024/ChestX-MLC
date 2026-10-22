"""只检查数据，不创建模型或启动训练。"""
import argparse
import json
from pathlib import Path

from PIL import Image
import torch
from .data import ChestXrayDataset, make_loader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/data.json')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    datasets = {}
    for split in ('train', 'valid', 'test'):
        item = config[split]
        dataset = ChestXrayDataset(item['csv'], item['images'], split != 'test', config['size'])
        datasets[split] = dataset
        batch = next(iter(make_loader(dataset, config['batch_size'], shuffle=True)))
        assert torch.isfinite(batch['image']).all()
        assert batch['image'].min() >= 0 and batch['image'].max() <= 1
        if split != 'test':
            assert batch['label'].shape == (len(batch['image']), 10)
        else:
            assert 'label' not in batch
        sizes = []
        lookup = dict(zip(dataset.frame['Dicom_id'], dataset.paths))
        for image_id in batch['Dicom_id']:
            with Image.open(lookup[image_id]) as image:
                sizes.append(image.size)
        print(json.dumps({'split': split, 'rows': len(dataset),
                          'columns': list(dataset.frame.columns),
                          'studies': dataset.frame['Study_id'].nunique(),
                          'matched_images': len(dataset.paths),
                          'original_sizes': sizes,
                          'image_shape': list(batch['image'].shape),
                          'label_shape': list(batch['label'].shape) if 'label' in batch else None}, ensure_ascii=False))
    for a, b in [('train', 'valid'), ('train', 'test'), ('valid', 'test')]:
        for key in ('Subject_id', 'Study_id', 'Dicom_id'):
            overlap = set(datasets[a].frame[key]) & set(datasets[b].frame[key])
            print(f'{a}/{b} {key} overlap={len(overlap)}')
            if overlap:
                raise ValueError('数据划分存在重叠，需要确认')


if __name__ == '__main__':
    main()
