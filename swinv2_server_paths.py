"""生成服务器五路径配置，不重建划分或下载权重。"""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--data', type=Path, required=True, help='包含train.csv、valid.csv、train图片目录的目录')
p.add_argument('--split', type=Path, help='冻结v2完整8文件目录；官方模式不需要')
p.add_argument('--official', action='store_true', help='官方train训练、valid选优')
p.add_argument('--weight', type=Path, required=True)
p.add_argument('--output', type=Path, default=Path('configs/internal_ab/paths.swinv2.server.json'))
a = p.parse_args()
if a.official:
    if a.split: p.error('官方模式不得传入split')
    values = {'official_train_csv': (a.data/'train.csv').resolve(), 'official_valid_csv': (a.data/'valid.csv').resolve(),
              'train_images': (a.data/'train').resolve(), 'valid_images': (a.data/'valid').resolve(),
              'pretrained_weight_path': a.weight.resolve()}
else:
    if not a.split: p.error('v2模式需要--split')
    values = {'split_directory': a.split.resolve(), 'official_train_csv': (a.data/'train.csv').resolve(),
              'official_valid_csv': (a.data/'valid.csv').resolve(), 'image_root': (a.data/'train').resolve(),
              'pretrained_weight_path': a.weight.resolve()}
for key, path in values.items():
    if not path.exists():
        raise FileNotFoundError(f'{key}: {path}')
a.output.parent.mkdir(parents=True, exist_ok=True)
with a.output.open('x', encoding='utf-8') as f:
    json.dump({k: str(v) for k, v in values.items()}, f, ensure_ascii=False, indent=2)
print(a.output)
