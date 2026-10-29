"""仅从官方train生成私有增强预览；不随代码包交付真实影像。"""
import argparse
import json
from itertools import zip_longest
from pathlib import Path
from PIL import Image, ImageDraw
import torch
from .data import ChestXrayDataset, preprocess
from .pt1 import load_config, data_settings


def create_preview(config: dict, output: Path) -> Path:
    data = data_settings(config)['train']
    ds = ChestXrayDataset(data['csv'], data['images'], size=320)
    sizes = ds.frame.groupby('Study_id').size()
    pools = []
    for multiple in (False, True):
        frame = ds.frame[ds.frame.Study_id.map(sizes).gt(1) == multiple]
        selected = []
        for _, group in frame.groupby('ViewPosition', dropna=False, sort=False):
            selected.append(int(group.index[0]))
        pools.append(selected)
    selected = [index for pair in zip_longest(*pools) for index in pair if index is not None][:6]
    output.mkdir(parents=True, exist_ok=False)
    canvas = Image.new('RGB', (5*320, len(selected)*345), 'white')
    draw = ImageDraw.Draw(canvas)
    with torch.random.fork_rng(devices=[]):
        for row, index in enumerate(selected):
            record = ds.frame.iloc[index]
            for col in range(5):
                torch.default_generator.manual_seed(config['seed'] + row*100 + col)
                with Image.open(ds.paths[index]) as image:
                    tensor = preprocess(image, 320, 'none', config['augmentation'] if col else 'none')
                tile = Image.fromarray((tensor.permute(1,2,0).numpy()*255).round().astype('uint8'))
                canvas.paste(tile, (col*320, row*345+25))
                title = 'A0' if col == 0 else f'A1 sample {col}'
                draw.text((col*320+5,row*345+5), f'{title}; views={sizes[record.Study_id]}; {record.ViewPosition}', fill='black')
    canvas.save(output/'preview.png')
    (output/'augmentation.json').write_text(json.dumps(config['augmentation'], indent=2), encoding='utf-8')
    return output/'preview.png'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    print(create_preview(load_config(args.config), args.output).resolve())


if __name__ == '__main__':
    main()
