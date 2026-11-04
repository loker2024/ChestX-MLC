"""导出 A-FUSE-5 的训练集 study 级离线软标签。"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from common.checkpoint import load_checkpoint
from common.config import config_from_dict
from common.constants import CLASS_IDS
from common.utils import resolve_device
from data.dataset import build_loader
from infer.ensemble import fuse_probability_vectors
from infer.predict import load_checkpoint_model, predict_loader
from run_ensemble import _check_labeled_loaders, _check_subject_loaders
from train.distillation import records_digest, sha256_file, write_teacher_cache


def fuse_teacher_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """按 Study_id 对齐后，使用提交计时路径的概率裁剪 logit 融合。"""
    ids = frames[0]['Study_id'].tolist()
    aligned = []
    cols = [f'p{i}' for i in CLASS_IDS]
    for frame in frames:
        if frame['Study_id'].duplicated().any() or set(frame['Study_id']) != set(ids):
            raise ValueError('教师输出 Study_id 不一致')
        aligned.append(frame.set_index('Study_id').loc[ids, cols].to_numpy())
    values = [fuse_probability_vectors([a[i] for a in aligned], [4, 2, 3], 'logit') for i in range(len(ids))]
    result = pd.DataFrame(np.asarray(values), columns=cols)
    result.insert(0, 'Study_id', ids)
    return result

def main() -> None:
    parser = argparse.ArgumentParser(description="导出 A-FUSE-5 train 软标签")
    parser.add_argument('--checkpoint', nargs=3, required=True, metavar=('D002_EMA', 'C005', 'E001'))
    parser.add_argument('--out', required=True, help='新建缓存目录')
    args = parser.parse_args()
    paths = [Path(p) for p in args.checkpoint]
    checkpoints = [load_checkpoint(path, map_location='cpu') for path in paths]
    if any(not ckpt.get('config') for ckpt in checkpoints):
        raise ValueError('教师 checkpoint 必须内嵌配置')
    configs = [config_from_dict(ckpt['config']) for ckpt in checkpoints]
    if any(c.data.resplit.enabled for c in configs):
        raise ValueError('教师导出不支持重划分')
    # 检查已记录的 A-FUSE-5 checkpoint 身份，防止顺序或权重混用。
    for path, expected in zip(paths, ('e97524bb', 'fb8bb525', '0ba09185')):
        md5 = hashlib.md5(path.read_bytes()).hexdigest()
        if not md5.startswith(expected):
            raise ValueError(f'教师 checkpoint 身份不匹配: {path} ({md5})')
    csv_hashes = [sha256_file(c.data.train.csv) for c in configs]
    if len(set(csv_hashes)) != 1:
        raise ValueError('教师训练 CSV 不一致')
    device = resolve_device(configs[0].device)
    loaders, frames = [], []
    for path, config in zip(paths, configs):
        loader = build_loader(config, 'train', train=False, pin_memory=device.type == 'cuda')
        loaders.append(loader)
        model = load_checkpoint_model(config, path, device)
        frames.append(predict_loader(model, loader, device, config))
        del model
    _check_subject_loaders(loaders)
    _check_labeled_loaders(loaders)
    index_hashes = [records_digest(loader.dataset.records) for loader in loaders]
    if len(set(index_hashes)) != 1:
        raise ValueError("教师图片索引或视角不一致")
    result = fuse_teacher_frames(frames)
    ids = result['Study_id'].tolist()
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    manifest = {'train_csv_sha256': csv_hashes[0], 'index_sha256': index_hashes[0], 'checkpoints': [{'path': str(p.resolve()), 'sha256': sha256_file(p), 'config': c.to_dict()} for p, c in zip(paths, configs)], 'fusion': 'logit_clipped_probability', 'weights': [4, 2, 3], 'classes': list(CLASS_IDS), 'git_head': revision, 'split': 'train'}
    write_teacher_cache(Path(args.out), result, manifest)
    print(f'教师缓存已生成: {args.out}，{len(ids)} studies；请按计划核对预测和数据来源。')


if __name__ == '__main__':
    main()
