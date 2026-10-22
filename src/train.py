"""默认运行单批次测试；--full-one-epoch 运行完整训练和验证一轮。"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import random
import sys
from uuid import uuid4

import numpy as np
import torch
from torch import nn
import torchvision

from .data import ChestXrayDataset, make_loader
from .model import build_model, load_model
from .devices import select_device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_step(model: nn.Module, images: torch.Tensor, labels: torch.Tensor,
               optimizer: torch.optim.Optimizer) -> tuple[float, list[int]]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(images)
    if logits.shape != labels.shape or logits.ndim != 2 or logits.shape[1] != 10:
        raise ValueError('输出和标签必须具有相同的 [B,10] 形状')
    if not torch.isfinite(labels).all() or not ((labels == 0) | (labels == 1)).all():
        raise ValueError('标签必须是有限的 0/1 值')
    if not torch.isfinite(logits).all():
        raise ValueError('模型输出包含非有限值')
    loss = nn.BCEWithLogitsLoss()(logits, labels)
    if not torch.isfinite(loss):
        raise ValueError('损失不是有限数值')
    loss.backward()
    if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
        raise ValueError('梯度包含非有限值')
    optimizer.step()
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('更新后的参数包含非有限值')
    return loss.item(), list(logits.shape)


def save_model(model: nn.Module, path: Path, metadata: dict) -> None:
    # 排他创建，避免覆盖已有模型；保存结构和预处理信息以便重新加载。
    with path.open('xb') as handle:
        torch.save({'architecture': 'resnet18', 'num_classes': 10,
                    'state_dict': model.state_dict(), 'metadata': metadata}, handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-config', default='configs/data.json')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--full-one-epoch', action='store_true')
    parser.add_argument('--device', choices=['auto','cuda','mps','cpu'], default='auto')
    parser.add_argument('--output',type=Path,help='完整一轮模式的新实验目录，存在时拒绝覆盖')
    parser.add_argument('--allow-nonofficial-counts',action='store_true',help='仅限小样本开发测试')
    args = parser.parse_args()
    if args.full_one_epoch:
        from .full_train import run
        run(args.device,Path(args.data_config),args.output or Path('outputs/baseline_v1'),args.seed,args.allow_nonofficial_counts)
        return
    if args.output is not None:
        parser.error('--output仅用于--full-one-epoch；单步测试自行创建独立目录')
    seed_everything(args.seed)
    torch.set_num_threads(2)
    device = select_device(args.device)
    print(f'Device: {device}; ' + (torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'), flush=True)
    config = json.loads(Path(args.data_config).read_text(encoding='utf-8'))
    dataset = ChestXrayDataset(config['train']['csv'], config['train']['images'], size=config['size'])
    # CPU/GPU 均只取两张，且固定仅更新一次。
    batch = next(iter(make_loader(dataset, batch_size=2, shuffle=True, seed=args.seed)))
    images, labels = batch['image'].to(device), batch['label'].to(device)
    model = build_model().to(device)
    before = model.fc.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    loss, shape = train_step(model, images, labels, optimizer)
    delta = (model.fc.weight.detach() - before).abs().max().item()
    if delta <= 0:
        raise RuntimeError('参数未更新')
    report = {'seed': args.seed, 'device': device,
              'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None,
              'python': sys.version, 'torch': str(torch.__version__),
              'torchvision': str(torchvision.__version__), 'split': 'train',
              'batch_size': 2, 'steps': 1, 'learning_rate': 1e-3, 'weight_decay': 1e-2,
              'weights': None, 'size': config['size'], 'preprocess': 'RGB letterbox /255',
              'output_shape': shape, 'label_shape': list(labels.shape), 'loss': loss,
              'fc_weight_max_change': delta}
    output = Path('outputs') / ('round2_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = output / 'model.pth'
    save_model(model, checkpoint, report)
    restored = load_model(str(checkpoint), device)
    for key, value in model.state_dict().items():
        if not torch.equal(value, restored.state_dict()[key]):
            raise RuntimeError(f'参数恢复不一致: {key}')
    model.eval()
    with torch.no_grad():
        expected, actual = model(images), restored(images)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        probabilities = actual.sigmoid()
        if not torch.isfinite(probabilities).all():
            raise RuntimeError('恢复模型预测无效')
    report.update(reload_equal=True, checkpoint=str(checkpoint),
                  probability_min=probabilities.min().item(), probability_max=probabilities.max().item())
    (output / 'smoke_result.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
