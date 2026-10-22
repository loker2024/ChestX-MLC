"""完整epoch边界的状态保存/恢复，不冒充旧权重具备历史动量。"""
import hashlib
from pathlib import Path
import random
import warnings
import numpy as np
import torch

CONFIG_KEYS = ('size','batch_size','seed','learning_rate','weight_decay','preprocess','aggregation','threshold')


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_training_state(path: Path, model_path: Path, optimizer: torch.optim.Optimizer,
                        loader, validation_loader, epoch: int, config: dict,
                        train_fingerprint: str, validation_fingerprint: str) -> None:
    numpy_state = np.random.get_state()
    if path.parent.resolve() != model_path.parent.resolve():
        raise ValueError('模型与训练状态必须保存到同一轮次目录')
    device = config['device']
    state = {'schema':2,'epoch':epoch,'model_file':model_path.name,'model_sha256':_hash(model_path),
             'optimizer':optimizer.state_dict(), 'config':{k:config[k] for k in CONFIG_KEYS},
             'train_fingerprint':train_fingerprint,'validation_fingerprint':validation_fingerprint,
             'python_rng':random.getstate(),
             'numpy_rng':[numpy_state[0],numpy_state[1].tolist(),int(numpy_state[2]),int(numpy_state[3]),float(numpy_state[4])],
             'torch_rng':torch.get_rng_state(),
             'cuda_rng':torch.cuda.get_rng_state_all() if device == 'cuda' else [],
             'mps_rng':torch.mps.get_rng_state() if device == 'mps' else None,
             'loader_rng':loader.generator.get_state(),
             'validation_loader_rng':validation_loader.generator.get_state(),
             'device':device,'torch_version':str(torch.__version__),
             'num_workers':loader.num_workers}
    if loader.num_workers != 0:
        raise ValueError('完整随机状态恢复当前仅支持num_workers=0')
    with path.open('xb') as handle:
        torch.save(state,handle)


def load_training_state(path: Path) -> tuple[dict, Path]:
    state = torch.load(path,map_location='cpu',weights_only=True)
    if state.get('schema') != 2:
        raise ValueError('旧training_state没有完整状态及模型绑定，请使用仅权重续训；不能宣称断点恢复')
    name = state['model_file']
    if not isinstance(name,str) or Path(name).name != name or Path(name).is_absolute():
        raise ValueError('训练状态中的模型路径不合法')
    model_path = path.parent/name
    if _hash(model_path) != state['model_sha256']:
        raise ValueError('模型和优化器状态不是同一个检查点')
    checkpoint = torch.load(model_path,map_location='cpu',weights_only=True)
    metadata = checkpoint['metadata']
    if metadata.get('completed_epoch',metadata.get('epochs')) != state['epoch']:
        raise ValueError('模型轮次和状态轮次不一致')
    return state,model_path


def restore_training_state(state: dict, optimizer: torch.optim.Optimizer,
                           loader, validation_loader, config: dict,
                           train_fingerprint: str, validation_fingerprint: str) -> None:
    if state['device'] != config['device']:
        raise ValueError('完整恢复要求设备类型一致；跨设备请明确使用仅权重新实验')
    if state['config'] != {k:config[k] for k in CONFIG_KEYS}:
        raise ValueError('完整恢复不能同时改变学习率、批量、种子或预处理')
    if state['train_fingerprint'] != train_fingerprint or state['validation_fingerprint'] != validation_fingerprint:
        raise ValueError('完整恢复的数据划分或标签发生变化')
    if loader.num_workers != 0 or state['num_workers'] != 0:
        raise ValueError('当前仅支持单进程数据加载器的完整状态恢复')
    if state['torch_version'] != str(torch.__version__):
        warnings.warn('PyTorch版本不同，恢复状态不保证数值逐位一致',RuntimeWarning)
    optimizer.load_state_dict(state['optimizer'])
    random.setstate(state['python_rng'])
    saved = state['numpy_rng']
    np.random.set_state((saved[0],np.asarray(saved[1],dtype=np.uint32),saved[2],saved[3],saved[4]))
    torch.set_rng_state(state['torch_rng'])
    if config['device'] == 'cuda':
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    elif config['device'] == 'mps':
        torch.mps.set_rng_state(state['mps_rng'])
    loader.generator.set_state(state['loader_rng'])
    validation_loader.generator.set_state(state['validation_loader_rng'])
