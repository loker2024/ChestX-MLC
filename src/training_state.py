"""完整epoch边界的状态保存/恢复，不冒充旧权重具备历史动量。"""
import hashlib
from pathlib import Path
import random
import warnings
import numpy as np
import torch

CONFIG_KEYS = ('size','batch_size','seed','learning_rate','weight_decay','preprocess','aggregation','threshold')


def _loader_resume_mode(loader, validation_loader) -> str:
    """支持原生Dataset；轻增强仅用PyTorch RNG，由有序、非持久worker的base seed重建。"""
    if loader.num_workers == validation_loader.num_workers == 0:
        return 'single_process'
    from .data import ChestXrayDataset
    if any(type(item.dataset) is not ChestXrayDataset or item.persistent_workers
           for item in (loader,validation_loader)):
        raise ValueError('多进程恢复只支持原生ChestXrayDataset且persistent_workers=False')
    if getattr(loader.dataset,'augmentation','none') != 'none':
        return 'torch_seeded_augmentation_ordered_epoch_boundary_v1'
    return 'deterministic_ordered_epoch_boundary_v1'


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
    state['optimizer_spec'] = {key:config[key] for key in ('backbone_lr','head_lr') if key in config} or None
    state['augmentation'] = config.get('augmentation','none')
    state['architecture'] = config.get('backbone','resnet18')
    state['model_backend'] = config.get('model_backend','torchvision')
    state['lr_schedule'] = config.get('lr_schedule')
    state['lr_schedule_epoch'] = epoch if config.get('lr_schedule') else None
    state['loader_resume_mode'] = _loader_resume_mode(loader,validation_loader)
    state['validation_num_workers'] = validation_loader.num_workers
    state['study_sampler_spec'] = {k:config[k] for k in ('sampler','loss_reduction','studies_per_batch','max_images_per_batch','pin_memory') if k in config} or None
    state['study_sampler_rng'] = loader.batch_sampler.generator.get_state() if config.get('sampler')=='study_batch' else None
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
    if state.get('augmentation','none') != config.get('augmentation','none'):
        raise ValueError('完整恢复不能改变训练增强参数')
    if state['device'] != config['device']:
        raise ValueError('完整恢复要求设备类型一致；跨设备请明确使用仅权重新实验')
    if state.get('architecture','resnet18') != config.get('backbone','resnet18'):
        raise ValueError('完整恢复不能改变骨干结构')
    if state.get('model_backend','torchvision')!=config.get('model_backend','torchvision'):
        raise ValueError('完整恢复不能改变RA/TV模型后端')
    if state.get('study_sampler_spec') != ({k:config[k] for k in ('sampler','loss_reduction','studies_per_batch','max_images_per_batch','pin_memory') if k in config} or None):
        raise ValueError('完整恢复不能改变Study采样、图片上限或Loss语义')
    if state['config'] != {k:config[k] for k in CONFIG_KEYS}:
        raise ValueError('完整恢复不能同时改变学习率、批量、种子或预处理')
    if state.get('optimizer_spec') != ({key:config[key] for key in ('backbone_lr','head_lr') if key in config} or None):
        raise ValueError('完整恢复不能改变PT1骨干/分类头分组学习率')
    if state.get('lr_schedule') != config.get('lr_schedule'):
        raise ValueError('完整恢复不能改变调度类型、warmup或Cosine预算')
    if state.get('lr_schedule'):
        from .lr_schedule import epoch_factor
        if state.get('lr_schedule_epoch') != state['epoch']:
            raise ValueError('调度轮次与模型状态不匹配')
        factor = epoch_factor(state['lr_schedule'], state['epoch'])
        for group in state['optimizer']['param_groups']:
            if group.get('name') not in ('backbone','head') or group['lr'] != config[group['name']+'_lr'] * factor:
                raise ValueError('保存的参数组学习率与调度位置不匹配')
    if state['train_fingerprint'] != train_fingerprint or state['validation_fingerprint'] != validation_fingerprint:
        raise ValueError('完整恢复的数据划分或标签发生变化')
    if (loader.num_workers != state['num_workers'] or
            validation_loader.num_workers != state.get('validation_num_workers',0)):
        raise ValueError('完整恢复必须保持训练和验证num_workers不变')
    if _loader_resume_mode(loader,validation_loader) != state.get('loader_resume_mode','single_process'):
        raise ValueError('数据加载恢复模式不一致')
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
    if config.get('sampler')=='study_batch':
        if state.get('study_sampler_rng') is None:
            raise ValueError('缺少Study采样器恢复状态')
        loader.batch_sampler.generator.set_state(state['study_sampler_rng'])
