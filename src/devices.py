"""跨 CUDA、MPS、CPU 的设备选择和统计；不在非 CUDA 设备调用 CUDA API。"""
import torch


def select_device(requested: str = 'auto') -> str:
    available = {'cuda': torch.cuda.is_available(),
                 'mps': torch.backends.mps.is_available(), 'cpu': True}
    if requested == 'auto':
        return next(name for name in ('cuda','mps','cpu') if available[name])
    if requested not in available or not available[requested]:
        raise ValueError(f'设备不可用: {requested}')
    return requested


def synchronize(device: str) -> None:
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()


def reset_peak(device: str) -> None:
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()


def peak_memory(device: str) -> dict:
    return {'peak_allocated_bytes': torch.cuda.max_memory_allocated() if device == 'cuda' else None,
            'peak_reserved_bytes': torch.cuda.max_memory_reserved() if device == 'cuda' else None}
