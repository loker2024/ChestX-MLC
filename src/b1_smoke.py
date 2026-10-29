"""DenseNet121单批验收与本地batch1初测；不运行完整训练或正式test。"""
import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from .data import ChestXrayDataset, preprocess
from .devices import select_device, synchronize, reset_peak, peak_memory
from .model import load_model
from .pretrained import file_sha256
from .pt1 import load_config, data_settings, smoke


def run(config_path: Path, device_name: str = 'auto', interface_only: bool = True,
        repeats: int = 30) -> Path:
    config = load_config(config_path)
    if config['backbone'] != 'densenet121' or not 5 <= repeats <= 100:
        raise ValueError('仅支持DenseNet121及5到100次初测')
    device = select_device(device_name)
    free,total = torch.cuda.mem_get_info() if device == 'cuda' else (None,None)
    reset_peak(device)
    # 调用原单批程序；interface模式使用随机初始化，不跳过许可加载候选。
    output = smoke(config_path,device,interface_only,batch_size=config['batch_size'])
    training_peak = peak_memory(device)
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    model = load_model(str(output/'model.pth'),device)
    data = data_settings(config)
    dataset = ChestXrayDataset(data['train']['csv'],data['train']['images'],size=224,normalization='imagenet')
    image_path = dataset.paths[0]
    image_hash = file_sha256(image_path)
    fixed = dataset[0]['image'].unsqueeze(0).to(device)
    timings,whole = [],[]
    timing_dir = output/'timing'
    timing_dir.mkdir(exist_ok=False)
    with torch.inference_mode():
        for _ in range(5):
            model(fixed).sigmoid()
        synchronize(device)
        for _ in range(repeats):
            synchronize(device)
            start = time.perf_counter()
            model(fixed).sigmoid()
            synchronize(device)
            timings.append((time.perf_counter()-start)*1000)
        for i in range(repeats):
            synchronize(device)
            start = time.perf_counter()
            with Image.open(image_path) as image:
                tensor = preprocess(image,224,'imagenet').unsqueeze(0).to(device)
            probabilities = model(tensor).sigmoid().cpu().tolist()[0]
            with (timing_dir/f'probabilities_{i:03d}.json').open('x',encoding='utf-8') as handle:
                json.dump(probabilities,handle,allow_nan=False)
            synchronize(device)
            whole.append((time.perf_counter()-start)*1000)
    if file_sha256(image_path) != image_hash:
        raise RuntimeError('测速源训练图片发生变化')
    def summary(values: list[float]) -> dict:
        return {'mean_ms':float(np.mean(values)),'median_ms':float(np.median(values)),
                'p95_ms':float(np.percentile(values,95))}
    report = {'architecture':'densenet121','num_classes':10,'parameters':sum(p.numel() for p in model.parameters()),
        'checkpoint_bytes':(output/'model.pth').stat().st_size,'device':device,
        'gpu':torch.cuda.get_device_name(0) if device=='cuda' else None,
        'gpu_free_before_bytes':free,'gpu_total_bytes':total,
        'tested_training_batch_size':config['batch_size'],'single_batch_process_peak':training_peak,
        'precision':'float32; no AMP','inference_batch_size':1,'warmup':5,'repeats':repeats,
        'forward_sigmoid_gpu_input':summary(timings),'read_preprocess_transfer_forward_sigmoid_write_json':summary(whole),
        'timing_scope':'same official training image repeated; warm OS cache; loaded eval model; no Study aggregation',
        'official_T4_tested':False,'official_100ms_compliance_confirmed':False,
        'full_500MB_delivery_audited':False,'candidate_weight_used':not interface_only,
        'full_epoch_executed':False,'official_test_set_used':False,
        'study_macro_auc':None,'study_macro_f1':None}
    with (output/'profile_result.json').open('x',encoding='utf-8') as handle:
        json.dump(report,handle,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=Path('configs/experiments/B1_DenseNet121.json'))
    parser.add_argument('--device',choices=['auto','cuda','cpu','mps'],default='auto')
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--interface-smoke',action='store_true',help='明确使用随机初始化，不加载候选权重')
    modes.add_argument('--pretrained-smoke',action='store_true',help='仅许可核准后真实加载候选')
    args = parser.parse_args()
    print('COMPLETE '+str(run(args.config,args.device,args.interface_smoke).resolve()))


if __name__ == '__main__':
    main()
