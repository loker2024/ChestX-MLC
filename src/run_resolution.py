"""R1/R2高分辨率入口；默认计划，--check只CPU预检，--execute由用户手动训练。"""
import argparse
import json
from pathlib import Path

from .internal_ab import prepare, execute
from .internal_split import read_paths
from .resolution_experiments import ARM_IDS, REFERENCES, configuration, check_reference, memory_estimate


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',choices=['R1','R2'],required=True)
    p.add_argument('--config',type=Path)
    p.add_argument('--reference',type=Path)
    p.add_argument('--paths',type=Path,default=Path('configs/internal_ab/paths.v2.local.json'))
    p.add_argument('--output',type=Path)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    action=p.add_mutually_exclusive_group();action.add_argument('--check',action='store_true');action.add_argument('--execute',action='store_true')
    args=p.parse_args()
    reference=args.reference or REFERENCES[args.arm]
    config_path=args.config or Path('configs/experiments')/(ARM_IDS[args.arm]+'.json')
    config=json.loads(config_path.read_text(encoding='utf-8-sig'))
    if config!=configuration(args.arm,reference):raise ValueError('配置与原320实验不只分辨率不同')
    output=args.output or Path('outputs')/ARM_IDS[args.arm]
    if not args.check and not args.execute:
        print(json.dumps({'config':config,'reference':str(reference),'output':str(output),'training_started':False},ensure_ascii=False,indent=2));return
    paths=read_paths(args.paths)
    if args.check:
        datasets,audit,approval=prepare(config,paths)
        source=check_reference(config,reference,audit,image_root=paths['image_root'])
        # 仅两张CPU图像读取/预处理，不创建模型或使用GPU。
        samples={s:{'shape':list(d[0]['image'].shape),'source_path':str(d.paths[0].resolve()),
                    'studies':int(d.frame.Study_id.nunique()),'images':len(d)} for s,d in datasets.items()}
        print(json.dumps({'samples':samples,'weight':approval,'memory_estimate':memory_estimate(source['metrics']),
            'batch_size':config['batch_size'],'studies_per_batch':config['studies_per_batch'],
            'max_images_per_batch':config['max_images_per_batch'],'num_workers':config['num_workers'],
            'model_forward_performed':False,'GPU_used':False,'training_started':False},ensure_ascii=False,indent=2));return
    result=execute(config,paths,output,args.device,reference_run=reference)
    print('COMPLETE '+str(result.resolve()),flush=True)


if __name__=='__main__':main()
