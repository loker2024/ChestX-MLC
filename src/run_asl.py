"""C1固定ASL单次实验；默认计划，--execute明确训练，不搜索参数。"""
import argparse
import copy
import json
from pathlib import Path

from .analysis_common import dump
from .asl import ASL_PARAMETERS
from .compare_internal_ab import read_run
from .internal_ab import execute, validate_config
from .internal_split import read_paths


def configuration(reference: Path) -> dict:
    # 只读取A的配置和预测核验身份，绝不以A最佳state_dict初始化。
    source=read_run(reference)
    config=copy.deepcopy(source['config'])
    config.update(experiment_id='C1_ASL_DenseNet121_320_v2',loss='ASL',
        loss_reduction='study_balanced_image_asl',asl=ASL_PARAMETERS)
    return validate_config(config)


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('configs/experiments/C1_ASL_DenseNet121_320_v2.json'))
    p.add_argument('--reference',type=Path,default=Path('outputs/Internal_AB_local_v2/A'))
    p.add_argument('--paths',type=Path,default=Path('configs/internal_ab/paths.v2.local.json'))
    p.add_argument('--output',type=Path,default=Path('outputs/C1_ASL_DenseNet121_320_v2'))
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda');p.add_argument('--execute',action='store_true')
    a=p.parse_args();config=validate_config(json.loads(a.config.read_text(encoding='utf-8-sig')))
    if config!=configuration(a.reference):raise ValueError('C1配置与A的实际配置不一致')
    if not a.execute:
        print(json.dumps({'config':config,'training_started':False},ensure_ascii=False,indent=2));return
    result=execute(config,read_paths(a.paths),a.output,a.device,reference_run=a.reference)
    print('TRAIN_COMPLETE '+str(result.resolve()),flush=True)
    from .asl_comparison import compare_asl
    compare_asl(a.reference,result,result/'comparison')
    print('COMPLETE '+str(result.resolve()),flush=True)


if __name__=='__main__':main()
