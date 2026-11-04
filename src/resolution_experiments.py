"""R1/R2固定384对照：仅改输入尺寸和实验编号；不生成448实验。"""
import copy
import json
from pathlib import Path

ARM_IDS={'R1':'R1_DenseNet121_384_BCE_v2','R2':'R2_DenseNet121_384_ASL_v2'}
REFERENCE_IDS={'R1':'Internal_A_DenseNet121_320_Study','R2':'C1_ASL_DenseNet121_320_v2'}
REFERENCES={'R1':Path('outputs/Internal_AB_local_v2/A'),'R2':Path('outputs/C1_ASL_DenseNet121_320_v2_run2')}
V2_SHA='5c8cc5a132454472254973f1ad3b22d800e05c0215660f6f41e4ddd1f05a6b91'


def arm_for(config: dict) -> str:
    matches=[arm for arm,name in ARM_IDS.items() if config.get('experiment_id')==name]
    if len(matches)!=1:raise ValueError('只允许已计划的R1/R2固定384实验')
    return matches[0]


def validate_resolution_config(config: dict) -> dict:
    from .internal_ab import validate_config
    arm=arm_for(config)
    if type(config.get('input_size')) is not int or config['input_size']!=384:
        raise ValueError('R1/R2输入必须为384，不自动开展448')
    proxy=copy.deepcopy(config);proxy.update(experiment_id=REFERENCE_IDS[arm],input_size=320)
    validate_config(proxy)
    return copy.deepcopy(config)


def configuration(arm: str, reference: Path) -> dict:
    from .internal_ab import validate_config
    if arm not in ARM_IDS:raise ValueError('未计划的实验')
    settings=json.loads((reference/'train_config.json').read_text(encoding='utf-8'))
    source=validate_config(settings['experiment_config'])
    if source['experiment_id']!=REFERENCE_IDS[arm] or settings['size']!=320:
        raise ValueError('R1必须对应原A，R2必须对应原C1，不允许续训或换对照')
    config=copy.deepcopy(source);config.update(experiment_id=ARM_IDS[arm],input_size=384)
    return validate_resolution_config(config)


def check_reference(config: dict, reference: Path, audit: dict, image_root: Path | None = None) -> dict:
    from .compare_internal_ab import read_run
    arm=arm_for(config)
    if config!=configuration(arm,reference):
        raise ValueError('384与320源配置除尺寸/编号外存在差异，停止')
    source=read_run(reference)
    if audit['split_manifest_sha256']!=V2_SHA:
        raise ValueError('384实验必须使用同一固定v2')
    for key in ('split_manifest_sha256','train_fingerprint','validation_fingerprint'):
        if audit[key]!=source['audit'][key]:raise ValueError('384划分与原对照不一致：'+key)
    if image_root is None or any(Path(source['metadata']['data'][s]['images']).resolve()!=image_root.resolve() for s in ('train','valid')):
        raise ValueError('384必须读取原320实验使用的原始图片目录，禁止替换为320图片缓存')
    return source


def memory_estimate(reference_metrics: dict, parameter_count: int = 6964106) -> dict:
    ratio=(384/320)**2
    peak=reference_metrics.get('peak_allocated_bytes')
    reserved=reference_metrics.get('peak_reserved_bytes')
    fixed=parameter_count*16  # FP32参数、梯度与AdamW两组动量；仅近似，不含所有实现开销。
    estimate=fixed+max(0,peak-fixed)*ratio if peak is not None else None
    return {'pixel_area_ratio':ratio,'physical_batch_policy':'same 8 studies / maximum 16 images; variable image batch retained',
        'reference_peak_allocated_bytes':peak,'reference_peak_reserved_bytes':reserved,
        'rough_384_peak_allocated_bytes':estimate,
        'rough_384_peak_allocated_GiB':estimate/1024**3 if estimate is not None else None,
        'rough_reserved_linear_GiB':reserved*ratio/1024**3 if reserved is not None else None,
        'GPU_measured':False,'fits_GPU_guaranteed':False,'automatic_batch_fallback':False,
        'AMP_added':False,'gradient_accumulation_added':False,'BN_frozen':False,
        'note':'激活近似随像素面积增长；cuDNN工作区/验证重载/碎片不能精确外推。reserved线性估计不是必需显存，不可据此断定一定OOM。'}
