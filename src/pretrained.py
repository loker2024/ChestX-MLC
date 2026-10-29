"""本地预训练权重审查门禁；不联网、不自动下载，也不自动推断法律许可。"""
import hashlib
import json
from pathlib import Path
import re

REGISTRY = Path('configs/pretrained_registry.json')
WEIGHT_NAME = 'ResNet18_Weights.IMAGENET1K_V1'
WEIGHT_URL = 'https://download.pytorch.org/models/resnet18-f37072fd.pth'
WEIGHT_SPECS = {'efficientnet_b2':('EfficientNet_B2_Weights.IMAGENET1K_V1','https://download.pytorch.org/models/efficientnet_b2_rwightman-c35c1473.pth'),
    'resnet18':(WEIGHT_NAME,WEIGHT_URL),
    'densenet121':('DenseNet121_Weights.IMAGENET1K_V1','https://download.pytorch.org/models/densenet121-a639ec97.pth'),
    'densenet121_ra':('timm/densenet121.ra_in1k','https://huggingface.co/timm/densenet121.ra_in1k/resolve/92007b6200e0b4a4fe68cb4e3947022a928aaaae/model.safetensors')}


def file_sha256(path: Path) -> str:
    with path.open('rb') as handle:
        file_digest = getattr(hashlib,'file_digest',None)
        if file_digest is not None:
            digest = file_digest(handle,'sha256')
        else:
            # 保留队友Python3.10环境兼容性；file_digest在3.11才加入。
            digest = hashlib.sha256()
            while chunk := handle.read(1024*1024):
                digest.update(chunk)
    return digest.hexdigest()


def require_approved_weight(path: Path, expected_sha256: str,
                            registry_path: Path = REGISTRY, *, backbone: str = 'resnet18',
                            variant: str | None = None) -> dict:
    """审查记录需绑定具体权重和证据文件；pending条目不能进入模型加载。"""
    if not isinstance(expected_sha256,str) or not re.fullmatch(r'[0-9a-f]{64}',expected_sha256):
        raise ValueError('必须提供完整小写SHA-256，不能用文件名中的8位哈希代替')
    if backbone not in ('resnet18','densenet121','efficientnet_b2') or (variant is not None and (variant!='densenet121_ra' or backbone!='densenet121')):
        raise ValueError('不支持的预训练骨干')
    weight_name,weight_url = WEIGHT_SPECS[variant or backbone]
    registry = json.loads(registry_path.read_text(encoding='utf-8'))
    matches = [row for row in registry['weights'] if row.get('sha256') == expected_sha256
               and row.get('weights_name') == weight_name]
    if len(matches) != 1:
        raise ValueError('登记表没有唯一匹配的候选权重')
    row = matches[0]
    if row.get('review_status') != 'approved':
        raise PermissionError('权重许可尚未核实，禁止加载用于训练：'+str(row.get('blocking_reason','pending')))
    required = {'competition_use':True,'redistribution':True,'nc_restriction':False,'gated':False}
    if any(row.get('permissions',{}).get(key) is not value for key,value in required.items()):
        raise PermissionError('赛事使用/再分发/NC/gated审查未全部通过')
    if row.get('primary_dataset') != 'ImageNet-1K' or row.get('download_url') != weight_url:
        raise PermissionError('只支持已审核并绑定来源的ImageNet-1K候选，不接受医学来源或替换链接')
    if not all(isinstance(row.get(key),str) and row[key].strip() for key in ('reviewer','reviewed_at','weight_license')):
        raise PermissionError('缺少审查人、日期或权重许可证名称')
    if row['weight_license'].lower() in ('unknown','pending','noassertion') or 'nc' in row['weight_license'].lower():
        raise PermissionError('权重许可未定或含NC限制')
    evidence = row.get('approval_evidence',[])
    kinds = {'weight_license','upstream_data','competition_and_redistribution'}
    if not kinds.issubset({item.get('kind') for item in evidence}):
        raise PermissionError('缺少权重许可全文、上游数据说明或赛事/再分发证据')
    for item in evidence:
        if not str(item.get('url','')).startswith('https://'):
            raise PermissionError('证据须记录公开HTTPS来源')
        document = Path(item['path'])
        if not document.is_file() or file_sha256(document) != item.get('sha256'):
            raise PermissionError('证据文件缺失或哈希不符')
    if not path.is_file():
        raise FileNotFoundError(f'已核准本地权重不存在：{path}；程序不会自动下载')
    if file_sha256(path) != expected_sha256:
        raise ValueError('本地预训练文件SHA-256不匹配')
    return {'weights_name':weight_name,'sha256':expected_sha256,'registry_sha256':file_sha256(registry_path),
            'registry_path':str(registry_path.resolve()),'primary_dataset':row['primary_dataset'],
            'download_url':row['download_url'],'weight_license':row['weight_license'],
            'reviewer':row['reviewer'],'reviewed_at':row['reviewed_at'],'review_status':'approved',
            'review_basis':row.get('review_basis','registry_review'),
            'variant':variant or backbone,'recipe':row.get('recipe','torchvision_original'),
            'revision':row.get('revision',''),
            'user_confirmation_path':row.get('user_confirmation_path',''),
            'user_confirmation_sha256':row.get('user_confirmation_sha256','')}
