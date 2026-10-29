"""十类别骨干工厂：ResNet18、DenseNet121、EfficientNet-B2及已核准本地权重。"""
from pathlib import Path
import re
import torch
from torch import nn
from torchvision.models import resnet18, densenet121, efficientnet_b2
from .pretrained import require_approved_weight, REGISTRY

BACKBONES = {'resnet18':resnet18, 'densenet121':densenet121, 'efficientnet_b2':efficientnet_b2}


def classification_head(model: nn.Module) -> nn.Linear:
    """保留原生分类器布局，返回可训练的末层Linear供参数分组。"""
    if getattr(model,'cxr_backbone',None)=='efficientnet_b2':
        head=model.classifier[1]
        if not isinstance(head,nn.Linear):
            raise ValueError('EfficientNet分类器末层必须是Linear')
        return head
    head = model.classifier if getattr(model,'cxr_backbone','resnet18') == 'densenet121' else model.fc
    if not isinstance(head,nn.Linear):
        raise ValueError('分类层必须是Linear')
    return head


def normalize_densenet_keys(state: dict) -> dict:
    """沿用TorchVision官方旧LuaTorch键名迁移；迁移后仍strict加载。"""
    pattern = re.compile(r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
    converted = state.copy()
    if hasattr(state,'_metadata'):
        converted._metadata = state._metadata
    for key in list(state):
        match = pattern.match(key)
        if match:
            new_key = match.group(1)+match.group(2)
            if new_key in converted:
                raise ValueError('DenseNet旧/新权重键名冲突')
            converted[new_key] = converted.pop(key)
    return converted


def verify_backbone_state(model: nn.Module, source: dict, backbone: str) -> dict:
    """在首次更新前逐张量对照本地源权重；分类头不属于核对范围。"""
    if backbone not in BACKBONES:
        raise ValueError('不支持的骨干身份核对')
    head_prefix = 'fc.' if backbone == 'resnet18' else 'classifier.'
    compared,legacy_counters = 0,0
    for name,value in model.state_dict().items():
        if name.startswith(head_prefix):
            continue
        if name not in source:
            # 旧Torch权重没有BN计数器，官方加载器将其初始化为零。
            if name.endswith('.num_batches_tracked') and int(value) == 0:
                legacy_counters += 1
                continue
            raise RuntimeError('骨干源权重缺失：'+name)
        expected = source[name]
        if value.dtype != expected.dtype or not torch.equal(value.cpu(),expected.cpu()):
            raise RuntimeError('骨干与源预训练权重不一致：'+name)
        compared += 1
    if compared == 0:
        raise RuntimeError('没有可核对的骨干张量')
    return {'backbone':backbone,'compared_tensors':compared,'legacy_zero_counters':legacy_counters,
            'all_equal':True,'checked_before_training':True,'classification_head_excluded':True}


def build_model(pretrained_weight_path: str | Path | None = None,
                pretrained_weight_sha256: str | None = None,
                pretrained_registry: str | Path = REGISTRY, *,
                backbone: str = 'resnet18', model_backend: str = 'torchvision',
                pretrained_variant: str | None = None) -> nn.Module:
    if backbone not in BACKBONES:
        raise ValueError('不支持的骨干：'+str(backbone))
    provenance,verification = None,None
    if model_backend not in ('torchvision','timm') or (model_backend=='timm' and backbone!='densenet121'):
        raise ValueError('不支持的模型后端')
    if pretrained_variant is not None and (pretrained_variant!='densenet121_ra' or model_backend!='timm'):
        raise ValueError('RA权重必须使用明确的timm后端，不能回退为TorchVision')
    if model_backend=='timm' and pretrained_weight_path is not None and pretrained_variant!='densenet121_ra':
        raise ValueError('timm预训练必须显式指定RA身份')
    if pretrained_weight_path is not None:
        if pretrained_variant is not None:
            provenance = require_approved_weight(Path(pretrained_weight_path),pretrained_weight_sha256,Path(pretrained_registry),backbone=backbone,variant=pretrained_variant)
        elif backbone == 'resnet18':
            provenance = require_approved_weight(Path(pretrained_weight_path),pretrained_weight_sha256,Path(pretrained_registry))
        else:
            provenance = require_approved_weight(Path(pretrained_weight_path),pretrained_weight_sha256,Path(pretrained_registry),backbone=backbone)
    elif pretrained_weight_sha256 is not None:
        raise ValueError('提供SHA-256时必须同时指定本地权重路径')
    # 永远使用weights=None，禁止torchvision隐式下载。
    if model_backend=='timm':
        try:
            import timm
        except ImportError as exc:
            raise ImportError('RA实验需要安装timm==1.0.30；禁止回退到TorchVision或随机模型') from exc
        model=timm.create_model('densenet121.ra_in1k',pretrained=False)
        if model.pretrained_cfg.get('tag')!='ra_in1k':
            raise ValueError('timm模型身份不是ra_in1k')
        if (tuple(model.pretrained_cfg.get('mean',()))!=(.485,.456,.406) or
                tuple(model.pretrained_cfg.get('std',()))!=(.229,.224,.225)):
            raise ValueError('RA官方归一化与冻结训练配置不一致')
        if model.classifier.out_features!=1000 or model.classifier.in_features!=1024:
            raise ValueError('RA原始分类头不是预期的1024到1000结构')
        model.cxr_backend_version=timm.__version__
    else:
        model = BACKBONES[backbone](weights=None)
    if provenance is not None:
        if pretrained_variant=='densenet121_ra':
            from safetensors.torch import load_file
            state=load_file(str(pretrained_weight_path),device='cpu')
        else:
            state = torch.load(pretrained_weight_path,map_location='cpu',weights_only=True)
        if backbone == 'densenet121' and model_backend=='torchvision':
            state = normalize_densenet_keys(state)
        # 先严格匹配原始1000类结构，再替换10类头，不能用strict=False吞掉骨干错配。
        model.load_state_dict(state,strict=True)
        verification = verify_backbone_state(model,state,backbone)
        verification['source_sha256'] = pretrained_weight_sha256
    if backbone == 'resnet18':
        model.fc = nn.Linear(model.fc.in_features, 10)
    elif backbone == 'efficientnet_b2':
        # 保留TorchVision原生Dropout，只替换末层；参数分组使用实际Linear。
        model.classifier[1] = nn.Linear(model.classifier[1].in_features,10)
    else:
        model.classifier = nn.Linear(model.classifier.in_features, 10)
    model.cxr_backbone = backbone
    model.cxr_backend = model_backend
    model.pretrained_provenance = provenance
    model.pretrained_backbone_verification = verification
    return model


def load_model(path: str, device: str = 'cpu') -> nn.Module:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    architecture = checkpoint['architecture']
    if architecture not in BACKBONES or checkpoint['num_classes'] != 10:
        raise ValueError('不支持的模型结构')
    if checkpoint.get('metadata',{}).get('backbone',architecture) != architecture:
        raise ValueError('模型架构与元数据冲突')
    backend=checkpoint.get('model_backend','torchvision')
    if checkpoint.get('metadata',{}).get('model_backend',backend)!=backend:
        raise ValueError('检查点模型后端与元数据冲突')
    model = build_model(backbone=architecture,model_backend=backend).to(device)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model.eval()
