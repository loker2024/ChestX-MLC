from dataclasses import replace

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from common.config import ImageConfig, load_config
from data.transforms import Letterbox, _normalize_steps, build_train_transform
from run_train import _validate_resume_config


def test_default_train_transform_matches_previous_pipeline():
    image = Image.fromarray(np.arange(32 * 24, dtype=np.uint8).reshape(32, 24))
    old = T.Compose([Letterbox(32), T.RandomAffine(7, translate=(0.05, 0.05), scale=(0.9, 1.1), interpolation=InterpolationMode.BILINEAR, fill=0), T.ColorJitter(brightness=0.15, contrast=0.15), *_normalize_steps('imagenet', 3, 255)])
    torch.manual_seed(42)
    expected = old(image)
    torch.manual_seed(42)
    actual = build_train_transform(32, ImageConfig(size=32))(image)
    assert torch.equal(actual, expected)


def test_jitter_parameters_apply_to_standard_and_xrv_16bit():
    cfg = ImageConfig(size=32, brightness=0.05, contrast=0.25)
    standard = build_train_transform(32, cfg)
    jitter = next(x for x in standard.transforms if isinstance(x, T.ColorJitter))
    assert jitter.brightness == pytest.approx((0.95, 1.05))
    assert jitter.contrast == pytest.approx((0.75, 1.25))
    xrv = build_train_transform(32, replace(cfg, normalize='xrv', in_channels=1, xrv_maxval=65535))
    jitter16 = next(x for x in xrv.augment.transforms if isinstance(x, T.ColorJitter))
    assert jitter16.brightness == pytest.approx(jitter.brightness)
    assert jitter16.contrast == pytest.approx(jitter.contrast)
    image16 = Image.fromarray(np.arange(32 * 32, dtype=np.uint16).reshape(32, 32))
    assert torch.isfinite(xrv(image16)).all()


@pytest.mark.parametrize('field,value', [('brightness', -0.1), ('contrast', -0.1), ('brightness', float('nan')), ('contrast', float('inf')), ('brightness', 1.01)])
def test_invalid_jitter_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        ImageConfig(**{field: value})


def test_resume_rejects_changed_train_augmentation():
    cfg = load_config('configs/exp_A001.yaml')
    changed = replace(cfg, image=replace(cfg.image, brightness=0.05))
    with pytest.raises(ValueError, match='image.brightness'):
        _validate_resume_config(changed, cfg)
