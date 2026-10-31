from dataclasses import replace
from pathlib import Path
import random

import numpy as np
import pytest
import torch
from PIL import Image

import run_train
from common.config import load_config
from common.checkpoint import save_checkpoint
from data.dataset import load_image
from data.transforms import build_eval_transform
from infer.predict import load_checkpoint_model
from model.network import build_model
from model.xrv import XrvDenseNetClassifier


def cfg():
    return load_config('configs/default.yaml')


def test_prediction_rejects_checkpoint_image_or_aggregation_mismatch(tmp_path):
    c = replace(cfg(), model=replace(cfg().model, pretrained=False))
    path = save_checkpoint(tmp_path / 'best.pt', build_model(c), config=c)
    altered = replace(c, image=replace(c.image, size=c.image.size + 1))
    with pytest.raises(ValueError, match='image.size'):
        load_checkpoint_model(altered, path, torch.device('cpu'))
    altered = replace(c, model=replace(c.model, aggregation='view_weighted'))
    with pytest.raises(ValueError, match='model.aggregation'):
        load_checkpoint_model(altered, path, torch.device('cpu'))


def test_research_checkpoint_rejected_for_submission(tmp_path):
    import run_infer
    import run_ensemble
    c = load_config('configs/exp_B001.yaml')
    path = tmp_path / 'research.pt'
    torch.save({'config': c.to_dict(), 'model': {}}, path)
    with pytest.raises(ValueError, match='研究'):
        run_infer.main(['--config', 'configs/exp_B001.yaml', '--checkpoint', str(path)])
    with pytest.raises(ValueError, match='研究'):
        run_ensemble.main(['--checkpoint', str(path), '--split', 'test'])


def test_cosine_resume_rejects_different_epochs():
    c = cfg()
    assert c.train.lr_scheduler == 'cosine'
    with pytest.raises(ValueError, match='train.epochs'):
        run_train._validate_resume_config(replace(c, train=replace(c.train, epochs=c.train.epochs + 1)), c)


def test_rng_checkpoint_restores_all_cpu_generators():
    random.seed(19)
    np.random.seed(19)
    torch.manual_seed(19)
    state = run_train._capture_rng_state()
    expected = (random.random(), np.random.random(), torch.rand(3))
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    run_train._restore_rng_state(state)
    actual = (random.random(), np.random.random(), torch.rand(3))
    assert actual[:2] == expected[:2]
    assert torch.equal(actual[2], expected[2])


def test_resumed_xrv_metadata_keeps_original_source():
    c = load_config('configs/exp_B001.yaml')
    backbone = torch.nn.Module()
    backbone.classifier = torch.nn.Linear(1, 1)
    model = XrvDenseNetClassifier(backbone, 10)
    previous = {'pretrained_loaded': True, 'weights_name': 'original', 'weights_path': '/weights', 'weights_sha256': 'abc', 'arch': 'xrv_densenet121'}
    result = run_train._checkpoint_metadata(model, c, torch.device('cpu'), True, previous)
    for key, value in previous.items():
        assert result[key] == value


def test_xrv_16bit_file_retains_values_and_validates_max(tmp_path):
    values = np.array([[0, 1000], [32768, 65535]], dtype=np.uint16)
    path = tmp_path / 'image.png'
    Image.fromarray(values).save(path)
    image = load_image(path, convert_rgb=False)
    actual = build_eval_transform(2, True, 'xrv', 1, 65535)(image)
    np.testing.assert_allclose(actual.numpy()[0], (values.astype(np.float32) / 65535 * 2 - 1) * 1024, atol=0.01)
    with pytest.raises(ValueError, match='xrv_maxval'):
        build_eval_transform(2, True, 'xrv', 1, 255)(image)


def test_interrupted_training_matches_continuous_cpu(tmp_path, labeled_split, write_config, monkeypatch):
    from common.checkpoint import load_checkpoint
    from common.utils import seed_everything

    split = (labeled_split['csv'], labeled_split['images'])
    config_path = write_config({'train': split, 'valid': split, 'test': split})
    c = load_config(config_path)
    c = replace(c, model=replace(c.model, pretrained=False, dropout=0.2),
                train=replace(c.train, epochs=2, warmup_epochs=0, num_workers=0, early_stop_patience=0))
    full_dir, interrupted_dir = tmp_path / 'full', tmp_path / 'interrupted'
    full_dir.mkdir()
    interrupted_dir.mkdir()
    seed_everything(c.seed)
    full = run_train.run_training(c, torch.device('cpu'), full_dir)
    original_fit = run_train.fit

    class Interrupted(Exception):
        pass

    def interrupt_fit(*args, on_epoch_end, **kwargs):
        def stop(progress, runtime):
            on_epoch_end(progress, runtime)
            if progress.epoch == 1:
                raise Interrupted
        return original_fit(*args, on_epoch_end=stop, **kwargs)

    seed_everything(c.seed)
    monkeypatch.setattr(run_train, 'fit', interrupt_fit)
    with pytest.raises(Interrupted):
        run_train.run_training(c, torch.device('cpu'), interrupted_dir)
    monkeypatch.setattr(run_train, 'fit', original_fit)
    checkpoint = load_checkpoint(interrupted_dir / 'last.pt')
    seed_everything(c.seed)
    resumed = run_train.run_training(c, torch.device('cpu'), interrupted_dir, checkpoint=checkpoint)
    for a, b in zip(full['history'], resumed['history']):
        assert a == pytest.approx(b, nan_ok=True)
    full_state = load_checkpoint(full_dir / 'last.pt')['model']
    resumed_state = load_checkpoint(interrupted_dir / 'last.pt')['model']
    assert all(torch.equal(full_state[k], resumed_state[k]) for k in full_state)


def test_16bit_xrv_dataset_uses_original_pixels(tmp_path):
    from data.dataset import StudyDataset
    from data.index import StudyRecord
    from data.transforms import build_train_transform
    from common.config import ImageConfig
    values = np.array([[0, 1000], [32768, 65535]], dtype=np.uint16)
    path = tmp_path / 'image.png'
    Image.fromarray(values).save(path)
    record = StudyRecord(study_id='s', subject_id='p', image_paths=(path,), labels=None)
    transform = build_eval_transform(2, True, 'xrv', 1, 65535)
    image = StudyDataset([record], transform, preserve_mode=True)[0]['images'][0, 0]
    np.testing.assert_allclose(image.numpy(), (values.astype(np.float32) / 65535 * 2 - 1) * 1024, atol=0.01)
    train = build_train_transform(8, ImageConfig(size=8, normalize='xrv', in_channels=1, xrv_maxval=65535))
    output = train(load_image(path, convert_rgb=False))
    assert output.shape == (1, 8, 8)
    assert torch.isfinite(output).all()
