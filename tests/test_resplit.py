"""合并后的患者划分与运行清单契约。"""
from dataclasses import replace

import pytest

from common.config import load_config, ResplitConfig
from data.dataset import build_loader
from data.resplit import prepare_resplit


@pytest.fixture
def config(tmp_path, make_image, write_csv, write_config):
    splits = {}
    for name, subjects in [('train', range(8)), ('valid', range(8, 12))]:
        rows = []
        for i in subjects:
            for j in range(2 if i == 0 else 1):
                dicom = f'd{i:02}_{j}'
                make_image(f'{name}/{dicom}.png')
                rows.append([dicom, f'p{i:02}', f's{i:02}_{j}', 'AP', '', '48', '64', '1' if i % 2 else '2'])
        splits[name] = (write_csv(f'{name}.csv', rows), tmp_path / name)
    splits['test'] = splits['valid']
    cfg = load_config(write_config(splits))
    return replace(cfg, data=replace(cfg.data, resplit=ResplitConfig(enabled=True, valid_fraction=0.25, seed=42)))


def test_resplit_keeps_patients_together_and_reuses_manifest(config, tmp_path):
    path = tmp_path / 'split.json'
    prepare_resplit(config, path)
    train = build_loader(config, 'train', train=True, manifest_path=path).dataset.records
    valid = build_loader(config, 'valid', manifest_path=path).dataset.records
    assert len({r.subject_id for r in valid}) == 3
    assert len({r.subject_id for r in train}) == 9
    assert not ({r.subject_id for r in train} & {r.subject_id for r in valid})
    assert len(train) + len(valid) == 13
    assert sum(r.subject_id == 'p00' for r in train) in (0, 2)
    assert sum(r.subject_id == 'p00' for r in valid) in (0, 2)
    first = path.read_text()
    prepare_resplit(config, path, resume=True)
    assert path.read_text() == first


def test_resplit_detects_changed_labels(config, tmp_path):
    path = tmp_path / 'split.json'
    prepare_resplit(config, path)
    csv = config.data.train.csv
    csv.write_text(csv.read_text().replace(',2\n', ',3\n', 1))
    with pytest.raises(ValueError, match='划分|数据'):
        build_loader(config, 'valid', manifest_path=path)


def test_resplit_refuses_missing_manifest_on_resume(config, tmp_path):
    with pytest.raises(FileNotFoundError):
        prepare_resplit(config, tmp_path / 'missing.json', resume=True)


def test_resplit_rejects_invalid_fraction():
    with pytest.raises(ValueError):
        ResplitConfig(enabled=True, valid_fraction=1.0)


def test_checkpoint_requires_matching_manifest(config, tmp_path):
    from data.resplit import checkpoint_manifest_path, manifest_digest

    path = tmp_path / 'split_manifest.json'
    prepare_resplit(config, path)
    checkpoint = {'config': config.to_dict(), 'metadata': {'split_manifest_sha256': manifest_digest(path)}}
    assert checkpoint_manifest_path(config, checkpoint, tmp_path / 'best.pt') == path
    path.write_text(path.read_text() + ' ')
    with pytest.raises(ValueError, match='SHA-256'):
        checkpoint_manifest_path(config, checkpoint, tmp_path / 'best.pt')


def test_cross_source_patient_overlap_is_rejected(config, tmp_path):
    csv = config.data.valid.csv
    csv.write_text(csv.read_text().replace('p08', 'p00'))
    with pytest.raises(ValueError, match='subject_id'):
        prepare_resplit(config, tmp_path / 'split.json')


def test_training_writes_bound_checkpoints_and_resume_reuses_split(config, tmp_path):
    import torch
    from common.checkpoint import load_checkpoint
    from data.resplit import manifest_digest
    from run_train import run_training

    config = replace(config, model=replace(config.model, pretrained=False),
                     train=replace(config.train, epochs=1, warmup_epochs=0, batch_studies=9,
                                   early_stop_patience=0, amp=False))
    run_training(config, torch.device('cpu'), tmp_path)
    digest = manifest_digest(tmp_path / 'split_manifest.json')
    last = load_checkpoint(tmp_path / 'last.pt')
    best = load_checkpoint(tmp_path / 'best.pt')
    assert last['metadata']['split_manifest_sha256'] == digest
    assert best['metadata']['split_manifest_sha256'] == digest
    run_training(config, torch.device('cpu'), tmp_path, checkpoint=last)
    assert manifest_digest(tmp_path / 'split_manifest.json') == digest
