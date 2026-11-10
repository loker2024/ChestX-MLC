"""患者折清单的合成契约；不启动训练。"""
import pandas as pd
import pytest

from data.patient_folds import assign_patient_folds, study_table, sha256_file
from run_patient_oof import fixed_recipe, join_oof
from common.config import load_config


def metadata():
    return pd.DataFrame([{'Subject_id': f'{i:03}', 'Study_id': f'{i:04}',
                          'Dicom_id': f'{i:05}', 'Predict_class': '1;9'} for i in range(15)])


def test_patient_fold_deterministic_and_no_leakage():
    frame = pd.concat([metadata(), metadata().iloc[:2]], ignore_index=True)
    a = assign_patient_folds(study_table(frame), 5, 2026)
    b = assign_patient_folds(study_table(frame.sample(frac=1, random_state=1)), 5, 2026)
    assert a == b
    assert set(a.values()) == set(range(5))
    assert '0000' in a


def test_conflicting_patient_and_labels_rejected():
    frame = pd.concat([metadata(), metadata().iloc[:1]], ignore_index=True)
    frame.loc[len(frame)-1, 'Subject_id'] = 'other'
    with pytest.raises(ValueError):
        study_table(frame)
    frame.loc[len(frame)-1, 'Subject_id'] = '000'
    frame.loc[len(frame)-1, 'Predict_class'] = '2'
    with pytest.raises(ValueError):
        study_table(frame)


def test_fixed_recipe_does_not_modify_base():
    base = load_config('configs/exp_D002a.yaml')
    recipe = fixed_recipe(base)
    assert base.train.epochs == 8
    assert recipe.train.epochs == 5
    assert recipe.train.save_last and recipe.train.ema_enabled
    assert recipe.train.early_stop_patience == 0
    assert not recipe.data.full_train and not recipe.data.resplit.enabled
    assert recipe.seed == 42


def test_file_fingerprint_changes(tmp_path):
    path = tmp_path / 'source.csv'
    path.write_text('a', encoding='utf-8')
    before = sha256_file(path)
    path.write_text('b', encoding='utf-8')
    assert before != sha256_file(path)


def test_join_rejects_duplicate_and_wrong_fold():
    manifest = {'studies': [{'Subject_id': '001', 'Study_id': '0001', 'fold': 0,
                             'labels': [1] * 10}]}
    row = {'Subject_id': '001', 'Study_id': '0001', 'fold': 1,
           **{f'label_{i}': 1 for i in range(1, 11)},
           **{f'p{i}': .5 for i in range(1, 11)}}
    with pytest.raises(ValueError):
        join_oof([pd.DataFrame([row])], manifest)
    row['fold'] = 0
    with pytest.raises(ValueError):
        join_oof([pd.DataFrame([row, row])], manifest)


def test_prepare_nested_pools_exclude_outer_holdout(tmp_path):
    from dataclasses import replace
    import json
    import yaml
    from data.patient_folds import read_metadata, verify_manifest
    from run_patient_oof import prepare

    source = tmp_path / 'train.csv'
    metadata().to_csv(source, index=False, encoding='utf-8')
    base = load_config('configs/exp_D002a.yaml')
    base = replace(base, data=replace(base.data, train=replace(base.data.train, csv=source)))
    config = tmp_path / 'model.yaml'
    config.write_text(yaml.safe_dump(base.to_dict()), encoding='utf-8')
    root = tmp_path / 'outputs'
    path = prepare(config, root)
    top = json.loads(path.read_text(encoding='utf-8'))
    verify_manifest(top)
    assert len(top['studies']) == 15
    for outer in top['outer_folds']:
        inner_path = root / f"outer_{outer['fold']}" / 'inner_manifest.json'
        inner = json.loads(inner_path.read_text(encoding='utf-8'))
        verify_manifest(inner)
        assert not ({row['Subject_id'] for row in inner['studies']} & set(outer['valid_subjects']))
        assert inner['n_splits'] == 3
        assert read_metadata(outer['valid_csv']).iloc[0]['Subject_id'].startswith('0')
    with pytest.raises(ValueError):
        prepare(config, root)
    source.write_text('tampered', encoding='utf-8')
    with pytest.raises(ValueError):
        verify_manifest(top)


def test_run_manifest_option():
    from run_patient_oof import build_parser
    args = build_parser().parse_args(['run', '--manifest', '/tmp/manifest.json', '--stage', 'outer', '--fold', '0'])
    assert args.manifest == '/tmp/manifest.json'
    assert args.fold == 0 and args.stage == 'outer'
