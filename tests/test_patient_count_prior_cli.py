"""离线次数先验的输入边界与固定十类统计契约。未授权执行。"""
import numpy as np
import pandas as pd
import pytest

from infer.patient_count_analysis import choose_beta, metric_report, paired_bootstrap
from run_patient_count_prior import build_parser, require_research_split, output_directory


def test_test_research_is_rejected():
    with pytest.raises(ValueError, match="test"):
        require_research_split("test")


def test_output_directory_refuses_existing_results(tmp_path):
    (tmp_path / "report.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="非空"):
        output_directory(tmp_path)


def test_beta_tie_prefers_smaller_and_no_gain_returns_zero():
    assert choose_beta({0.: .7, .05: .7000005, .1: .699, .2: .69}) == 0.
    assert choose_beta({0.: .7, .05: .71, .1: .7100005, .2: .7}) == .05


def test_degenerate_class_does_not_become_fixed_ten_class_auc():
    y = np.array([[0] * 10, [1] * 9 + [0]])
    report = metric_report(y, np.full((2, 10), .5))
    assert report["macro_auc_10"] is None
    assert report["valid_classes"] == list(range(1, 10))


def test_bootstrap_invalid_samples_block_promotion():
    frame = pd.DataFrame({"fold": [0, 0], "Subject_id": ["01", "02"]})
    y = np.zeros((2, 10))
    report = paired_bootstrap(frame, y, np.full_like(y, .5), np.full_like(y, .5))
    assert report["invalid_fraction"] == 1.
    assert report["ci_reliable"] is False


def test_cli_requires_explicit_inputs():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["select"])


def historical_inputs(tmp_path):
    """多图、重复患者、打乱的 study 顺序；不依赖影像或 OOF。"""
    metadata = tmp_path / 'valid.csv'
    metadata.write_text('Subject_id,Study_id,Predict_class\n'
                        '001,0001,1\n001,0001,1\n001,0002,10\n002,0003,10\n', encoding='utf-8')
    predictions = tmp_path / 'predictions.csv'
    pd.DataFrame({'Study_id': ['0003', '0002', '0001'],
                  **{f'p{i}': [.5, .5, .5] for i in range(1, 11)}}).to_csv(predictions, index=False)
    return metadata, predictions


def test_explore_reads_old_predictions_without_prior_or_oof(tmp_path):
    """抓住要求预先选参、按行错配、计数包含图片重复及覆盖原概率。"""
    import json
    from run_patient_count_prior import main
    metadata, predictions = historical_inputs(tmp_path)
    original = predictions.read_bytes()
    out = tmp_path / 'exploration'
    main(['explore', '--predictions', str(predictions), '--metadata', str(metadata), '--out', str(out)])
    counts = pd.read_csv(out / 'patient_counts.csv', dtype={'Subject_id': str, 'Study_id': str})
    assert counts.set_index('Study_id').patient_study_count.to_dict() == {'0001': 2, '0002': 2, '0003': 1}
    zero = pd.read_csv(out / 'adjusted_beta_0.csv', dtype={'Study_id': str})
    adjusted = pd.read_csv(out / 'adjusted_beta_0.2.csv', dtype={'Study_id': str})
    assert adjusted.Study_id.tolist() == ['0003', '0002', '0001']
    assert zero.p1.tolist() == [.5, .5, .5]
    assert adjusted.p1.iloc[0] == .5
    assert adjusted.p1.iloc[1] > .5 and adjusted.p10.iloc[1] < .5
    assert adjusted.label_1.tolist() == [0, 0, 1]
    assert predictions.read_bytes() == original
    report = json.loads((out / 'report.json').read_text(encoding='utf-8'))
    assert report['evidence'] == 'historical_valid_exploration'
    assert report['promotion'] is False
    assert len(report['candidates']) == 4
    assert report['candidates']['0.0']['delta_f1'] == 0.
    assert len(pd.read_csv(out / 'summary.csv')) == 4
    assert not (out / 'patient_count_prior.json').exists()


@pytest.mark.parametrize('problem', ['label', 'duplicate', 'missing', 'probability', 'subject', 'partial_labels'])
def test_explore_rejects_mismatched_historical_inputs(tmp_path, problem):
    from run_patient_count_prior import main
    metadata, predictions = historical_inputs(tmp_path)
    frame = pd.read_csv(predictions, dtype={'Study_id': str})
    if problem == 'label':
        for i in range(1, 11):
            frame[f'label_{i}'] = 0
    elif problem == 'partial_labels':
        frame['label_1'] = 0
    elif problem == 'duplicate':
        frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    elif problem == 'missing':
        frame = frame.iloc[:-1]
    elif problem == 'probability':
        frame.loc[0, 'p1'] = float('inf')
    elif problem == 'subject':
        frame['Subject_id'] = ['999', '001', '001']
    frame.to_csv(predictions, index=False)
    with pytest.raises(ValueError):
        main(['explore', '--predictions', str(predictions), '--metadata', str(metadata),
              '--out', str(tmp_path / 'result')])


def test_explore_rejects_test_before_reading_inputs(tmp_path):
    from run_patient_count_prior import main
    out = tmp_path / 'result'
    with pytest.raises(ValueError, match='test'):
        main(['explore', '--split', 'test', '--predictions', 'missing.csv',
              '--metadata', 'missing.csv', '--out', str(out)])
    assert not out.exists()


def test_oof_producer_and_selector_share_verified_contract(tmp_path):
    """抓住 producer/selector 字段错配，以及无来源预测被接受。"""
    from dataclasses import replace
    import json
    from pathlib import Path
    import yaml
    from common.config import load_config
    from data.patient_folds import sha256_file
    from run_patient_oof import prepare
    from run_patient_count_prior import main, validate_oof
    from infer.patient_count_prior import load_prior

    source = tmp_path / 'train.csv'
    pd.DataFrame([{'Subject_id': f'{i:03}', 'Study_id': f'{i:04}', 'Dicom_id': f'{i:05}',
                   'Predict_class': '1;9' if i % 2 else '10'} for i in range(15)]).to_csv(source, index=False)
    base = load_config('configs/exp_D002a.yaml')
    base = replace(base, data=replace(base.data, train=replace(base.data.train, csv=source)))
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump(base.to_dict()), encoding='utf-8')
    manifest_path = prepare(config, tmp_path / 'folds')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    rows = [{**{key: row[key] for key in ('Subject_id', 'Study_id', 'fold')},
             **{f'label_{i+1}': row['labels'][i] for i in range(10)},
             **{f'p{i}': .5 for i in range(1, 11)}} for row in manifest['studies']]
    oof = tmp_path / 'oof.csv'
    pd.DataFrame(rows).to_csv(oof, index=False)
    # 没有逐折的原始预测与来源记录，即使 ID/标签全匹配也必须失败。
    with pytest.raises(FileNotFoundError):
        validate_oof(str(oof), str(manifest_path))
    for entry in manifest['outer_folds']:
        folder = Path(entry['output_dir'])
        predictions = folder / 'predictions.csv'
        selected = [row for row in rows if row['fold'] == entry['fold']]
        pd.DataFrame(selected).to_csv(predictions, index=False)
        # 本用例只检查字节摘要契约，不加载或伪装真实 PyTorch 权重。
        last, ema = folder / 'last.bin', folder / 'ema.bin'
        last.write_bytes(b'synthetic last checkpoint')
        ema.write_bytes(b'synthetic EMA checkpoint')
        receipt = {'manifest_sha256': sha256_file(manifest_path), 'config_sha256': entry['config_sha256'],
                   'recipe_sha256': manifest['recipe_sha256'], 'fold': entry['fold'],
                   'train_subjects': entry['train_subjects'], 'valid_subjects': entry['valid_subjects'],
                   'epoch': 5, 'weights_kind': 'fixed_last_ema', 'initialization': 'imagenet',
                   'last_checkpoint': str(last), 'ema_checkpoint': str(ema), 'predictions': str(predictions)}
        for key in ('last_checkpoint', 'ema_checkpoint', 'predictions'):
            receipt[f'{key}_sha256'] = sha256_file(receipt[key])
        (folder / 'provenance.json').write_text(json.dumps(receipt), encoding='utf-8')
    main(['select', '--config', 'configs/patient_count_prior.yaml', '--oof', str(oof),
          '--manifest', str(manifest_path), '--out', str(tmp_path / 'selection')])
    prior = load_prior(tmp_path / 'selection/patient_count_prior.json')
    assert prior.beta == 0.
    assert prior.recipe_sha256 == manifest['recipe_sha256']
    assert prior.provenance['oof_sha256'] == sha256_file(oof)
    damaged = pd.read_csv(oof, dtype={'Subject_id': str, 'Study_id': str, 'fold': str})
    damaged.loc[0, 'p1'] = .8
    damaged.to_csv(oof, index=False)
    with pytest.raises(ValueError, match='概率'):
        validate_oof(str(oof), str(manifest_path))
