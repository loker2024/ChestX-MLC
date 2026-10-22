"""隔离复现换机路径、坏参考、登记失败和同划分排名。"""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import pandas as pd

from src.experiments import register, rebuild_summary
from src.predict import prediction_config, compare_reference
from src.fingerprints import dataset_fingerprint, EVALUATION_PROTOCOL
from src.submission import PROB_COLUMNS


def fixture_source(source: Path, auc: float = 0.7, fingerprint: str | None = None) -> None:
    source.mkdir()
    (source/'best_model.pth').write_bytes(b'unit-fixture-only')
    (source/'train_config.json').write_text(json.dumps({'size':224,'batch_size':16,'learning_rate':0.001}),encoding='utf-8')
    metrics = {'epoch':1,'macro_auc':auc,'macro_f1':0.2,'study_count':1}
    if fingerprint:
        metrics.update(validation_fingerprint=fingerprint,evaluation_protocol=EVALUATION_PROTOCOL)
    (source/'validation_metrics.json').write_text(json.dumps(metrics),encoding='utf-8')
    (source/'history.csv').write_text('epoch\n1\n',encoding='utf-8')
    pd.DataFrame([{'Subject_id':'01','Study_id':'001',**dict.fromkeys(PROB_COLUMNS,0.2)}]).to_csv(source/'val_study_predictions.csv',index=False,encoding='utf-8')


class ReliabilityTests(unittest.TestCase):
    def test_data_path_override_preserves_preprocessing(self) -> None:
        config = {'size':224,'batch_size':16,'seed':42,'preprocess':'RGB letterbox /255',
                  'data':{'test':{'csv':'old/test.csv','images':'old/images'}}}
        original = copy.deepcopy(config)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            path = Path(temporary)/'data.json'
            path.write_text(json.dumps({'size':224,'batch_size':4,'test':{'csv':'new/test.csv','images':'new/images'}}),encoding='utf-8')
            with patch('src.predict.torch.load',return_value={'metadata':config}):
                actual,_ = prediction_config(Path('model.pth'),path)
                self.assertEqual(actual['data']['test']['csv'],'new/test.csv')
                self.assertEqual(actual['batch_size'],16)
                self.assertEqual(config,original)
                path.write_text('{"size":320}',encoding='utf-8')
                with self.assertRaises(ValueError):
                    prediction_config(Path('model.pth'),path)

    def test_bad_optional_references_warn_instead_of_blocking(self) -> None:
        studies = pd.DataFrame([{'Subject_id':'01','Study_id':'001',**dict.fromkeys(PROB_COLUMNS,0.2)}])
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            root = Path(temporary)
            for problem in ('duplicate','bad_json','missing_column','nan_metric'):
                with self.subTest(problem=problem):
                    reference = {'macro_auc':float('nan') if problem == 'nan_metric' else .7,'macro_f1':.2}
                    (root/'validation_metrics.json').write_text('{bad' if problem == 'bad_json' else json.dumps(reference),encoding='utf-8')
                    previous = pd.concat([studies,studies]) if problem == 'duplicate' else studies
                    if problem == 'missing_column':
                        previous = previous.drop(columns='p10')
                    previous.to_csv(root/'val_study_predictions.csv',index=False,encoding='utf-8')
                    metrics = {'macro_auc':.7,'macro_f1':.2}
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always')
                        compare_reference(studies,metrics,root,False,1e-4,1e-4)
                    self.assertEqual(metrics['reference_comparison'],'invalid')
                    self.assertTrue(caught)
                    with self.assertRaises(RuntimeError):
                        compare_reference(studies,{'macro_auc':.7,'macro_f1':.2},root,True,1e-4,1e-4)

    def test_registration_failure_is_retryable_and_does_not_poison_summary(self) -> None:
        original = Path.cwd()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            try:
                os.chdir(temporary)
                source = Path('source')
                fixture_source(source)
                metrics = (source/'validation_metrics.json').read_text(encoding='utf-8')
                (source/'validation_metrics.json').write_text('{bad',encoding='utf-8')
                with self.assertRaises(ValueError):
                    register(source,'exp_001','invalid')
                self.assertFalse(Path('outputs/experiments/exp_001').exists())
                (source/'validation_metrics.json').write_text(metrics,encoding='utf-8')
                with patch('src.experiments.shutil.copy2',side_effect=OSError('simulated disk error')):
                    with self.assertRaises(OSError):
                        register(source,'exp_001','copy failure')
                self.assertFalse(Path('outputs/experiments/exp_001').exists())
                self.assertEqual(list(Path('outputs/experiments').glob('.staging-*')),[])
                register(source,'exp_001','fixed')
                broken = Path('outputs/experiments/exp_broken')
                broken.mkdir()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    register(source,'exp_002','valid next registration')
                self.assertTrue(caught)
                self.assertEqual(len(pd.read_csv('outputs/experiment_summary.csv')),2)
            finally:
                os.chdir(original)

    def test_empty_summary_and_separate_comparison_groups(self) -> None:
        original = Path.cwd()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            try:
                os.chdir(temporary)
                rebuild_summary()
                self.assertTrue(pd.read_csv('outputs/experiment_summary.csv').empty)
                for index,(auc,fingerprint) in enumerate([(.7,'a'*64),(.8,'a'*64),(.99,'b'*64),(.9,None)],1):
                    source = Path('source'+str(index))
                    fixture_source(source,auc,fingerprint)
                    register(source,f'exp_{index:03d}','test grouping')
                summary = pd.read_csv('outputs/experiment_summary.csv').set_index('experiment_id')
                self.assertEqual(summary.loc['exp_001','rank_in_group'],2)
                self.assertEqual(summary.loc['exp_002','rank_in_group'],1)
                self.assertEqual(summary.loc['exp_003','rank_in_group'],1)
                self.assertTrue(pd.isna(summary.loc['exp_004','rank_in_group']))
            finally:
                os.chdir(original)

    def test_fingerprint_tracks_labels_not_csv_order(self) -> None:
        frame = pd.DataFrame({'Subject_id':['01','02'],'Study_id':['001','002'],
                              'Dicom_id':['a','b'],'Predict_class':['7;9','10']})
        reordered = frame.iloc[::-1].copy()
        reordered.loc[0,'Predict_class'] = '9;7'
        self.assertEqual(dataset_fingerprint(frame),dataset_fingerprint(reordered))
        reordered.loc[0,'Predict_class'] = '9'
        self.assertNotEqual(dataset_fingerprint(frame),dataset_fingerprint(reordered))


if __name__ == '__main__':
    unittest.main()
