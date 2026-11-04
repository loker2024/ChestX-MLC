"""官方验证模式的CPU模拟测试，不启动训练或GPU。"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from PIL import Image

from src.swinv2_official_data import read_official_paths, audit_official
from src.swinv2_experiment import load_config, SwinDataset
from src.run_swinv2 import prepare


class OfficialTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path('outputs').resolve())
        self.root = Path(self.tmp.name)
        self.paths = {}
        for split in ('train', 'valid'):
            folder = self.root / split
            folder.mkdir()
            rows = []
            for i, label in enumerate(('1;2;3;4;5', '6;7;8;9;10')):
                uid = split + str(i)
                Image.new('L', (40, 70), 120).save(folder / (uid+'.jpg'))
                rows.append(dict(Dicom_id=uid, Subject_id=uid, Study_id=uid, Predict_class=label))
            csv = self.root / (split+'.csv')
            pd.DataFrame(rows).to_csv(csv, index=False)
            self.paths['official_'+split+'_csv'] = csv
            self.paths[split+'_images'] = folder
        self.paths['pretrained_weight_path'] = self.root / 'not_loaded.pth'

    def tearDown(self):
        self.tmp.cleanup()

    def test_config(self):
        c = load_config('configs/experiments/X_SWINV2_01_official20.json')
        self.assertEqual((c['epochs'], c['num_workers'], c['validation_mode']), (20, 8, 'official_valid'))
        self.assertEqual(c['early_stopping']['patience'], 3)

    def test_audit_fixture(self):
        frames, audit = audit_official(self.paths, require_official=False)
        self.assertEqual(audit['statistics']['valid']['studies'], 2)
        self.assertTrue(audit['official_valid_used_for_selection'])
        self.assertFalse(audit['official_valid_in_training'])
        self.assertFalse(audit['test_accessed'])

    def test_fixture_rejected_as_official(self):
        with self.assertRaises(ValueError):
            audit_official(self.paths)

    def test_patient_leakage_rejected(self):
        f = pd.read_csv(self.paths['official_valid_csv'], dtype=str)
        f.loc[0, 'Subject_id'] = 'train0'
        f.to_csv(self.paths['official_valid_csv'], index=False)
        with self.assertRaisesRegex(ValueError, 'Subject_id'):
            audit_official(self.paths, require_official=False)

    def test_relative_paths_and_no_test(self):
        p = self.root/'paths.json'
        p.write_text(json.dumps({k:str(v.relative_to(self.root)) for k,v in self.paths.items()}))
        self.assertEqual(read_official_paths(p), self.paths)
        values = json.loads(p.read_text()); values['test_csv'] = 'test.csv'
        p.write_text(json.dumps(values))
        with self.assertRaises(ValueError): read_official_paths(p)

    def test_mode_routes_without_v2(self):
        c = load_config('configs/experiments/X_SWINV2_01_official20.json')
        with patch('src.swinv2_official_data.prepare_official', return_value='official') as official:
            with patch('src.run_swinv2.verify_split', side_effect=AssertionError('不得读取v2')):
                self.assertEqual(prepare(c, self.root/'paths.json'), 'official')
        official.assert_called_once()

    def test_separate_roots_shape_and_labels(self):
        import torch
        for split in ('train', 'valid'):
            ds = SwinDataset(self.paths['official_'+split+'_csv'], self.paths[split+'_images'],
                             size=384, normalization='imagenet', augment=split=='train')
            torch.manual_seed(42)
            item = ds[0]
            self.assertEqual(tuple(item['image'].shape), (3,384,384))
            self.assertEqual(tuple(item['label'].shape), (10,))
            self.assertEqual(item['Study_id'], split+'0')


if __name__ == '__main__': unittest.main()
