"""合成样本仅用于单元测试；正式验收使用 check_data 中的真实数据。"""
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

from src.data import ChestXrayDataset, index_images, make_loader, parse_labels, preprocess, read_csv


class DataTests(unittest.TestCase):
    def test_labels(self) -> None:
        self.assertEqual(parse_labels('7;9').tolist(), [0, 0, 0, 0, 0, 0, 1, 0, 1, 0])
        self.assertEqual(parse_labels('10').tolist(), [0] * 9 + [1])
        for value in ('', '-1', 'None', '0', '11', '7;', '7;7', None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_labels(value)

    def test_aspect_ratio(self) -> None:
        tensor = preprocess(Image.new('L', (100, 50), 255), 100)
        self.assertEqual(tuple(tensor.shape), (3, 100, 100))
        self.assertEqual(tensor[:, :25].sum().item(), 0)
        self.assertTrue((tensor[:, 25:75] == 1).all())
        self.assertEqual(tensor[:, 75:].sum().item(), 0)

    def test_dataset_and_validation(self) -> None:
        # 将临时夹限制在已授权的测试目录内，避开 Windows 沙箱系统临时目录权限问题。
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            rows = [{'Subject_id': '001', 'Study_id': '002', 'Dicom_id': name,
                     'Predict_class': '7;9'} for name in ('003', '004')]
            csv_path = root / 'labels.csv'
            pd.DataFrame(rows).to_csv(csv_path, index=False, encoding='utf-8')
            for row in rows:
                Image.new('L', (20, 10)).save(root / (row['Dicom_id'] + '.png'))
            dataset = ChestXrayDataset(csv_path, root)
            self.assertEqual(dataset.frame['Study_id'].nunique(), 1)
            self.assertEqual(dataset[0]['Subject_id'], '001')
            batch = next(iter(make_loader(dataset, 2)))
            self.assertEqual(tuple(batch['label'].shape), (2, 10))
            pd.DataFrame(rows).drop(columns='Predict_class').to_csv(csv_path, index=False, encoding='utf-8')
            self.assertNotIn('label', ChestXrayDataset(csv_path, root, labeled=False)[0])
            rows[1]['Predict_class'] = '10'
            pd.DataFrame(rows).to_csv(csv_path, index=False, encoding='utf-8')
            with self.assertRaises(ValueError):
                read_csv(csv_path, True)
            rows[1]['Predict_class'] = '9;7'
            pd.DataFrame(rows).to_csv(csv_path, index=False, encoding='utf-8')
            self.assertEqual(len(read_csv(csv_path, True)), 2)
            (root / '004.png').unlink()
            with self.assertRaises(FileNotFoundError):
                ChestXrayDataset(csv_path, root)
            Image.new('L', (10, 10)).save(root / '003.jpg')
            with self.assertRaises(ValueError):
                index_images(root)


if __name__ == '__main__':
    unittest.main()
