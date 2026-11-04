"""服务器20轮配置回归；不启动GPU或完整训练。"""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from src.swinv2_experiment import load_config, rates
from src.early_stopping import EarlyStopping


class Server20Tests(unittest.TestCase):
    def test_server_config_and_schedule(self):
        c = load_config('configs/experiments/X_SWINV2_01_server20.json')
        self.assertEqual(c['epochs'], 20)
        self.assertEqual(c['num_workers'], 8)
        self.assertEqual(c['early_stopping']['patience'], 3)
        self.assertEqual(c['amp_dtype'], 'bfloat16')
        self.assertEqual(rates(c, 2), {'backbone': 2e-5, 'head': 2e-4})
        self.assertEqual(rates(c, 20), {'backbone': 1e-6, 'head': 1e-6})
        self.assertGreater(rates(c, 12)['backbone'], 1e-6)

    def test_legacy_12_preserved(self):
        c = load_config('configs/experiments/X_SWINV2_01.json')
        self.assertEqual(c['epochs'], 12)
        self.assertEqual(rates(c, 12)['head'], 1e-6)

    def test_early_stop_three_bad_epochs(self):
        stop = EarlyStopping(3, .0001)
        for epoch, score in enumerate([.8, .79, .78, .77], 1):
            stop.step(score, epoch)
            self.assertEqual(stop.should_stop, epoch == 4)

    def test_negative_workers_rejected(self):
        c = copy.deepcopy(load_config('configs/experiments/X_SWINV2_01_server20.json'))
        c['num_workers'] = -1
        with tempfile.TemporaryDirectory(dir=Path('outputs').resolve()) as d:
            p = Path(d) / 'bad.json'
            p.write_text(json.dumps(c), encoding='utf-8')
            with self.assertRaises(ValueError):
                load_config(p)


if __name__ == '__main__':
    unittest.main()
