import json
import os
from pathlib import Path
import tempfile
import unittest
import pandas as pd
from src.experiments import register, sha256
from src.continue_train import run


class ExperimentTests(unittest.TestCase):
    def test_snapshot_preserves_source_and_refuses_overwrite(self) -> None:
        original = Path.cwd()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            try:
                os.chdir(directory)
                source = Path('source')
                source.mkdir()
                (source/'best_model.pth').write_bytes(b'unit-test-only')
                (source/'train_config.json').write_text(json.dumps({'size':224,'learning_rate':0.001,'batch_size':16}),encoding='utf-8')
                (source/'validation_metrics.json').write_text(json.dumps({'epoch':1,'macro_auc':0.7,'macro_f1':0.2}),encoding='utf-8')
                (source/'history.csv').write_text('epoch\n1\n',encoding='utf-8')
                pd.DataFrame([{'Subject_id':'01','Study_id':'001',**{f'p{i}':0.2 for i in range(1,11)}}]).to_csv(source/'val_study_predictions.csv',index=False,encoding='utf-8')
                before = sha256(source/'best_model.pth')
                folder = register(source,'exp_001','test')
                self.assertEqual(sha256(folder/'best_model.pth'),before)
                self.assertEqual(sha256(source/'best_model.pth'),before)
                self.assertEqual(pd.read_csv('outputs/experiment_summary.csv').iloc[0].experiment_id,'exp_001')
                with self.assertRaises(FileExistsError):
                    register(source,'exp_001','overwrite')
                with self.assertRaises(FileExistsError):
                    run(source,folder)
            finally:
                os.chdir(original)


if __name__ == '__main__':
    unittest.main()
