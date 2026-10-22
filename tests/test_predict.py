import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import pandas as pd
from PIL import Image
import torch

from src.devices import select_device, synchronize, peak_memory
from src.model import build_model
from src.train import save_model
from src.predict import run, prediction_config
from src.experiments import sha256
from src.submission import PROB_COLUMNS


class PredictionTests(unittest.TestCase):
    def test_device_fallback(self) -> None:
        with patch('torch.cuda.is_available',return_value=False), patch('torch.backends.mps.is_available',return_value=True):
            self.assertEqual(select_device(),'mps')
            self.assertEqual(select_device('cpu'),'cpu')
        with patch('torch.cuda.is_available',return_value=False), patch('torch.backends.mps.is_available',return_value=False):
            self.assertEqual(select_device(),'cpu')
            with self.assertRaises(ValueError):
                select_device('cuda')
        with patch('torch.cuda.synchronize') as cuda_sync:
            synchronize('cpu')
            self.assertIsNone(peak_memory('cpu')['peak_allocated_bytes'])
            cuda_sync.assert_not_called()

    def test_epoch_checkpoint_predicts_without_companion_config(self) -> None:
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            root = Path(folder)
            rows = []
            for study in range(12):
                for image in range(2):
                    image_id = f'{study:03d}_{image}'
                    Image.new('L',(40+image,50),100+study).save(root/(image_id+'.png'))
                    rows.append({'Subject_id':f'{study:03d}', 'Study_id':f'{study:04d}',
                                 'Dicom_id':image_id,'Predict_class':str(study%10+1)})
            frame = pd.DataFrame(rows)
            frame.to_csv(root/'valid.csv',index=False,encoding='utf-8')
            frame.drop(columns='Predict_class').to_csv(root/'test.csv',index=False,encoding='utf-8')
            config = {'size':64,'batch_size':4,'seed':42,'preprocess':'RGB letterbox /255',
                      'threshold':0.5,'aggregation':'mean image probabilities per Study_id',
                      'data':{split:{'csv':str(root/(split+'.csv')),'images':str(root)} for split in ('valid','test')}}
            checkpoint = root/'model.pth'
            # 模拟同学拿到来自旧电脑的模型，内嵌路径在当前机器不可用。
            embedded = {**config,'data':{split:{'csv':'old-machine/'+split+'.csv','images':'old-machine/images'} for split in ('valid','test')}}
            save_model(build_model(),checkpoint,embedded)
            paths_override = root/'data.json'
            paths_override.write_text(json.dumps({'size':64,'valid':config['data']['valid'],'test':config['data']['test']}),encoding='utf-8')
            self.assertEqual(prediction_config(checkpoint)[1],'checkpoint.metadata')
            # 历史指标故意不同，默认应告警并继续；严格模式应失败。
            previous = frame[['Subject_id','Study_id']].drop_duplicates().assign(**dict.fromkeys(PROB_COLUMNS,0.1))
            previous.to_csv(root/'val_study_predictions.csv',index=False,encoding='utf-8')
            (root/'validation_metrics.json').write_text(json.dumps({'macro_auc':0.01,'macro_f1':0.01}),encoding='utf-8')
            before = sha256(checkpoint)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                output = run(checkpoint,'both','all_classes',device_name='cpu',allow_nonofficial_counts=True,data_config=paths_override)
            self.assertTrue(any('验证复测与历史' in str(w.message) for w in caught))
            metrics = json.loads((output/'validation_metrics.json').read_text(encoding='utf-8'))
            self.assertFalse(metrics['reference_matches'])
            self.assertEqual(len(pd.read_csv(output/'submission.csv')),120)
            self.assertEqual(sha256(checkpoint),before)
            with self.assertRaises(RuntimeError):
                run(checkpoint,'valid',strict_reproduce=True,device_name='cpu',allow_nonofficial_counts=True,data_config=paths_override)
            with self.assertRaises(ValueError):
                run(checkpoint,'test')


if __name__ == '__main__':
    unittest.main()
