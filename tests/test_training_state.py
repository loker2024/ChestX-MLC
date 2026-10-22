import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
import numpy as np
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader,TensorDataset
from src.training_state import save_training_state,load_training_state,restore_training_state


class TrainingStateTests(unittest.TestCase):
    def test_training_entry_resume_matches_uninterrupted_resnet(self) -> None:
        from src.full_train import run as first_epoch
        from src.continue_train import run as continuation
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            root=Path(temporary)
            config={'size':64,'batch_size':2}
            for split in ('train','valid'):
                images=root/split
                images.mkdir()
                rows=[]
                for study in range(2):
                    for number in range(2):
                        identifier=f'{split}_{study}_{number}'
                        Image.new('L',(50,60),80+study*100).save(images/(identifier+'.png'))
                        rows.append({'Subject_id':f'{split}_{study}','Study_id':f'{split}_s{study}',
                                     'Dicom_id':identifier,'Predict_class':'1;2;3;4;5;6;7;8;9' if study==0 else '10'})
                csv=root/(split+'.csv')
                pd.DataFrame(rows).to_csv(csv,index=False,encoding='utf-8')
                config[split]={'csv':str(csv),'images':str(images)}
            path=root/'data.json'
            path.write_text(json.dumps(config),encoding='utf-8')
            baseline=root/'baseline'
            first_epoch('cpu',path,baseline,7,True)
            direct=root/'direct'
            interrupted=root/'interrupted'
            resumed=root/'resumed'
            continuation(baseline,direct,2,device_name='cpu',resume_state=baseline/'training_state.pth',allow_nonofficial_counts=True)
            continuation(baseline,interrupted,1,device_name='cpu',resume_state=baseline/'training_state.pth',allow_nonofficial_counts=True)
            continuation(interrupted,resumed,1,device_name='cpu',resume_state=interrupted/'epoch_2/training_state.pth',allow_nonofficial_counts=True)
            expected=torch.load(direct/'epoch_3/model.pth',weights_only=True,map_location='cpu')['state_dict']
            actual=torch.load(resumed/'epoch_3/model.pth',weights_only=True,map_location='cpu')['state_dict']
            for key,value in expected.items():
                torch.testing.assert_close(value,actual[key],rtol=0,atol=0)

    def test_resume_matches_uninterrupted_next_epoch(self) -> None:
        torch.set_num_threads(2)
        torch.manual_seed(7)
        random.seed(7)
        np.random.seed(7)
        def model_factory():
            return torch.nn.Sequential(torch.nn.Linear(4,8),torch.nn.Dropout(.2),torch.nn.Linear(8,10))
        dataset = TensorDataset(torch.randn(6,4),torch.randint(0,2,(6,10)).float(),torch.arange(6))
        def loaders():
            return (DataLoader(dataset,batch_size=2,shuffle=True,generator=torch.Generator().manual_seed(5)),
                    DataLoader(dataset,batch_size=2,generator=torch.Generator().manual_seed(6)))
        def epoch(model,optimizer,loader):
            order=[]
            model.train()
            for images,labels,ids in loader:
                order.extend(ids.tolist())
                optimizer.zero_grad()
                torch.nn.functional.binary_cross_entropy_with_logits(model(images),labels).backward()
                optimizer.step()
            return order
        model=model_factory()
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        loader,validation_loader=loaders()
        config={'size':224,'batch_size':2,'seed':5,'learning_rate':.001,'weight_decay':.01,
                'preprocess':'RGB letterbox /255','aggregation':'mean image probabilities per Study_id',
                'threshold':.5,'device':'cpu','epochs':1}
        epoch(model,optimizer,loader)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            root=Path(temporary)
            checkpoint=root/'model.pth'
            torch.save({'state_dict':model.state_dict(),'metadata':config},checkpoint)
            save_training_state(root/'training_state.pth',checkpoint,optimizer,loader,validation_loader,1,config,'train','valid')
            expected_random=(random.random(),np.random.random(),torch.rand(1))
            expected_order=epoch(model,optimizer,loader)
            expected=copy.deepcopy(model.state_dict())
            state,path=load_training_state(root/'training_state.pth')
            resumed=model_factory()
            resumed.load_state_dict(torch.load(path,weights_only=True)['state_dict'])
            resumed_optimizer=torch.optim.AdamW(resumed.parameters(),lr=.001)
            resumed_loader,resumed_validation=loaders()
            random.seed(99); np.random.seed(99); torch.manual_seed(99)
            restore_training_state(state,resumed_optimizer,resumed_loader,resumed_validation,config,'train','valid')
            self.assertEqual(expected_random[0],random.random())
            self.assertEqual(expected_random[1],np.random.random())
            torch.testing.assert_close(expected_random[2],torch.rand(1),rtol=0,atol=0)
            self.assertEqual(expected_order,epoch(resumed,resumed_optimizer,resumed_loader))
            for key,value in expected.items():
                torch.testing.assert_close(value,resumed.state_dict()[key],rtol=0,atol=0)
            for changes,train_hash in [({'learning_rate':.01},'train'),({},'wrong-data'),({'device':'cuda'},'train')]:
                with self.assertRaises(ValueError):
                    restore_training_state(state,resumed_optimizer,resumed_loader,resumed_validation,{**config,**changes},train_hash,'valid')
            checkpoint.write_bytes(b'wrong model')
            with self.assertRaises(ValueError):
                load_training_state(root/'training_state.pth')
            torch.save({'epoch':1},root/'legacy.pth')
            with self.assertRaises(ValueError):
                load_training_state(root/'legacy.pth')


if __name__ == '__main__':
    unittest.main()
