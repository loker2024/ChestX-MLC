"""ASL接入Swin的CPU对照与门禁测试。"""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
from src.asl import AsymmetricLabelLoss
from src.sw02 import study_asl, check_reference
from src.swinv2_experiment import load_config, accumulated_update, EMA


class SW02Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        self.c=load_config('configs/experiments/SW02_SwinV2_ASL_EMA.json')

    def test_only_loss_differs(self):
        a=load_config('configs/experiments/X_SWINV2_01_official20.json')
        changed={k for k in set(a)|set(self.c) if a.get(k)!=self.c.get(k)}
        self.assertEqual(changed,{'experiment_id','run_label','loss','asl'})

    def test_shapes_labels_and_extremes(self):
        for y in (torch.zeros(4,10),torch.ones(4,10),torch.randint(2,(4,10)).float()):
            x=torch.linspace(-1000,1000,40).reshape(4,10).requires_grad_()
            loss=study_asl(x,y,torch.tensor([0,1,1,1]))
            loss.backward()
            self.assertTrue(torch.isfinite(loss) and torch.isfinite(x.grad).all())
        with self.assertRaises(ValueError):study_asl(torch.zeros(2,9),torch.zeros(2,9),torch.tensor([0,1]))
        with self.assertRaises(ValueError):study_asl(torch.zeros(2,10),torch.full((2,10),-1.),torch.tensor([0,1]))

    def test_study_equal_weight(self):
        x=torch.randn(4,10);y=torch.randint(2,(4,10)).float()
        label=AsymmetricLabelLoss()(x,y)
        torch.testing.assert_close(study_asl(x,y,torch.tensor([0,1,1,1])),(label[0].mean()+label[1:].mean())/2)

    def test_bce_limit_and_easy_negatives(self):
        x=torch.randn(4,10);y=torch.randint(2,(4,10)).float()
        torch.testing.assert_close(AsymmetricLabelLoss(0,0,0)(x,y),torch.nn.functional.binary_cross_entropy_with_logits(x,y,reduction='none'))
        x=torch.full((1,10),-5.);y=torch.zeros_like(x)
        self.assertLess(AsymmetricLabelLoss()(x,y).mean(),torch.nn.functional.binary_cross_entropy_with_logits(x,y))

    def test_accumulation_matches_full_study_loss(self):
        a=nn.Linear(3,10);b=copy.deepcopy(a);x=torch.randn(4,3);y=torch.randint(2,(4,10)).float()
        batches=[{'image':x[:1],'label':y[:1],'Study_id':['a'],'study_index':torch.tensor([0])},
                 {'image':x[1:],'label':y[1:],'Study_id':['b']*3,'study_index':torch.tensor([0,0,0])}]
        opt=torch.optim.SGD(a.parameters(),lr=.01);e=EMA(a)
        r=accumulated_update(a,opt,e,batches,'cpu',torch.amp.GradScaler('cpu',enabled=False),False,1000,loss_function=study_asl)
        ob=torch.optim.SGD(b.parameters(),lr=.01);ob.zero_grad();study_asl(b(x),y,torch.tensor([0,1,1,1])).backward();ob.step()
        for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p,q)
        self.assertTrue(r['stepped']);self.assertEqual(e.updates,1)

    def test_mixed_precision_toy_update(self):
        for dtype in (torch.bfloat16,torch.float16):
            m=nn.Linear(3,10);e=EMA(m);opt=torch.optim.AdamW(m.parameters(),lr=1e-3)
            batch={'image':torch.randn(4,3),'label':torch.randint(2,(4,10)).float(),'Study_id':['a','b','b','b'],'study_index':torch.tensor([0,1,1,1])}
            r=accumulated_update(m,opt,e,[batch],'cpu',torch.amp.GradScaler('cpu',enabled=dtype==torch.float16,init_scale=128),True,1.,amp_dtype=dtype,loss_function=study_asl)
            self.assertTrue(r['stepped']);self.assertTrue(r['gradients_finite'])

    def test_missing_or_old_reference(self):
        self.assertFalse(check_reference(self.c,Path('not_read.json'))['verified'])
        with tempfile.TemporaryDirectory(dir=Path('outputs').resolve()) as folder:
            p=Path(folder)
            (p/'train_config.json').write_text(json.dumps({'loss':'BCEWithLogitsLoss','validation_mode':'fixed_v2_internal','epochs':12}))
            with self.assertRaisesRegex(ValueError,'不是官方'):check_reference(self.c,Path('not_read.json'),p)

    def test_reference_accepts_and_rejects_config_change(self):
        from src.analysis_common import sha256
        import pandas as pd
        with tempfile.TemporaryDirectory(dir=Path('outputs').resolve()) as folder:
            p=Path(folder);c=load_config('configs/experiments/X_SWINV2_01_official20.json')
            (p/'train_config.json').write_text(json.dumps(c))
            audit={'train_fingerprint':'train','validation_fingerprint':'valid','statistics':{},'official_valid_in_training':False}
            (p/'split_audit.json').write_text(json.dumps(audit))
            best={}
            for v in ('normal','ema'):
                w=p/('best_'+v+'.pth');w.write_bytes(b'fixture-not-real-model')
                best[v]={'epoch':20,'sha256':sha256(w),'metrics':{'macro_auc':.8}}
            summary={'completed_epochs':20,'best':best,'early_stopping':{v:{'should_stop':False} for v in best}}
            (p/'run_summary.json').write_text(json.dumps(summary))
            pd.DataFrame([{'epoch':20,'variant':v,'macro_auc':.8} for v in best]).to_csv(p/'history.csv',index=False)
            with patch('src.sw02.audit_official',return_value=({},audit)),patch('src.sw02.read_official_paths',return_value={}):
                self.assertTrue(check_reference(self.c,Path('fixture'),p)['verified'])
                changed=copy.deepcopy(self.c);changed['amp_dtype']='float16'
                with self.assertRaisesRegex(ValueError,'配置不同'):check_reference(changed,Path('fixture'),p)


if __name__=='__main__':unittest.main()
