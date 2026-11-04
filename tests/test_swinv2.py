import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from torch import nn
from src.swinv2_experiment import *
class SwinTests(unittest.TestCase):
 def setUp(self):torch.set_num_threads(2);torch.manual_seed(42);self.c=load_config('configs/experiments/X_SWINV2_01.json')
 def batch(self,n,ids):return {'image':torch.randn(n,3),'label':torch.randint(0,2,(n,10)).float(),'study_index':torch.tensor(ids),'Study_id':[str(x) for x in ids]}
 def test_accumulation_gradient_matches_joint(self):
  a=nn.Linear(3,10);b=copy.deepcopy(a);batch1=self.batch(1,[0]);batch2=self.batch(3,[0,0,1]);opt=torch.optim.SGD(a.parameters(),lr=.01);ema=EMA(a)
  accumulated_update(a,opt,ema,[batch1,batch2],'cpu',torch.amp.GradScaler('cuda',enabled=False),False,1000)
  op=torch.optim.SGD(b.parameters(),lr=.01);op.zero_grad();x=torch.cat([batch1['image'],batch2['image']]);y=torch.cat([batch1['label'],batch2['label']]);loss=study_balanced_bce(b(x),y,torch.tensor([0,1,1,2]));loss.backward();op.step()
  for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p,q)
  self.assertEqual(ema.updates,1)
 def test_tail_groups(self):
  batches=[{'image':torch.zeros(n,1)} for n in (8,8,8,8,5)]
  groups=list(accumulation_groups(batches));self.assertEqual([sum(len(b['image']) for b in g) for g in groups],[32,5])
 def test_overshoot(self):
  g=list(accumulation_groups([{'image':torch.zeros(7,1)} for _ in range(6)]));self.assertEqual([sum(len(b['image']) for b in x) for x in g],[35,7])
 def test_ema_math(self):
  m=nn.Linear(1,1);e=EMA(m,.999);old=e.model.weight.clone()
  with torch.no_grad():m.weight.add_(2)
  e.update(m);torch.testing.assert_close(e.model.weight,old+.002);self.assertFalse(e.model.training)
 def test_skipped_amp_does_not_update_ema(self):
  class Fake:
   scale_value=8
   def scale(self,x):return x
   def unscale_(self,o):pass
   def get_scale(self):return self.scale_value
   def step(self,o):pass
   def update(self):self.scale_value=4
  m=nn.Linear(3,10);e=EMA(m);opt=torch.optim.SGD(m.parameters(),lr=.1)
  r=accumulated_update(m,opt,e,[self.batch(1,[0])],'cpu',Fake(),False)
  self.assertFalse(r['stepped']);self.assertEqual(e.updates,0)
 def test_real_cpu_scaler(self):
  m=nn.Linear(3,10);e=EMA(m);opt=torch.optim.SGD(m.parameters(),lr=.01)
  scaler=torch.amp.GradScaler('cpu',init_scale=128)
  r=accumulated_update(m,opt,e,[self.batch(1,[0])],'cpu',scaler,False)
  self.assertTrue(r['stepped']);self.assertEqual(e.updates,1)
 def test_duplicate_config_augmentation_rejected(self):
  with tempfile.TemporaryDirectory(dir=Path('outputs').resolve()) as d:
   c=copy.deepcopy(self.c);c['augmentation']['degrees']=9;p=Path(d)/'config.json';p.write_text(json.dumps(c),encoding='utf-8')
   with self.assertRaises(ValueError):load_config(p)
 def test_actual_overflow_json_safe(self):
  m=nn.Linear(3,10);e=EMA(m);opt=torch.optim.SGD(m.parameters(),lr=.01);old=m.weight.detach().clone()
  handle=m.weight.register_hook(lambda grad:torch.full_like(grad,float('inf')))
  r=accumulated_update(m,opt,e,[self.batch(1,[0])],'cpu',torch.amp.GradScaler('cpu',init_scale=1024),False)
  handle.remove();self.assertFalse(r['stepped']);self.assertIsNone(r['gradient_norm']);self.assertEqual(r['scale_after'],512);self.assertEqual(e.updates,0);torch.testing.assert_close(old,m.weight)
  json.dumps(r,allow_nan=False)
 def test_smoke_retry_then_success(self):
  class E:updates=0
  e=E();attempts=[]
  def update(*args):
   attempts.append(1);ok=len(attempts)==3
   if ok:e.updates+=1
   return {'stepped':ok,'gradient_norm':1. if ok else None}
  r=smoke_attempts(None,None,e,[],None,None,8,update)
  self.assertTrue(r['passed']);self.assertEqual(len(r['attempts']),3);self.assertEqual(e.updates,1)
 def test_smoke_continuous_overflow_fails(self):
  class E:updates=0
  r=smoke_attempts(None,None,E(),[],None,None,8,lambda *args:{'stepped':False,'gradient_norm':None})
  self.assertFalse(r['passed']);self.assertEqual(len(r['attempts']),8);self.assertEqual(r['successful_updates'],0)
 def test_invalid_json_no_partial_file(self):
  with tempfile.TemporaryDirectory(dir=Path('outputs').resolve()) as d:
   p=Path(d)/'bad.json'
   with self.assertRaises(ValueError):dump(p,{'value':float('inf')})
   self.assertFalse(p.exists())
 def test_bf16_update_without_scaler(self):
  m=nn.Linear(3,10);e=EMA(m);opt=torch.optim.SGD(m.parameters(),lr=.01);dtype,scaler=amp_components(self.c,'cpu')
  self.assertEqual(dtype,torch.bfloat16);self.assertFalse(scaler.is_enabled())
  r=accumulated_update(m,opt,e,[self.batch(2,[0,1])],'cpu',scaler,True,amp_dtype=dtype)
  self.assertTrue(r['stepped']);self.assertTrue(r['gradients_finite']);self.assertEqual(e.updates,1)
 def test_no_bf16_gpu_no_silent_fallback(self):
  from unittest.mock import patch
  with patch('torch.cuda.is_available',return_value=False):
   with self.assertRaises(RuntimeError):amp_components(self.c,'cuda')
 def test_unscaled_nonfinite_gradient_is_diagnostic(self):
  m=nn.Linear(3,10);e=EMA(m);opt=torch.optim.SGD(m.parameters(),lr=.01);handle=m.weight.register_hook(lambda g:torch.full_like(g,float('inf')))
  with self.assertRaisesRegex(FloatingPointError,'weight'):accumulated_update(m,opt,e,[self.batch(1,[0])],'cpu',torch.amp.GradScaler('cpu',enabled=False),False)
  handle.remove();self.assertEqual(e.updates,0)
 def test_lr(self):
  self.assertAlmostEqual(rates(self.c,1)['backbone'],2e-6);self.assertAlmostEqual(rates(self.c,2)['head'],2e-4)
  for v in rates(self.c,12).values():self.assertEqual(v,1e-6)
 def test_optimizer_groups(self):
  m=build(self.c,False);o=optimizer_for(m,self.c);ids=[id(p) for g in o.param_groups for p in g['params']]
  self.assertEqual(len(ids),len(set(ids)));self.assertEqual(set(ids),{id(p) for p in m.parameters()});self.assertEqual(o.param_groups[1]['lr'],2e-4)
 def test_shape_backward_and_checkpoint(self):
  m=build(self.c,False);x=torch.randn(1,3,384,384);z=m(x);self.assertEqual(tuple(z.shape),(1,10));z.sum().backward();self.assertTrue(torch.isfinite(m.head[1].weight.grad).all())
  with tempfile.TemporaryDirectory(dir=Path("outputs").resolve()) as d:
   path=Path(d)/'m.pth';save_checkpoint(path,m,self.c,{},5,'ema',{});loaded,payload=load_checkpoint(path,'cpu')
   m.eval()
   torch.testing.assert_close(predict_batch(m,x),predict_batch(loaded,x),rtol=0,atol=0);self.assertEqual(payload['variant'],'ema')
 def test_permission_gate(self):
  with tempfile.TemporaryDirectory(dir=Path('outputs').resolve()) as d:
   review=json.loads(Path(self.c['review_file']).read_text(encoding='utf-8'));review['status']='pending';p=Path(d)/'pending.json';p.write_text(json.dumps(review),encoding='utf-8');c=copy.deepcopy(self.c);c['review_file']=str(p)
   with self.assertRaises(PermissionError):license_check(c)
 def test_approved_gate(self):
  self.assertEqual(license_check(self.c)['status'],'approved')
 def test_evidence_tamper_gate(self):
  with tempfile.TemporaryDirectory(dir=Path('outputs').resolve()) as d:
   review=json.loads(Path(self.c['review_file']).read_text(encoding='utf-8'));review['evidence'][0]['sha256']='0'*64;p=Path(d)/'tampered.json';p.write_text(json.dumps(review),encoding='utf-8');c=copy.deepcopy(self.c);c['review_file']=str(p)
   with self.assertRaises(ValueError):license_check(c)
 def test_augmentation_metadata(self):
  from PIL import Image
  import pandas as pd
  with tempfile.TemporaryDirectory(dir=Path("outputs").resolve()) as d:
   d=Path(d);Image.new('L',(100,200),120).save(d/'pic.jpg');pd.DataFrame([{'Dicom_id':'pic','Subject_id':'sub','Study_id':'st','Predict_class':'1;4'}]).to_csv(d/'t.csv',index=False)
   ds=SwinDataset(d/'t.csv',d,size=384,normalization='imagenet',augment=True);torch.manual_seed(42);a=ds[0];torch.manual_seed(42);b=ds[0];torch.testing.assert_close(a['image'],b['image']);self.assertEqual(a['Study_id'],'st');self.assertEqual(tuple(a['image'].shape),(3,384,384))
if __name__=='__main__':unittest.main()
