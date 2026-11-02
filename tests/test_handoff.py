import unittest
import pandas as pd
import numpy as np
from PIL import Image
from src.ens02_infer import average,PROBS
from src.data import preprocess
class HandoffTests(unittest.TestCase):
    def test_average_and_order(self):
        a=pd.DataFrame([{'Subject_id':'s01','Study_id':'v01',**{p:.2 for p in PROBS}},{'Subject_id':'s02','Study_id':'v02',**{p:.6 for p in PROBS}}])
        b=a.copy();b[PROBS]=.8
        result=average(a,b.iloc[::-1][['Study_id','Subject_id',*reversed(PROBS)]])
        np.testing.assert_allclose(result[PROBS].to_numpy(),np.array([[.5]*10,[.7]*10]))
    def test_reject_wrong_patient(self):
        a=pd.DataFrame([{'Subject_id':'s01','Study_id':'v01',**{p:.2 for p in PROBS}}]);b=a.copy();b.Subject_id='wrong'
        with self.assertRaises(ValueError):average(a,b)
    def test_preprocess(self):
        t=preprocess(Image.new('L',(200,400),128),384,'imagenet')
        self.assertEqual(tuple(t.shape),(3,384,384))
if __name__=='__main__':unittest.main()
