import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from src.submission import aggregate_studies, make_submission, check_submission, PROB_COLUMNS


class SubmissionTests(unittest.TestCase):
    def test_aggregation_and_roundtrip(self) -> None:
        frame = pd.DataFrame({'Dicom_id':['01','02','03'], 'Subject_id':['001','001','002'], 'Study_id':['010','010','011']})
        predictions = pd.DataFrame([{'Dicom_id':k, **dict.fromkeys(PROB_COLUMNS,p)} for k,p in [('03',0.9),('01',0.2),('02',0.6)]])
        studies = aggregate_studies(frame,predictions)
        self.assertAlmostEqual(studies.loc[0,'p1'],0.4)
        self.assertEqual(studies.loc[0,'Study_id'],'010')
        result = make_submission(studies)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            path = Path(folder)/'submission.csv'
            result.to_csv(path,index=False,encoding='utf-8')
            self.assertTrue(check_submission(path,frame)['passed'])
            for damaged in [result.iloc[:-1],pd.concat([result,result.iloc[:1]]),
                            result.assign(Subject_id='wrong'),result.assign(Probability=np.nan),
                            result.assign(Probability=1.1),result.assign(Predict_class=0)]:
                damaged.to_csv(path,index=False,encoding='utf-8')
                with self.assertRaises(ValueError):
                    check_submission(path,frame)
        with self.assertRaises(ValueError):
            aggregate_studies(frame,predictions.iloc[:2])

    def test_threshold_selection_and_empty_studies(self) -> None:
        official = pd.DataFrame({'Subject_id':['001','002'],'Study_id':['010','011']})
        studies = official.assign(**dict.fromkeys(PROB_COLUMNS,0.2))
        studies.loc[0,'p1'] = 0.5
        selected = make_submission(studies,threshold=0.5)
        self.assertEqual(len(selected),1)
        self.assertEqual(selected.iloc[0].Predict_class,1)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            path = Path(folder)/'submission.csv'
            selected.to_csv(path,index=False,encoding='utf-8')
            self.assertTrue(check_submission(path,official,0.5,studies)['passed'])
            selected.assign(Probability=0.8).to_csv(path,index=False,encoding='utf-8')
            with self.assertRaises(ValueError):
                check_submission(path,official,0.5,studies)
            # 省略一个应为阳性的类别也必须被检测；无阳性检查允许不出现。
            selected.iloc[:0].to_csv(path,index=False,encoding='utf-8')
            with self.assertRaises(ValueError):
                check_submission(path,official,0.5,studies)
            empty = make_submission(studies,1.0)
            empty.to_csv(path,index=False,encoding='utf-8')
            self.assertTrue(check_submission(path,official,1.0,studies)['passed'])


if __name__ == '__main__':
    unittest.main()
