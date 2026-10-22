import unittest
import pandas as pd
from src.evaluate import study_metrics


class EvaluationTests(unittest.TestCase):
    def test_alignment_averaging_and_auc(self) -> None:
        frame = pd.DataFrame({'Dicom_id': ['a','b','c'], 'Subject_id': ['p','p','q'],
                              'Study_id': ['s','s','t'], 'Predict_class': ['1','1','2']})
        predictions = pd.DataFrame([{'Dicom_id': image, **{f'p{i}': value if i == 1 else 1-value for i in range(1,11)}}
                                    for image, value in [('c',0.1), ('b',0.8), ('a',0.6)]])
        result, metrics = study_metrics(frame, predictions)
        self.assertAlmostEqual(result.loc[0,'p1'], 0.7)
        self.assertEqual(metrics['class_auc']['1'], 1.0)
        self.assertEqual(metrics['class_auc']['2'], 1.0)
        self.assertIsNone(metrics['macro_auc'])
        self.assertEqual(metrics['undefined_auc_classes'], list(range(3,11)))
        with self.assertRaises(ValueError):
            study_metrics(frame, predictions.iloc[:2])


if __name__ == '__main__':
    unittest.main()
