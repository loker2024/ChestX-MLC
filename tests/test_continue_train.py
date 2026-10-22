import unittest
from src.continue_train import better_auc


class SelectionTests(unittest.TestCase):
    def test_select_by_auc_only(self) -> None:
        self.assertTrue(better_auc(0.75, 0.70))
        self.assertFalse(better_auc(0.69, 0.70))
        self.assertFalse(better_auc(0.70, 0.70))
        self.assertFalse(better_auc(None, 0.70))
