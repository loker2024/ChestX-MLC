import tempfile
import unittest
from pathlib import Path

import torch

from src.model import build_model, load_model
from src.train import save_model, seed_everything, train_step


class ModelTests(unittest.TestCase):
    def test_update_and_reload(self) -> None:
        torch.set_num_threads(2)
        seed_everything(42)
        model = build_model()
        images = torch.rand(2, 3, 64, 64)
        labels = torch.zeros(2, 10)
        labels[:, 6] = 1
        before = model.fc.weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss, shape = train_step(model, images, labels, optimizer)
        self.assertEqual(shape, [2, 10])
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertFalse(torch.equal(before, model.fc.weight))
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            path = Path(folder) / 'model.pth'
            save_model(model, path, {})
            with self.assertRaises(FileExistsError):
                save_model(model, path, {})
            restored = load_model(str(path))
            model.eval()
            with torch.no_grad():
                torch.testing.assert_close(model(images), restored(images), rtol=0, atol=0)
        with self.assertRaises(ValueError):
            train_step(model, images, labels[:, :9], optimizer)


if __name__ == '__main__':
    unittest.main()
