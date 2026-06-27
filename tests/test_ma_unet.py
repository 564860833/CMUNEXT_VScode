import unittest

import torch

from src.network.conv_based.MA_UNet import MA_UNet, ma_unet


class MAUNetTests(unittest.TestCase):
    def test_forward_returns_single_logits_tensor(self):
        model = MA_UNet(input_channel=3, num_classes=1).eval()

        with torch.no_grad():
            outputs = model(torch.randn(1, 3, 64, 64))

        self.assertIsInstance(outputs, torch.Tensor)
        self.assertEqual(outputs.shape, (1, 1, 64, 64))
        self.assertTrue(torch.isfinite(outputs).all())

    def test_factory_instantiates_model(self):
        model = ma_unet(num_classes=1)

        self.assertIsInstance(model, MA_UNet)


if __name__ == "__main__":
    unittest.main()
