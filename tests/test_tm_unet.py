import unittest

import torch

from src.network.conv_based.TM_UNet import TM_UNet, tm_unet


class TMUNetTests(unittest.TestCase):
    def test_forward_returns_single_logits_tensor(self):
        model = TM_UNet(input_channel=3, num_classes=1, img_size=64, model_size="t").eval()

        with torch.no_grad():
            outputs = model(torch.randn(1, 3, 64, 64))

        self.assertIsInstance(outputs, torch.Tensor)
        self.assertEqual(outputs.shape, (1, 1, 64, 64))
        self.assertTrue(torch.isfinite(outputs).all())

    def test_factory_instantiates_all_model_sizes(self):
        expected_dims = {
            "t": (8, 16, 32, 64, 128),
            "s": (16, 32, 64, 128, 256),
            "b": (32, 64, 128, 256, 512),
        }

        for model_size, embed_dims in expected_dims.items():
            with self.subTest(model_size=model_size):
                model = tm_unet(model_size=model_size, img_size=64)

                self.assertIsInstance(model, TM_UNet)
                self.assertEqual(model.model_size, model_size)
                self.assertEqual(model.embed_dims, embed_dims)

    def test_default_model_size_is_base(self):
        model = TM_UNet(img_size=64)

        self.assertEqual(model.model_size, "b")
        self.assertEqual(model.embed_dims, (32, 64, 128, 256, 512))

    def test_invalid_model_size_raises(self):
        with self.assertRaises(ValueError):
            TM_UNet(model_size="x")


if __name__ == "__main__":
    unittest.main()
