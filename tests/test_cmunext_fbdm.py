import unittest

import torch

from src.network.conv_based.CMUNeXt import cmunext
from src.network.conv_based.CMUNeXt_FBDM import (
    cmunext_fbdm,
    cmunext_fbdm_l,
    cmunext_fbdm_s,
)
from src.utils.losses import FBDMLoss


class CMUNeXtFBDMTests(unittest.TestCase):
    def _small_model(self, **kwargs):
        return cmunext_fbdm(
            dims=(4, 8, 16, 20, 24),
            depths=(1, 1, 1, 1, 1),
            kernels=(3, 3, 3, 3, 3),
            **kwargs,
        )

    def test_output_contract_is_fbdm_only(self):
        model = self._small_model().eval()
        with torch.no_grad():
            outputs = model(torch.randn(2, 3, 32, 32))

        self.assertEqual(set(outputs), {"seg", "edge"})
        self.assertNotIn("coarse", outputs)
        self.assertNotIn("uncertainty", outputs)
        self.assertFalse(hasattr(model, "prototype_mixer"))
        self.assertEqual(outputs["seg"].shape, (2, 1, 32, 32))
        self.assertEqual(outputs["edge"].shape, (2, 1, 32, 32))
        self.assertFalse(model.fbdm1.use_hspm_prior)
        self.assertFalse(model.fbdm1.edge_aux_only)
        self.assertAlmostEqual(model.fbdm1.effective_gamma().item(), 0.03, places=5)

    def test_edge_aux_only_output_contract_is_fbdm_only(self):
        model = self._small_model(fbdm_edge_aux_only=True).eval()
        with torch.no_grad():
            outputs = model(torch.randn(2, 3, 32, 32))

        self.assertTrue(model.fbdm1.edge_aux_only)
        self.assertEqual(set(outputs), {"seg", "edge"})
        self.assertEqual(outputs["seg"].shape, (2, 1, 32, 32))
        self.assertEqual(outputs["edge"].shape, (2, 1, 32, 32))

    def test_factories_smoke_test(self):
        for factory in (cmunext_fbdm_s, cmunext_fbdm_l):
            model = factory().eval()
            with torch.no_grad():
                outputs = model(torch.randn(1, 3, 32, 32))
            self.assertEqual(outputs["seg"].shape, (1, 1, 32, 32))
            self.assertEqual(outputs["edge"].shape, (1, 1, 32, 32))

    def test_fbdm_loss_backward(self):
        target = torch.zeros(2, 1, 16, 16)
        target[:, :, 4:12, 5:11] = 1.0
        outputs = {
            "seg": torch.randn(2, 1, 16, 16, requires_grad=True),
            "edge": torch.randn(2, 1, 16, 16, requires_grad=True),
        }

        loss = FBDMLoss(edge_weight=0.05)(outputs, target)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

        self.assertIsNotNone(outputs["seg"].grad)
        self.assertIsNotNone(outputs["edge"].grad)

    def test_baseline_cmunext_output_contract_is_unchanged(self):
        model = cmunext(
            dims=(4, 8, 16, 20, 24),
            depths=(1, 1, 1, 1, 1),
            kernels=(3, 3, 3, 3, 3),
        ).eval()
        with torch.no_grad():
            outputs = model(torch.randn(2, 3, 32, 32))

        self.assertIsInstance(outputs, torch.Tensor)
        self.assertEqual(outputs.shape, (2, 1, 32, 32))


if __name__ == "__main__":
    unittest.main()
