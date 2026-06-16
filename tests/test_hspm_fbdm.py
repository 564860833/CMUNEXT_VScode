import unittest

import torch

from src.network.conv_based.CMUNeXt_HSPM_FBDM import (
    FBDM,
    FixedHaarDWT,
    cmunext_hspm_fbdm,
)
from src.utils.losses import HSPMFBDMLoss, mask_to_edge


class HSPMFBDMTests(unittest.TestCase):
    def _small_model(self, **kwargs):
        return cmunext_hspm_fbdm(
            dims=(4, 8, 16, 20, 24),
            depths=(1, 1, 1, 1, 1),
            kernels=(3, 3, 3, 3, 3),
            **kwargs,
        )

    def test_output_contract_and_default_gamma(self):
        model = self._small_model().eval()
        with torch.no_grad():
            outputs = model(torch.randn(2, 3, 32, 32))

        self.assertEqual(set(outputs), {"seg", "coarse", "uncertainty", "edge"})
        self.assertEqual(outputs["seg"].shape, (2, 1, 32, 32))
        self.assertEqual(outputs["edge"].shape, (2, 1, 32, 32))
        self.assertEqual(outputs["coarse"].shape, (2, 1, 4, 4))
        self.assertAlmostEqual(model.fbdm1.effective_gamma().item(), 0.03, places=5)

    def test_dual_path_size_aware_fusion_diagnostics_are_preserved(self):
        model = self._small_model(
            hspm_backbone_mode="dual_path",
            hspm_fusion_mode="size_aware",
        ).eval()
        with torch.no_grad():
            outputs = model(torch.randn(2, 3, 32, 32))

        self.assertEqual(outputs["seg"].shape, (2, 1, 32, 32))
        self.assertIsNotNone(model.last_fusion_diagnostics)
        self.assertIn("spatial_gate", model.last_fusion_diagnostics)

    def test_fixed_haar_dwt_pads_odd_spatial_sizes(self):
        dwt = FixedHaarDWT()
        coeffs = dwt(torch.randn(2, 3, 15, 17))

        self.assertEqual(len(coeffs), 4)
        for coeff in coeffs:
            self.assertEqual(coeff.shape, (2, 3, 8, 9))
        self.assertEqual(sum(parameter.numel() for parameter in dwt.parameters()), 0)

    def test_hspm_prior_is_detached_by_default(self):
        module = FBDM(channels=4).eval()
        x = torch.randn(2, 4, 16, 16, requires_grad=True)
        coarse_logits = torch.randn(2, 1, 4, 4, requires_grad=True)
        uncertainty = torch.rand(2, 1, 4, 4, requires_grad=True)

        out, edge_logits = module(x, coarse_logits=coarse_logits, uncertainty=uncertainty)
        (out.mean() + edge_logits.mean()).backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNone(coarse_logits.grad)
        self.assertIsNone(uncertainty.grad)

    def test_hspm_prior_detach_can_be_disabled_for_ablation(self):
        module = FBDM(channels=4, detach_hspm_prior=False).eval()
        x = torch.randn(2, 4, 16, 16, requires_grad=True)
        coarse_logits = torch.randn(2, 1, 4, 4, requires_grad=True)
        uncertainty = torch.rand(2, 1, 4, 4, requires_grad=True)

        out, edge_logits = module(x, coarse_logits=coarse_logits, uncertainty=uncertainty)
        (out.mean() + edge_logits.mean()).backward()

        self.assertIsNotNone(coarse_logits.grad)
        self.assertIsNotNone(uncertainty.grad)

    def test_semantic_prior_weights_are_normalized_and_clamped(self):
        module = FBDM(
            channels=1,
            semantic_uncertainty_weight=7.0,
            semantic_coarse_weight=3.0,
        ).eval()
        x = torch.ones(1, 1, 8, 8)
        coarse_logits = torch.zeros(1, 1, 2, 2)
        uncertainty = torch.zeros(1, 1, 2, 2)

        _, semantic_prior, boundary_gate = module.build_boundary_prior(
            x,
            coarse_logits=coarse_logits,
            uncertainty=uncertainty,
        )

        self.assertTrue(torch.allclose(semantic_prior, torch.full_like(semantic_prior, 0.15)))
        self.assertGreaterEqual(semantic_prior.min().item(), 0.0)
        self.assertLessEqual(semantic_prior.max().item(), 1.0)
        self.assertGreaterEqual(boundary_gate.min().item(), 0.0)
        self.assertLessEqual(boundary_gate.max().item(), 1.0)

        _, semantic_prior, _ = module.build_boundary_prior(
            x,
            coarse_logits=coarse_logits,
            uncertainty=torch.full_like(uncertainty, 2.0),
        )
        self.assertLessEqual(semantic_prior.max().item(), 1.0)

    def test_hspm_prior_can_be_disabled(self):
        module = FBDM(channels=1, use_hspm_prior=False).eval()
        edge_prior, semantic_prior, boundary_gate = module.build_boundary_prior(torch.randn(1, 1, 8, 8))

        self.assertTrue(torch.equal(semantic_prior, torch.ones_like(semantic_prior)))
        self.assertTrue(torch.equal(boundary_gate, edge_prior))

    def test_hspm_fbdm_loss_and_mask_to_edge(self):
        target = torch.zeros(2, 1, 16, 16)
        target[:, :, 4:12, 5:11] = 1.0
        edge_target = mask_to_edge(target, kernel_size=3)

        self.assertEqual(edge_target.shape, target.shape)
        self.assertGreater(edge_target.sum().item(), 0.0)
        with self.assertRaises(ValueError):
            mask_to_edge(target, kernel_size=2)

        outputs = {
            "seg": torch.randn(2, 1, 16, 16, requires_grad=True),
            "coarse": torch.randn(2, 1, 4, 4, requires_grad=True),
            "edge": torch.randn(2, 1, 16, 16, requires_grad=True),
        }
        loss = HSPMFBDMLoss(coarse_weight=0.3, edge_weight=0.05)(outputs, target)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(outputs["edge"].grad)


if __name__ == "__main__":
    unittest.main()
