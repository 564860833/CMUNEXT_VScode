import unittest
import sys
from argparse import Namespace
from unittest import mock

import torch

with mock.patch.object(sys, "argv", ["main.py"]):
    import main as training_main

from src.network.conv_based.CMUNeXt_HSPM_FBDM import (
    FBDM,
    FBDMLogitCorrection,
    FixedHaarDWT,
    cmunext_hspm_fbdm,
    cmunext_hspm_fbdm_v2,
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

    def _small_v2_model(self, **kwargs):
        return cmunext_hspm_fbdm_v2(
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
        self.assertFalse(model.fbdm1.edge_aux_only)
        self.assertAlmostEqual(model.fbdm1.effective_gamma().item(), 0.03, places=5)

    def test_fbdm_residual_scale_controls_effective_gamma(self):
        module = FBDM(channels=4)

        self.assertAlmostEqual(module.effective_gamma().item(), 0.03, places=5)
        module.set_residual_scale(0.5)
        self.assertAlmostEqual(module.effective_gamma().item(), 0.015, places=5)
        module.set_residual_scale(-1.0)
        self.assertAlmostEqual(module.effective_gamma().item(), 0.0, places=5)
        module.set_residual_scale(2.0)
        self.assertAlmostEqual(module.effective_gamma().item(), 0.03, places=5)

    def test_fbdm_edge_aux_only_returns_original_feature_but_trains_edge_head(self):
        module = FBDM(channels=4, use_hspm_prior=False, edge_aux_only=True).eval()
        x = torch.randn(2, 4, 16, 16, requires_grad=True)

        out, edge_logits = module(x)

        self.assertTrue(torch.equal(out, x))
        self.assertEqual(edge_logits.shape, (2, 1, 16, 16))
        edge_logits.mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(module.edge_head.weight.grad)

    def test_hspm_fbdm_edge_aux_only_output_contract_is_preserved(self):
        model = self._small_model(fbdm_edge_aux_only=True).eval()
        with torch.no_grad():
            outputs = model(torch.randn(2, 3, 32, 32))

        self.assertTrue(model.fbdm1.edge_aux_only)
        self.assertEqual(set(outputs), {"seg", "coarse", "uncertainty", "edge"})
        self.assertEqual(outputs["seg"].shape, (2, 1, 32, 32))
        self.assertEqual(outputs["edge"].shape, (2, 1, 32, 32))

    def test_fbdm_logit_correction_is_bounded(self):
        correction = FBDMLogitCorrection(
            channels=4,
            correction_scale_init=0.05,
            correction_scale_max=0.3,
        ).eval()
        correction.set_schedule_scale(0.5)
        with torch.no_grad():
            correction.correction_head.bias.fill_(10.0)

        feature = torch.randn(2, 4, 16, 16)
        base_logits = torch.randn(2, 1, 16, 16, requires_grad=True)
        edge_logits = torch.randn(2, 1, 16, 16, requires_grad=True)
        boundary_gate = torch.rand(2, 1, 16, 16)

        final_logits, logit_correction = correction(
            feature,
            base_logits,
            edge_logits,
            boundary_gate,
        )

        bound = correction.schedule_scale * correction.effective_correction_scale()
        self.assertLessEqual(logit_correction.abs().max().item(), bound.item() + 1e-6)
        self.assertEqual(final_logits.shape, base_logits.shape)

    def test_hspm_fbdm_v2_output_contract_and_zero_correction(self):
        model = self._small_v2_model().eval()
        with torch.no_grad():
            outputs = model(torch.randn(2, 3, 32, 32))

        self.assertEqual(
            set(outputs),
            {"seg", "base_seg", "coarse", "uncertainty", "edge", "logit_correction"},
        )
        self.assertTrue(model.fbdm1.edge_aux_only)
        self.assertEqual(outputs["seg"].shape, (2, 1, 32, 32))
        self.assertEqual(outputs["base_seg"].shape, (2, 1, 32, 32))
        self.assertEqual(outputs["edge"].shape, (2, 1, 32, 32))
        self.assertTrue(torch.allclose(outputs["seg"], outputs["base_seg"]))
        self.assertTrue(torch.allclose(outputs["logit_correction"], torch.zeros_like(outputs["seg"])))

    def test_hspm_fbdm_v2_can_disable_hspm_prior(self):
        model = self._small_v2_model(fbdm_use_hspm_prior=False).eval()
        with torch.no_grad():
            outputs = model(torch.randn(2, 3, 32, 32))

        self.assertFalse(model.fbdm1.use_hspm_prior)
        self.assertEqual(outputs["seg"].shape, (2, 1, 32, 32))

    def test_hspm_fbdm_v2_loss_backward(self):
        model = self._small_v2_model().eval()
        target = torch.zeros(2, 1, 32, 32)
        target[:, :, 8:24, 10:22] = 1.0

        outputs = model(torch.randn(2, 3, 32, 32))
        total, components = HSPMFBDMLoss(
            coarse_weight=0.1,
            edge_weight=0.03,
        )(outputs, target, return_components=True)

        self.assertTrue(torch.isfinite(total))
        self.assertIn("edge_weighted", components)
        total.backward()
        self.assertIsNotNone(model.Conv_1x1.weight.grad)
        self.assertIsNotNone(model.fbdm1.edge_head.weight.grad)
        self.assertIsNotNone(model.fbdm_correction.correction_head.weight.grad)

    def test_hspm_fbdm_v2_boundary_band_loss_components_sum(self):
        model = self._small_v2_model().eval()
        target = torch.zeros(2, 1, 32, 32)
        target[:, :, 8:24, 10:22] = 1.0
        outputs = model(torch.randn(2, 3, 32, 32))

        total, components = HSPMFBDMLoss(
            coarse_weight=0.1,
            edge_weight=0.03,
            boundary_band_weight=1.0,
            boundary_band_kernel_size=7,
        )(outputs, target, return_components=True)

        expected = (
            components["seg"]
            + components["coarse_weighted"]
            + components["edge_weighted"]
            + components["boundary_band_weighted"]
        )
        self.assertTrue(torch.allclose(total, expected))
        self.assertGreater(components["boundary_band_weighted"].item(), 0.0)
        self.assertTrue(torch.isfinite(total))

    def test_hspm_fbdm_v2_empty_boundary_band_is_safe(self):
        model = self._small_v2_model().eval()
        target = torch.zeros(2, 1, 32, 32)
        outputs = model(torch.randn(2, 3, 32, 32))

        total, components = HSPMFBDMLoss(
            coarse_weight=0.0,
            edge_weight=0.0,
            boundary_band_weight=1.0,
            boundary_band_kernel_size=7,
        )(outputs, target, return_components=True)

        self.assertTrue(torch.isfinite(total))
        self.assertEqual(components["boundary_band_weighted"].item(), 0.0)

    def test_hspm_fbdm_v2_boundary_band_loss_only_updates_correction(self):
        model = self._small_v2_model().train()
        target = torch.zeros(2, 1, 32, 32)
        target[:, :, 8:24, 10:22] = 1.0
        outputs = model(torch.randn(2, 3, 32, 32))

        _, components = HSPMFBDMLoss(
            coarse_weight=0.0,
            edge_weight=0.0,
            boundary_band_weight=1.0,
            boundary_band_kernel_size=7,
        )(outputs, target, return_components=True)
        components["boundary_band_weighted"].backward()

        self.assertGreater(model.fbdm_correction.correction_head.weight.grad.abs().sum().item(), 0.0)
        self.assertIsNone(model.Conv_1x1.weight.grad)
        self.assertIsNone(model.Up_conv2.conv[0].weight.grad)

    def test_hspm_fbdm_boundary_band_loss_requires_v2_outputs(self):
        model = self._small_model().eval()
        target = torch.zeros(2, 1, 32, 32)
        target[:, :, 8:24, 10:22] = 1.0
        outputs = model(torch.randn(2, 3, 32, 32))

        with self.assertRaises(KeyError):
            HSPMFBDMLoss(
                coarse_weight=0.0,
                edge_weight=0.0,
                boundary_band_weight=1.0,
            )(outputs, target)

    def test_fbdm_boundary_band_weight_schedule(self):
        args = Namespace(
            fbdm_boundary_band_loss_weight=0.03,
            fbdm_boundary_band_loss_final_weight=0.005,
            fbdm_boundary_band_loss_decay_epochs=150,
        )
        expected = {
            0: 0.03,
            75: 0.0175,
            150: 0.005,
            300: 0.005,
        }
        for epoch, band_weight in expected.items():
            self.assertAlmostEqual(
                training_main.get_fbdm_boundary_band_weight(args, epoch),
                band_weight,
            )

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
