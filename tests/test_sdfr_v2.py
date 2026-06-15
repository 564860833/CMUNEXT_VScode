import sys
import unittest
from argparse import Namespace
from unittest import mock

import torch

with mock.patch.object(sys, "argv", ["main.py"]):
    import main as training_main

import infer as inference_main
from src.network.conv_based.CMUNeXt_HSPM_SDFR import CMUNeXt_HSPM_SDFR
from src.network.conv_based.CMUNeXt_HSPM_SDFR_V2 import (
    CMUNeXt_HSPM_SDFR_V2,
    SignedDistanceLogitCorrection,
    cmunext_hspm_sdfr_v2,
)
from src.utils.losses import SDFRV2Loss


class SDFRV2Tests(unittest.TestCase):
    def _small_model(self):
        return cmunext_hspm_sdfr_v2(
            dims=(4, 8, 16, 20, 24),
            depths=(1, 1, 1, 1, 1),
            kernels=(3, 3, 3, 3, 3),
            hspm_mixer_mode="stable",
            hspm_gamma_init=0.05,
            hspm_gamma_max=0.3,
            hspm_temperature=0.5,
        )

    def test_output_contract_and_zero_initialization(self):
        model = self._small_model().eval()
        self.assertEqual(model.sdfr.sdf_head.weight.count_nonzero().item(), 0)
        self.assertEqual(model.sdfr.correction_head.weight.count_nonzero().item(), 0)
        with torch.no_grad():
            outputs = model(torch.randn(2, 3, 32, 32))
        self.assertEqual(
            set(outputs),
            {
                "seg",
                "base_seg",
                "coarse",
                "uncertainty",
                "sdf",
                "logit_correction",
            },
        )
        self.assertTrue(torch.equal(outputs["seg"], outputs["base_seg"]))
        self.assertTrue(
            torch.equal(
                outputs["logit_correction"],
                torch.zeros_like(outputs["logit_correction"]),
            )
        )

    def test_schedule_zero_disables_nonzero_correction(self):
        model = self._small_model().eval()
        model.sdfr.correction_head.bias.data.fill_(1.0)
        model.set_sdfr_refine_schedule_scale(0.0)
        with torch.no_grad():
            outputs = model(torch.randn(2, 3, 32, 32))
        self.assertTrue(torch.equal(outputs["seg"], outputs["base_seg"]))
        self.assertTrue(
            torch.equal(
                outputs["logit_correction"],
                torch.zeros_like(outputs["logit_correction"]),
            )
        )

    def test_correction_is_bounded_and_boundary_gated(self):
        module = SignedDistanceLogitCorrection(channels=8).eval()
        module.correction_head.bias.data.fill_(100.0)
        feature = torch.randn(2, 8, 16, 16)
        base_logits = torch.randn(2, 1, 16, 16)
        with torch.no_grad():
            final_logits, _, logit_correction = module(feature, base_logits)
        expected = (
            module.schedule_scale
            * module.last_boundary_gate
            * module.last_bounded_correction
        )
        self.assertTrue(torch.allclose(logit_correction, expected))
        self.assertTrue(torch.allclose(final_logits, base_logits + logit_correction))
        self.assertLessEqual(
            logit_correction.abs().max().item(),
            module.effective_correction_scale().item() + 1e-6,
        )

    def test_correction_scale_is_bounded(self):
        module = SignedDistanceLogitCorrection(channels=8)
        self.assertAlmostEqual(module.effective_correction_scale().item(), 1.0, places=5)
        for value in (-100.0, 0.0, 100.0):
            module.correction_scale_raw.data.fill_(value)
            scale = module.effective_correction_scale().item()
            self.assertGreaterEqual(scale, 0.0)
            self.assertLessEqual(scale, 3.000001)

    def test_loss_components_sum_and_empty_band_is_safe(self):
        outputs = self._small_model().eval()(torch.randn(2, 3, 32, 32))
        target = torch.randint(0, 2, (2, 1, 32, 32)).float()
        target_sdf = torch.ones(2, 1, 32, 32)
        total, components = SDFRV2Loss()(
            outputs,
            target,
            target_sdf,
            return_components=True,
        )
        expected = (
            components["seg"]
            + components["base_weighted"]
            + components["coarse_weighted"]
            + components["sdf_weighted"]
            + components["band_weighted"]
        )
        self.assertTrue(torch.allclose(total, expected))
        self.assertEqual(components["band_weighted"].item(), 0.0)
        self.assertTrue(torch.isfinite(total))

    def test_segmentation_and_band_losses_do_not_update_sdf_head(self):
        model = self._small_model().train()
        model.sdfr.correction_head.bias.data.fill_(0.2)
        model.set_sdfr_refine_schedule_scale(1.0)
        outputs = model(torch.randn(2, 3, 32, 32))
        target = torch.randint(0, 2, (2, 1, 32, 32)).float()
        target_sdf = torch.full((2, 1, 32, 32), 0.1)
        SDFRV2Loss(
            coarse_weight=0.0,
            sdf_weight=0.0,
            base_weight=0.0,
            band_weight=1.0,
        )(outputs, target, target_sdf).backward()
        self.assertIsNone(model.sdfr.sdf_head.weight.grad)
        self.assertIsNone(model.sdfr.sdf_features[0].weight.grad)

    def test_sdf_and_band_losses_update_expected_paths(self):
        model = self._small_model().train()
        model.set_sdfr_refine_schedule_scale(1.0)
        outputs = model(torch.randn(2, 3, 32, 32))
        target = torch.randint(0, 2, (2, 1, 32, 32)).float()
        target_sdf = torch.full((2, 1, 32, 32), 0.1)
        SDFRV2Loss(
            coarse_weight=0.0,
            sdf_weight=0.2,
            base_weight=0.0,
            band_weight=1.0,
        )(outputs, target, target_sdf).backward()
        self.assertGreater(model.sdfr.sdf_head.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(
            model.sdfr.correction_head.weight.grad.abs().sum().item(),
            0.0,
        )
        self.assertIsNotNone(model.Up_conv2.conv[0].weight.grad)

    def test_training_schedule_and_dynamic_band_weight(self):
        args = Namespace(
            sdfr_refine_start_epoch=10,
            sdfr_refine_warmup_epochs=30,
            sdfr_v2_band_loss_weight=0.1,
        )
        expected = {
            0: 0.0,
            10: 0.0,
            25: 0.05,
            40: 0.1,
        }
        for epoch, band_weight in expected.items():
            self.assertAlmostEqual(
                training_main.get_sdfr_v2_band_weight(args, epoch),
                band_weight,
            )

    def test_training_compute_loss_uses_dynamic_band_weight(self):
        args = Namespace(model="CMUNeXt_HSPM_SDFR_V2")
        outputs = self._small_model().eval()(torch.randn(2, 3, 32, 32))
        target = torch.randint(0, 2, (2, 1, 32, 32)).float()
        sampled_batch = {"sdf": torch.zeros(2, 1, 32, 32)}
        _, components = training_main.compute_loss(
            args,
            SDFRV2Loss(),
            outputs,
            target,
            sampled_batch=sampled_batch,
            aux_weight=0.1,
            sdf_weight=0.0,
            band_weight=0.0,
        )
        self.assertEqual(components["sdf_weighted"].item(), 0.0)
        self.assertEqual(components["band_weighted"].item(), 0.0)

    def test_training_inference_entries_and_strict_reload(self):
        args = Namespace(**vars(training_main.args))
        args.model = "CMUNeXt_HSPM_SDFR_V2"
        args.hspm_mixer_mode = "stable"
        args.hspm_gamma_init = 0.05
        args.hspm_temperature = 0.5
        with mock.patch.object(torch.nn.Module, "cuda", lambda module: module):
            training_model = training_main.get_model(args)
            criterion = training_main.get_criterion(args)
        self.assertIsInstance(training_model, CMUNeXt_HSPM_SDFR_V2)
        self.assertIsInstance(criterion, SDFRV2Loss)

        inference_model = inference_main.build_model(args, parser=None)
        self.assertIsInstance(inference_model, CMUNeXt_HSPM_SDFR_V2)
        inference_model.load_state_dict(inference_model.state_dict(), strict=True)

        v1_args = Namespace(**vars(args))
        v1_args.model = "CMUNeXt_HSPM_SDFR"
        with mock.patch.object(torch.nn.Module, "cuda", lambda module: module):
            v1_model = training_main.get_model(v1_args)
        self.assertIsInstance(v1_model, CMUNeXt_HSPM_SDFR)
        v1_model.load_state_dict(v1_model.state_dict(), strict=True)


if __name__ == "__main__":
    unittest.main()
