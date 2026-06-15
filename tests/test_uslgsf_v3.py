import sys
import unittest
from unittest import mock

import torch

with mock.patch.object(sys, "argv", ["main.py"]):
    import main as training_main

import infer as inference_main
from src.network.conv_based.CMUNeXt_USLGSF_V3 import (
    USLGSFV3,
    cmunext_uslgsf_v3,
)
from src.utils.losses import BCEDiceLoss


class USLGSFV3Tests(unittest.TestCase):
    def _small_model(self, stages=(0, 1), mode="full", context_downsample=2):
        return cmunext_uslgsf_v3(
            dims=(4, 8, 16, 20, 24),
            depths=(1, 1, 1, 1, 1),
            kernels=(3, 3, 3, 3, 3),
            uslgsf_stages=stages,
            uslgsf_smooth_kernels=(3, 7),
            uslgsf_context_downsample=context_downsample,
            uslgsf_alpha_init=0.05,
            uslgsf_alpha_max=0.5,
            uslgsf_mode=mode,
            uslgsf_residual_init_scale=0.05,
        )

    @staticmethod
    def _has_nonzero_grad(parameter):
        return parameter.grad is not None and parameter.grad.abs().sum().item() > 0.0

    def test_route_scale_zero_is_strict_identity(self):
        module = USLGSFV3(channels=8).eval()
        module.set_route_scale(0)
        encoder = torch.randn(2, 8, 17, 19)
        decoder = torch.randn(2, 8, 17, 19)

        with torch.no_grad():
            output = module(encoder, decoder)

        self.assertTrue(torch.equal(output, encoder))
        self.assertEqual(
            module.last_diagnostics["injection_encoder_rms_ratio"].item(),
            0.0,
        )

    def test_dynamic_weights_sum_to_one_and_diagnostics_are_finite(self):
        module = USLGSFV3(channels=8).eval()
        with torch.no_grad():
            module(torch.randn(2, 8, 17, 19), torch.randn(2, 8, 9, 10))

        weights = module.last_mix_weights
        self.assertEqual(weights.shape, (2, 2, 17, 19))
        self.assertTrue(
            torch.allclose(
                weights.sum(dim=1),
                torch.ones_like(weights[:, 0]),
                atol=1e-6,
            )
        )
        self.assertTrue(((weights >= 0.0) & (weights <= 1.0)).all())
        self.assertTrue(((module.last_active_gate >= 0.0) & (module.last_active_gate <= 1.0)).all())
        for value in module.last_diagnostics.values():
            self.assertTrue(torch.isfinite(value).all())

    def test_final_segmentation_loss_trains_all_full_mode_paths(self):
        image = torch.randn(2, 3, 32, 32)
        target = torch.randint(0, 2, (2, 1, 32, 32)).float()
        model = self._small_model(stages=(0,)).train()

        BCEDiceLoss()(model(image), target).backward()

        module = model.uslgsf_modules["0"]
        parameters = (
            module.structure_predictor[0].weight,
            module.encoder_projection[0].weight,
            module.decoder_projection[0].weight,
            module.relevance_predictor[0].weight,
            module.mix_predictor[0].weight,
            module.context_projection[0].weight,
            module.detail_projection[0].weight,
            module.fusion[0].weight,
            module.residual_projection.weight,
            module.alpha_raw,
        )
        for parameter in parameters:
            self.assertTrue(self._has_nonzero_grad(parameter))

    def test_modes_stage_limits_and_output_contract(self):
        expected_diagnostics = {
            "structure_reliability_mean",
            "decoder_relevance_mean",
            "structure_weight_mean",
            "relevance_weight_mean",
            "active_gate_mean",
            "route_scale",
            "effective_alpha",
            "residual_delta_abs_mean",
            "injection_encoder_rms_ratio",
        }
        for mode in ("full", "context_only", "structure_only", "relevance_only"):
            model = self._small_model(mode=mode, context_downsample=3).eval()
            with torch.no_grad():
                outputs = model(torch.randn(2, 3, 32, 32))

            self.assertIsInstance(outputs, torch.Tensor)
            self.assertEqual(outputs.shape, (2, 1, 32, 32))
            self.assertEqual(model.uslgsf_modules["0"].context_downsample, 1)
            self.assertEqual(model.uslgsf_modules["1"].context_downsample, 3)
            for diagnostics in model.last_uslgsf_diagnostics.values():
                self.assertEqual(set(diagnostics), expected_diagnostics)
                for value in diagnostics.values():
                    self.assertTrue(torch.isfinite(value).all())

        for invalid_stages in ((2,), (3,), (0, 2)):
            with self.assertRaises(ValueError):
                self._small_model(stages=invalid_stages)

    def test_v3_has_no_coarse_or_uncertainty_interface(self):
        module = USLGSFV3(channels=8)
        self.assertFalse(hasattr(module, "decoder_coarse_head"))
        self.assertFalse(hasattr(module, "build_decoder_uncertainty"))
        self.assertFalse(hasattr(training_main.losses, "USLGSFV3Loss"))

    def test_parameter_budget_remains_bounded(self):
        model = cmunext_uslgsf_v3()
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        self.assertLessEqual(parameter_count, 3_500_000)

    def test_training_inference_entries_standard_loss_and_strict_reload(self):
        args = training_main.args
        original_model = args.model
        try:
            args.model = "CMUNeXt_USLGSF_V3"
            with mock.patch.object(torch.nn.Module, "cuda", lambda module: module):
                training_model = training_main.get_model(args)
                criterion = training_main.get_criterion(args)
            inference_model = inference_main.build_model(args, parser=None)
        finally:
            args.model = original_model

        self.assertEqual(type(training_model).__name__, "CMUNeXt_USLGSF_V3")
        self.assertEqual(type(inference_model).__name__, "CMUNeXt_USLGSF_V3")
        self.assertIsInstance(criterion, BCEDiceLoss)
        inference_model.load_state_dict(training_model.state_dict(), strict=True)


if __name__ == "__main__":
    unittest.main()
