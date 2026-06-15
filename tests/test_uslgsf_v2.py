import sys
import unittest
from unittest import mock

import torch

with mock.patch.object(sys, "argv", ["main.py"]):
    import main as training_main

import infer as inference_main
from src.network.conv_based.CMUNeXt_USLGSF_V2 import (
    USLGSFV2,
    cmunext_uslgsf_v2,
)
from src.utils.losses import BCEDiceLoss


class USLGSFV2Tests(unittest.TestCase):
    def _small_model(self, stages=(0, 1), mode="full", context_downsample=2):
        return cmunext_uslgsf_v2(
            dims=(4, 8, 16, 20, 24),
            depths=(1, 1, 1, 1, 1),
            kernels=(3, 3, 3, 3, 3),
            uslgsf_stages=stages,
            uslgsf_smooth_kernels=(3, 7),
            uslgsf_context_downsample=context_downsample,
            uslgsf_alpha_init=0.05,
            uslgsf_alpha_max=0.5,
            uslgsf_mode=mode,
        )

    @staticmethod
    def _has_nonzero_grad(parameter):
        return parameter.grad is not None and parameter.grad.abs().sum().item() > 0.0

    def test_zero_initialized_residual_is_strict_identity(self):
        module = USLGSFV2(channels=8).eval()
        encoder = torch.randn(2, 8, 17, 19)
        decoder = torch.randn(2, 8, 17, 19)

        with torch.no_grad():
            output = module(encoder, decoder)

        self.assertTrue(torch.equal(output, encoder))
        self.assertEqual(module.last_diagnostics["residual_delta_abs_mean"].item(), 0.0)
        self.assertEqual(
            module.last_diagnostics["injection_encoder_rms_ratio"].item(),
            0.0,
        )

    def test_first_step_updates_projection_then_unlocks_upstream_paths(self):
        module = USLGSFV2(channels=8).train()
        optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
        encoder = torch.randn(2, 8, 16, 16)
        decoder = torch.randn(2, 8, 16, 16)
        target = torch.randn_like(encoder)

        loss = (module(encoder, decoder) - target).square().mean()
        optimizer.zero_grad()
        loss.backward()
        self.assertTrue(self._has_nonzero_grad(module.residual_projection.weight))
        optimizer.step()

        loss = (module(encoder, decoder) - target).square().mean()
        optimizer.zero_grad()
        loss.backward()
        upstream_parameters = (
            module.structure_predictor[0].weight,
            module.encoder_projection[0].weight,
            module.decoder_projection[0].weight,
            module.relevance_predictor[0].weight,
            module.context_projection[0].weight,
            module.detail_projection[0].weight,
            module.fusion[0].weight,
            module.alpha_raw,
        )
        for parameter in upstream_parameters:
            self.assertTrue(self._has_nonzero_grad(parameter))

    def test_decoder_uncertainty_is_bounded_exact_and_detached(self):
        for relevance_value, expected_uncertainty in (
            (0.0, 0.0),
            (0.5, 1.0),
            (1.0, 0.0),
        ):
            relevance = torch.full(
                (2, 1, 8, 8),
                relevance_value,
                requires_grad=True,
            )
            uncertainty = USLGSFV2.build_decoder_uncertainty(relevance)
            self.assertFalse(uncertainty.requires_grad)
            self.assertTrue(
                torch.equal(
                    uncertainty,
                    torch.full_like(uncertainty, expected_uncertainty),
                )
            )

    def test_stage_zero_context_does_not_downsample(self):
        model = self._small_model(stages=(0, 1), context_downsample=3)

        self.assertEqual(model.uslgsf_modules["0"].context_downsample, 1)
        self.assertEqual(model.uslgsf_modules["1"].context_downsample, 3)

    def test_all_modes_keep_contract_and_record_bounded_diagnostics(self):
        encoder = torch.randn(2, 8, 17, 19)
        decoder = torch.randn(2, 8, 17, 19)
        expected_keys = {
            "structure_reliability",
            "decoder_relevance",
            "decoder_uncertainty",
            "active_gate",
            "effective_alpha",
            "residual_delta_abs_mean",
            "injection_encoder_rms_ratio",
        }

        for mode in ("full", "context_only", "structure_only", "relevance_only"):
            module = USLGSFV2(channels=8, mode=mode).eval()
            with torch.no_grad():
                output = module(encoder, decoder)

            self.assertEqual(output.shape, encoder.shape)
            self.assertTrue(torch.isfinite(output).all())
            self.assertEqual(set(module.last_diagnostics), expected_keys)
            structure = module.last_diagnostics["structure_reliability"]
            uncertainty = module.last_diagnostics["decoder_uncertainty"]
            expected_gate = {
                "full": structure * uncertainty,
                "context_only": structure * uncertainty,
                "structure_only": structure,
                "relevance_only": uncertainty,
            }[mode]
            self.assertTrue(
                torch.equal(module.last_diagnostics["active_gate"], expected_gate)
            )
            for key in (
                "structure_reliability",
                "decoder_relevance",
                "decoder_uncertainty",
                "active_gate",
            ):
                value = module.last_diagnostics[key]
                self.assertGreaterEqual(value.min().item(), 0.0)
                self.assertLessEqual(value.max().item(), 1.0)
                self.assertFalse(value.requires_grad)
            for key in (
                "effective_alpha",
                "residual_delta_abs_mean",
                "injection_encoder_rms_ratio",
            ):
                self.assertTrue(torch.isfinite(module.last_diagnostics[key]).all())

    def test_model_modes_keep_output_contract(self):
        image = torch.randn(2, 3, 32, 32)
        for mode in ("full", "context_only", "structure_only", "relevance_only"):
            model = self._small_model(mode=mode).eval()
            with torch.no_grad():
                output = model(image)

            self.assertEqual(output.shape, (2, 1, 32, 32))
            self.assertTrue(torch.isfinite(output).all())
            self.assertEqual(set(model.last_uslgsf_diagnostics), {0, 1})

    def test_parameter_budget_remains_bounded(self):
        model = cmunext_uslgsf_v2()
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        self.assertLessEqual(parameter_count, 3_500_000)

    def test_training_inference_entries_and_strict_reload(self):
        args = training_main.args
        original_model = args.model
        try:
            args.model = "CMUNeXt_USLGSF_V2"
            with mock.patch.object(torch.nn.Module, "cuda", lambda module: module):
                training_model = training_main.get_model(args)
                criterion = training_main.get_criterion(args)
            inference_model = inference_main.build_model(args, parser=None)
        finally:
            args.model = original_model

        self.assertEqual(type(training_model).__name__, "CMUNeXt_USLGSF_V2")
        self.assertEqual(type(inference_model).__name__, "CMUNeXt_USLGSF_V2")
        self.assertIsInstance(criterion, BCEDiceLoss)
        inference_model.load_state_dict(training_model.state_dict(), strict=True)


if __name__ == "__main__":
    unittest.main()
