import sys
import unittest
from unittest import mock

import torch

with mock.patch.object(sys, "argv", ["main.py"]):
    import main as training_main

import infer as inference_main
from src.network.conv_based.CMUNeXt_USLGSF import USLGSF, cmunext_uslgsf
from src.utils.losses import BCEDiceLoss


class USLGSFTests(unittest.TestCase):
    def _small_model(self, stages=(0, 1), mode="full"):
        return cmunext_uslgsf(
            dims=(4, 8, 16, 20, 24),
            depths=(1, 1, 1, 1, 1),
            kernels=(3, 3, 3, 3, 3),
            uslgsf_stages=stages,
            uslgsf_smooth_kernels=(3, 7),
            uslgsf_context_downsample=2,
            uslgsf_alpha_init=0.05,
            uslgsf_alpha_max=0.5,
            uslgsf_mode=mode,
        )

    def test_module_keeps_shape_and_records_bounded_diagnostics(self):
        module = USLGSF(channels=8).eval()
        with torch.no_grad():
            output = module(
                torch.randn(2, 8, 17, 19),
                torch.randn(2, 8, 17, 19),
            )

        self.assertEqual(output.shape, (2, 8, 17, 19))
        self.assertEqual(
            set(module.last_diagnostics),
            {
                "structure_reliability",
                "decoder_relevance",
                "final_reliability",
                "effective_alpha",
            },
        )
        for key in ("structure_reliability", "decoder_relevance", "final_reliability"):
            value = module.last_diagnostics[key]
            self.assertGreaterEqual(value.min().item(), 0.0)
            self.assertLessEqual(value.max().item(), 1.0)
            self.assertFalse(value.requires_grad)

    def test_effective_alpha_initialization_and_bounds(self):
        module = USLGSF(channels=8, alpha_init=0.05, alpha_max=0.5)
        self.assertAlmostEqual(module.effective_alpha().item(), 0.05, places=5)
        for raw_alpha in (-100.0, 0.0, 100.0):
            module.alpha_raw.data.fill_(raw_alpha)
            effective_alpha = module.effective_alpha().item()
            self.assertGreaterEqual(effective_alpha, 0.0)
            self.assertLessEqual(effective_alpha, 0.500001)

    def test_default_model_refines_only_shallow_skips(self):
        model = self._small_model().eval()
        with torch.no_grad():
            output = model(torch.randn(2, 3, 32, 32))

        self.assertEqual(output.shape, (2, 1, 32, 32))
        self.assertEqual(set(model.uslgsf_modules), {"0", "1"})
        self.assertEqual(set(model.last_uslgsf_diagnostics), {0, 1})
        self.assertEqual(
            model.last_uslgsf_diagnostics[0]["final_reliability"].shape,
            (2, 1, 32, 32),
        )
        self.assertEqual(
            model.last_uslgsf_diagnostics[1]["final_reliability"].shape,
            (2, 1, 16, 16),
        )

    def test_all_ablation_modes_keep_output_contract(self):
        image = torch.randn(2, 3, 32, 32)
        for mode in ("full", "context_only", "structure_only", "relevance_only"):
            model = self._small_model(mode=mode).eval()
            with torch.no_grad():
                output = model(image)
            self.assertEqual(output.shape, (2, 1, 32, 32))
            self.assertTrue(torch.isfinite(output).all())

    def test_full_model_loss_updates_all_selection_paths(self):
        model = self._small_model(stages=(0,)).train()
        output = model(torch.randn(2, 3, 32, 32))
        target = torch.randint(0, 2, output.shape).float()
        BCEDiceLoss()(output, target).backward()

        module = model.uslgsf_modules["0"]
        self.assertIsNotNone(module.structure_predictor[0].weight.grad)
        self.assertIsNotNone(module.encoder_projection[0].weight.grad)
        self.assertIsNotNone(module.decoder_projection[0].weight.grad)
        self.assertIsNotNone(module.relevance_predictor[0].weight.grad)
        self.assertIsNotNone(module.context_projection[0].weight.grad)
        self.assertIsNotNone(module.detail_projection[0].weight.grad)
        self.assertIsNotNone(module.fusion[0].weight.grad)
        self.assertIsNotNone(module.alpha_raw.grad)

    def test_stage_configuration_supports_deeper_ablation(self):
        model = self._small_model(stages=(0, 1, 2)).eval()
        with torch.no_grad():
            model(torch.randn(2, 3, 32, 32))
        self.assertEqual(set(model.last_uslgsf_diagnostics), {0, 1, 2})

    def test_invalid_configuration_is_rejected(self):
        invalid_module_kwargs = (
            {"smooth_kernels": (3,)},
            {"smooth_kernels": (4, 7)},
            {"smooth_kernels": (7, 3)},
            {"context_downsample": 0},
            {"alpha_init": 0.5, "alpha_max": 0.5},
            {"mode": "unsupported"},
        )
        for kwargs in invalid_module_kwargs:
            with self.assertRaises(ValueError):
                USLGSF(channels=8, **kwargs)

        with self.assertRaises(ValueError):
            self._small_model(stages=(4,))
        with self.assertRaises(ValueError):
            self._small_model(stages=())

    def test_cli_parsers_validate_stages_and_smoothing_kernels(self):
        self.assertEqual(training_main.parse_uslgsf_stages("0,1,1,2"), (0, 1, 2))
        self.assertEqual(training_main.parse_uslgsf_smooth_kernels("3,7"), (3, 7))
        for value in ("", "0,4"):
            with self.assertRaises(Exception):
                training_main.parse_uslgsf_stages(value)
        for value in ("3", "4,7", "7,3"):
            with self.assertRaises(Exception):
                training_main.parse_uslgsf_smooth_kernels(value)

    def test_parameter_budget_remains_bounded(self):
        model = cmunext_uslgsf()
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        self.assertLessEqual(parameter_count, 3_500_000)

    def test_training_inference_entries_and_strict_reload(self):
        args = training_main.args
        original_model = args.model
        try:
            args.model = "CMUNeXt_USLGSF"
            with mock.patch.object(torch.nn.Module, "cuda", lambda module: module):
                training_model = training_main.get_model(args)
                criterion = training_main.get_criterion(args)
            inference_model = inference_main.build_model(args, parser=None)
        finally:
            args.model = original_model

        self.assertEqual(type(training_model).__name__, "CMUNeXt_USLGSF")
        self.assertEqual(type(inference_model).__name__, "CMUNeXt_USLGSF")
        self.assertIsInstance(criterion, BCEDiceLoss)
        inference_model.load_state_dict(training_model.state_dict(), strict=True)


if __name__ == "__main__":
    unittest.main()
