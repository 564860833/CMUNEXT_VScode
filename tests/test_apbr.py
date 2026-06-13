import sys
import unittest
from argparse import Namespace
from unittest import mock

import torch

with mock.patch.object(sys, "argv", ["main.py"]):
    import main as training_main

import infer as inference_main
from src.network.conv_based.CMUNeXt_HSPM import cmunext_hspm
from src.network.conv_based.CMUNeXt_HSPM_APBR import (
    AmbiguityProgressiveBoundaryRefinement,
    cmunext_hspm_apbr,
)
from src.utils.losses import APBRLoss


class APBRTests(unittest.TestCase):
    def _small_model(self, mode="full", mixer_mode="stable"):
        return cmunext_hspm_apbr(
            dims=(4, 8, 16, 20, 24),
            depths=(1, 1, 1, 1, 1),
            kernels=(3, 3, 3, 3, 3),
            hspm_mixer_mode=mixer_mode,
            hspm_gamma_init=0.05 if mixer_mode != "legacy" else 0.1,
            hspm_gamma_max=0.3,
            hspm_temperature=0.5 if mixer_mode == "stable" else 0.1,
            hspm_fusion_mode="size_aware",
            apbr_mode=mode,
        )

    def test_output_contract(self):
        model = self._small_model().eval()
        with torch.no_grad():
            outputs = model(torch.randn(2, 3, 32, 32))
        self.assertEqual(set(outputs), {"seg", "coarse", "uncertainty", "refine_half"})
        self.assertEqual(outputs["coarse"].shape, (2, 1, 4, 4))
        self.assertEqual(outputs["refine_half"].shape, (2, 1, 16, 16))
        self.assertEqual(outputs["seg"].shape, (2, 1, 32, 32))

    def test_modes_support_forward_and_backward(self):
        target = torch.randint(0, 2, (2, 1, 32, 32)).float()
        for mode in ("full", "no_ambiguity", "no_detail"):
            model = self._small_model(mode=mode).train()
            outputs = model(torch.randn(2, 3, 32, 32))
            loss = APBRLoss()(outputs, target)
            loss.backward()
            self.assertIsNotNone(model.apbr_full.refine[0].weight.grad)

    def test_rejects_invalid_mode(self):
        with self.assertRaises(ValueError):
            self._small_model(mode="unsupported")

    def test_route_is_bounded_and_detached(self):
        module = AmbiguityProgressiveBoundaryRefinement(channels=8, gate_kernel=5)
        previous_logits = torch.randn(2, 1, 8, 8, requires_grad=True)
        probability, gate = module.build_route(previous_logits, (16, 16))
        self.assertFalse(probability.requires_grad)
        self.assertFalse(gate.requires_grad)
        self.assertGreaterEqual(gate.min().item(), 0.0)
        self.assertLessEqual(gate.max().item(), 1.0)

    def test_active_gate_applies_route_scale(self):
        module = AmbiguityProgressiveBoundaryRefinement(channels=8, gate_kernel=5)
        module.set_route_scale(0.25)
        module(
            torch.randn(2, 8, 16, 16),
            torch.randn(2, 8, 16, 16),
            torch.randn(2, 1, 8, 8),
        )
        self.assertTrue(
            torch.allclose(module.last_active_gate, 0.25 * module.last_raw_gate)
        )

    def test_route_diagnostics_are_finite_and_bounded(self):
        model = self._small_model().eval()
        model.set_apbr_route_scale(0.5)
        with torch.no_grad():
            model(torch.randn(2, 3, 32, 32))
        diagnostics = model.get_apbr_diagnostics()
        self.assertIsNotNone(diagnostics)
        for stage in ("half", "full"):
            self.assertAlmostEqual(diagnostics[f"{stage}_route_scale"].item(), 0.5)
            self.assertTrue(
                torch.isfinite(diagnostics[f"{stage}_active_gate_mean"])
            )
            self.assertGreaterEqual(diagnostics[f"{stage}_raw_gate_mean"].item(), 0.0)
            self.assertLessEqual(diagnostics[f"{stage}_raw_gate_mean"].item(), 1.0)
            self.assertGreaterEqual(diagnostics[f"{stage}_raw_gate_over_05"].item(), 0.0)
            self.assertLessEqual(diagnostics[f"{stage}_raw_gate_over_05"].item(), 1.0)
            self.assertGreaterEqual(diagnostics[f"{stage}_raw_gate_over_08"].item(), 0.0)
            self.assertLessEqual(diagnostics[f"{stage}_raw_gate_over_08"].item(), 1.0)

    def test_no_ambiguity_uses_full_gate(self):
        module = AmbiguityProgressiveBoundaryRefinement(
            channels=8,
            gate_kernel=5,
            mode="no_ambiguity",
        )
        _, gate = module.build_route(torch.randn(2, 1, 8, 8), (16, 16))
        self.assertTrue(torch.equal(gate, torch.ones_like(gate)))

    def test_no_detail_does_not_update_detail_extractor(self):
        module = AmbiguityProgressiveBoundaryRefinement(
            channels=8,
            gate_kernel=3,
            mode="no_detail",
        )
        refined_feature, refined_logits = module(
            torch.randn(2, 8, 16, 16),
            torch.randn(2, 8, 16, 16),
            torch.randn(2, 1, 8, 8),
        )
        (refined_feature.mean() + refined_logits.mean()).backward()
        self.assertIsNone(module.detail_extractor[0].weight.grad)

    def test_residual_scales_are_initialized_and_bounded(self):
        module = AmbiguityProgressiveBoundaryRefinement(channels=8, gate_kernel=3)
        self.assertAlmostEqual(module.effective_feature_scale().item(), 0.05, places=5)
        self.assertAlmostEqual(module.effective_logit_scale().item(), 0.1, places=5)
        for raw_value in (-100.0, 0.0, 100.0):
            module.feature_scale_raw.data.fill_(raw_value)
            module.logit_scale_raw.data.fill_(raw_value)
            self.assertGreaterEqual(module.effective_feature_scale().item(), 0.0)
            self.assertLessEqual(module.effective_feature_scale().item(), 0.300001)
            self.assertGreaterEqual(module.effective_logit_scale().item(), 0.0)
            self.assertLessEqual(module.effective_logit_scale().item(), 1.000001)

    def test_apbr_loss_updates_all_paths(self):
        model = self._small_model().train()
        outputs = model(torch.randn(2, 3, 32, 32))
        target = torch.randint(0, 2, (2, 1, 32, 32)).float()
        APBRLoss()(outputs, target).backward()

        self.assertIsNotNone(model.encoder5.up.conv[0].weight.grad)
        self.assertIsNotNone(model.high_resolution_context.input_proj[0].weight.grad)
        self.assertIsNotNone(model.hspm_projection[0].weight.grad)
        self.assertIsNotNone(model.apbr_half.refine[0].weight.grad)
        self.assertIsNotNone(model.apbr_full.refine[0].weight.grad)
        self.assertIsNotNone(model.apbr_half.feature_scale_raw.grad)
        self.assertIsNotNone(model.apbr_full.feature_scale_raw.grad)
        self.assertIsNotNone(model.apbr_half.logit_scale_raw.grad)
        self.assertIsNotNone(model.apbr_full.logit_scale_raw.grad)

    def test_final_prediction_route_does_not_update_coarse_head(self):
        model = self._small_model(mixer_mode="stable").train()
        outputs = model(torch.randn(2, 3, 32, 32))
        outputs["seg"].mean().backward()
        self.assertIsNone(model.prototype_mixer.coarse_head.weight.grad)
        self.assertIsNone(model.prototype_mixer.coarse_head.bias.grad)

    def test_coarse_weight_schedule(self):
        args = Namespace(
            apbr_coarse_loss_weight=0.1,
            apbr_coarse_loss_final_weight=0.02,
            apbr_coarse_loss_decay_epochs=150,
        )
        self.assertAlmostEqual(training_main.get_apbr_coarse_weight(args, 0), 0.1)
        self.assertAlmostEqual(training_main.get_apbr_coarse_weight(args, 75), 0.06)
        self.assertAlmostEqual(training_main.get_apbr_coarse_weight(args, 150), 0.02)
        self.assertAlmostEqual(training_main.get_apbr_coarse_weight(args, 300), 0.02)

    def test_route_scale_schedule(self):
        args = Namespace(apbr_route_warmup_epochs=30)
        self.assertAlmostEqual(training_main.get_apbr_route_scale(args, 0), 1.0 / 30.0)
        self.assertAlmostEqual(training_main.get_apbr_route_scale(args, 14), 0.5)
        self.assertAlmostEqual(training_main.get_apbr_route_scale(args, 29), 1.0)
        self.assertAlmostEqual(training_main.get_apbr_route_scale(args, 300), 1.0)
        args.apbr_route_warmup_epochs = 0
        self.assertAlmostEqual(training_main.get_apbr_route_scale(args, 0), 1.0)

    def test_apbr_loss_components_sum_to_total(self):
        outputs = self._small_model().eval()(torch.randn(2, 3, 32, 32))
        target = torch.randint(0, 2, (2, 1, 32, 32)).float()
        total, components = APBRLoss(boundary_weight=0.1)(
            outputs,
            target,
            return_components=True,
        )
        component_sum = (
            components["seg"]
            + components["coarse_weighted"]
            + components["intermediate_weighted"]
            + components["boundary_weighted"]
        )
        self.assertTrue(torch.allclose(total, components["total"]))
        self.assertTrue(torch.allclose(total, component_sum))

    def test_disabled_boundary_loss_component_is_zero(self):
        outputs = self._small_model().eval()(torch.randn(2, 3, 32, 32))
        target = torch.randint(0, 2, (2, 1, 32, 32)).float()
        _, components = APBRLoss(boundary_weight=0.0)(
            outputs,
            target,
            return_components=True,
        )
        self.assertEqual(components["boundary_weighted"].item(), 0.0)

    def test_apbr_parameter_overhead_is_bounded(self):
        baseline = cmunext_hspm(hspm_backbone_mode="dual_path")
        apbr = cmunext_hspm_apbr()
        baseline_parameters = sum(parameter.numel() for parameter in baseline.parameters())
        apbr_parameters = sum(parameter.numel() for parameter in apbr.parameters())
        self.assertLessEqual(apbr_parameters - baseline_parameters, 100_000)

    def test_training_schedule_supports_apbr_stable_mode(self):
        model = self._small_model().train()
        args = Namespace(
            model="CMUNeXt_HSPM_APBR",
            hspm_mixer_mode="stable",
            hspm_prototype_warmup_epochs=20,
            apbr_coarse_loss_weight=0.1,
            apbr_coarse_loss_final_weight=0.02,
            apbr_coarse_loss_decay_epochs=150,
            apbr_route_warmup_epochs=30,
        )
        coarse_weight, prototype_scale, effective_gamma = training_main.configure_hspm_epoch(
            args,
            model,
            10,
        )
        self.assertAlmostEqual(coarse_weight, 0.1 - 0.08 * (10.0 / 150.0))
        self.assertAlmostEqual(prototype_scale, 0.5)
        self.assertAlmostEqual(effective_gamma, 0.025, places=5)
        self.assertAlmostEqual(model.apbr_half.route_scale, 11.0 / 30.0)
        self.assertAlmostEqual(model.apbr_full.route_scale, 11.0 / 30.0)

    def test_inference_entry_builds_apbr_on_cpu(self):
        args = Namespace(
            model="CMUNeXt_HSPM_APBR",
            num_classes=1,
            hspm_mode="full",
            hspm_mixer_mode="stable",
            hspm_gamma_init=0.05,
            hspm_gamma_max=0.3,
            hspm_temperature=0.5,
            hspm_prototype_dropout=0.0,
            hspm_fusion_gate_init=0.05,
            hspm_fusion_gate_max=0.3,
            hspm_fusion_mode="global",
            hspm_small_area_threshold=0.05,
            hspm_small_area_temperature=0.02,
            apbr_mode="full",
        )
        model = inference_main.build_model(args, parser=None).eval()
        with torch.no_grad():
            outputs = model(torch.randn(2, 3, 32, 32))
        self.assertEqual(outputs["seg"].shape, (2, 1, 32, 32))

    def test_training_entry_selects_apbr_loss(self):
        args = Namespace(
            model="CMUNeXt_HSPM_APBR",
            apbr_coarse_loss_weight=0.1,
            apbr_intermediate_loss_weight=0.15,
            apbr_boundary_loss_weight=0.1,
        )
        with mock.patch.object(torch.nn.Module, "cuda", lambda module: module):
            criterion = training_main.get_criterion(args)
        self.assertIsInstance(criterion, APBRLoss)

    def test_training_entry_builds_apbr_on_cpu(self):
        args = Namespace(**vars(training_main.args))
        args.model = "CMUNeXt_HSPM_APBR"
        args.hspm_mixer_mode = "stable"
        args.hspm_gamma_init = 0.05
        args.hspm_temperature = 0.5
        args.apbr_mode = "full"
        with mock.patch.object(torch.nn.Module, "cuda", lambda module: module):
            model = training_main.get_model(args).eval()
        with torch.no_grad():
            outputs = model(torch.randn(2, 3, 32, 32))
        self.assertEqual(outputs["seg"].shape, (2, 1, 32, 32))


if __name__ == "__main__":
    unittest.main()
