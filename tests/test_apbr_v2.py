import sys
import unittest
from argparse import Namespace
from unittest import mock

import torch

with mock.patch.object(sys, "argv", ["main.py"]):
    import main as training_main

import infer as inference_main
from src.network.conv_based.CMUNeXt_HSPM_APBR import cmunext_hspm_apbr
from src.network.conv_based.CMUNeXt_HSPM_APBR_V2 import (
    CMUNeXt_HSPM_APBR_V2,
    CurrentBaselineProgressiveRefinement,
    cmunext_hspm_apbr_v2,
)
from src.utils.losses import APBRLoss
from tools.evaluate_apbr_oracle_recovery import build_model as build_oracle_model


class APBRV2Tests(unittest.TestCase):
    def _small_model(self, mode="full"):
        return cmunext_hspm_apbr_v2(
            dims=(4, 8, 16, 20, 24),
            depths=(1, 1, 1, 1, 1),
            kernels=(3, 3, 3, 3, 3),
            hspm_mixer_mode="stable",
            hspm_gamma_init=0.05,
            hspm_gamma_max=0.3,
            hspm_temperature=0.5,
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

    def test_correction_heads_are_zero_initialized(self):
        model = self._small_model()
        for module in (model.apbr_half, model.apbr_full):
            self.assertEqual(module.correction_head.weight.count_nonzero().item(), 0)
            self.assertEqual(module.correction_head.bias.count_nonzero().item(), 0)

    def test_route_scale_zero_equals_current_pre_base(self):
        module = CurrentBaselineProgressiveRefinement(
            channels=8,
            gate_kernel=3,
        ).eval()
        module.set_route_scale(0.0)
        decoder_feature = torch.randn(2, 8, 16, 16)
        with torch.no_grad():
            refined_feature, refined_logits = module(
                decoder_feature,
                torch.randn(2, 8, 16, 16),
                torch.randn(2, 1, 8, 8),
            )
            expected_logits = module.base_head(decoder_feature)
        self.assertTrue(torch.equal(refined_feature, decoder_feature))
        self.assertTrue(torch.equal(refined_logits, expected_logits))

    def test_route_uses_current_base_not_previous_probability(self):
        module = CurrentBaselineProgressiveRefinement(
            channels=8,
            gate_kernel=3,
        ).eval()
        module.base_head.weight.data.zero_()
        module.base_head.bias.data.zero_()
        decoder_feature = torch.randn(2, 8, 16, 16)
        skip_feature = torch.randn(2, 8, 16, 16)
        with torch.no_grad():
            module(
                decoder_feature,
                skip_feature,
                torch.full((2, 1, 8, 8), -10.0),
            )
            negative_previous_gate = module.last_raw_gate.clone()
            module(
                decoder_feature,
                skip_feature,
                torch.full((2, 1, 8, 8), 10.0),
            )
            positive_previous_gate = module.last_raw_gate.clone()
        self.assertTrue(torch.equal(negative_previous_gate, positive_previous_gate))
        self.assertTrue(torch.equal(positive_previous_gate, torch.ones_like(positive_previous_gate)))

    def test_correction_is_unbounded_and_can_exceed_one(self):
        module = CurrentBaselineProgressiveRefinement(
            channels=8,
            gate_kernel=3,
            mode="no_ambiguity",
        ).eval()
        module.correction_head.weight.data.zero_()
        module.correction_head.bias.data.fill_(2.5)
        with torch.no_grad():
            module(
                torch.randn(2, 8, 16, 16),
                torch.randn(2, 8, 16, 16),
                torch.randn(2, 1, 8, 8),
            )
        logit_delta = module.last_refined_logits - module.last_feature_refined_logits
        self.assertTrue(torch.allclose(logit_delta, torch.full_like(logit_delta, 2.5)))
        self.assertGreater(module.last_correction_logits.abs().max().item(), 1.0)

    def test_modes_support_forward_and_backward(self):
        target = torch.randint(0, 2, (2, 1, 32, 32)).float()
        for mode in ("full", "no_ambiguity", "no_detail"):
            model = self._small_model(mode=mode).train()
            outputs = model(torch.randn(2, 3, 32, 32))
            APBRLoss()(outputs, target).backward()
            self.assertIsNotNone(model.apbr_full.refine[0].weight.grad)
            self.assertIsNotNone(model.apbr_full.correction_head.weight.grad)
            self.assertIsNotNone(model.apbr_full.feature_scale_raw.grad)
            if mode == "no_detail":
                self.assertIsNone(model.apbr_full.detail_extractor[0].weight.grad)

    def test_feature_and_correction_paths_are_gated(self):
        module = CurrentBaselineProgressiveRefinement(
            channels=8,
            gate_kernel=3,
            mode="no_ambiguity",
        ).train()
        module.set_route_scale(0.0)
        _, logits = module(
            torch.randn(2, 8, 16, 16),
            torch.randn(2, 8, 16, 16),
            torch.randn(2, 1, 8, 8),
        )
        logits.mean().backward()
        self.assertEqual(module.feature_scale_raw.grad.item(), 0.0)
        self.assertEqual(module.correction_head.weight.grad.count_nonzero().item(), 0)
        self.assertEqual(module.detail_extractor[0].weight.grad.count_nonzero().item(), 0)

    def test_oracle_target_uses_current_base_prediction(self):
        module = CurrentBaselineProgressiveRefinement(
            channels=8,
            gate_kernel=3,
        ).eval()
        module.base_head.weight.data.zero_()
        module.base_head.bias.data.fill_(-10.0)
        with torch.no_grad():
            module(
                torch.randn(1, 8, 8, 8),
                torch.randn(1, 8, 8, 8),
                torch.full((1, 1, 4, 4), 10.0),
                oracle_target=torch.ones(1, 1, 8, 8),
            )
        self.assertGreater(module.last_oracle_recovery_target.min().item(), 0.99)
        self.assertGreater(module.last_oracle_recovery_gate.min().item(), 0.99)

    def test_oracle_forward_selects_requested_stages(self):
        model = self._small_model().eval()
        x = torch.randn(2, 3, 32, 32)
        target = torch.randint(0, 2, (2, 1, 32, 32)).float()
        with torch.no_grad():
            model.forward_oracle_recovery(x, target, oracle_stages={"half"})
        self.assertGreater(model.apbr_half.last_oracle_recovery_target.mean().item(), 0.0)
        self.assertEqual(model.apbr_full.last_oracle_recovery_target.sum().item(), 0.0)

        with torch.no_grad():
            model.forward_oracle_recovery(x, target, oracle_stages={"full"})
        self.assertEqual(model.apbr_half.last_oracle_recovery_target.sum().item(), 0.0)
        self.assertGreater(model.apbr_full.last_oracle_recovery_target.mean().item(), 0.0)

    def test_diagnostics_are_finite_and_gates_are_bounded(self):
        model = self._small_model().eval()
        with torch.no_grad():
            model(torch.randn(2, 3, 32, 32))
        diagnostics = model.get_apbr_diagnostics()
        gate_names = (
            "raw_gate_mean",
            "active_gate_mean",
            "ambiguity_gate_mean",
            "oracle_recovery_target_mean",
            "oracle_recovery_gate_mean",
            "combined_gate_mean",
            "recovery_added_mean",
            "recovery_dominant_ratio",
            "base_probability_mean",
        )
        for stage_name in ("half", "full"):
            for diagnostic_name in gate_names:
                value = diagnostics[f"{stage_name}_{diagnostic_name}"]
                self.assertTrue(torch.isfinite(value))
                self.assertGreaterEqual(value.item(), 0.0)
                self.assertLessEqual(value.item(), 1.0)
            for diagnostic_name in (
                "feature_logit_delta_abs_mean",
                "correction_logit_abs_mean",
                "correction_logit_abs_p99",
                "correction_logit_abs_max",
                "total_logit_delta_abs_mean",
                "effective_feature_scale",
            ):
                value = diagnostics[f"{stage_name}_{diagnostic_name}"]
                self.assertTrue(torch.isfinite(value))
                self.assertGreaterEqual(value.item(), 0.0)

    def test_apbr_loss_components_sum_to_total(self):
        outputs = self._small_model().eval()(torch.randn(2, 3, 32, 32))
        target = torch.randint(0, 2, (2, 1, 32, 32)).float()
        total, components = APBRLoss(boundary_weight=0.1)(
            outputs,
            target,
            return_components=True,
        )
        component_sum = sum(
            components[name]
            for name in (
                "seg",
                "coarse_weighted",
                "intermediate_weighted",
                "boundary_weighted",
            )
        )
        self.assertTrue(torch.allclose(total, components["total"]))
        self.assertTrue(torch.allclose(total, component_sum))

    def test_training_entry_builds_v2_and_selects_apbr_loss(self):
        args = Namespace(**vars(training_main.args))
        args.model = "CMUNeXt_HSPM_APBR_V2"
        args.hspm_mixer_mode = "stable"
        args.hspm_gamma_init = 0.05
        args.hspm_temperature = 0.5
        args.apbr_mode = "full"
        with mock.patch.object(torch.nn.Module, "cuda", lambda module: module):
            model = training_main.get_model(args).eval()
            criterion = training_main.get_criterion(args)
        with torch.no_grad():
            outputs = model(torch.randn(2, 3, 32, 32))
        self.assertEqual(outputs["seg"].shape, (2, 1, 32, 32))
        self.assertIsInstance(criterion, APBRLoss)

    def test_training_schedule_applies_to_v2(self):
        model = self._small_model().train()
        args = Namespace(
            model="CMUNeXt_HSPM_APBR_V2",
            hspm_mixer_mode="stable",
            hspm_prototype_warmup_epochs=20,
            apbr_coarse_loss_weight=0.1,
            apbr_coarse_loss_final_weight=0.02,
            apbr_coarse_loss_decay_epochs=150,
            apbr_route_warmup_epochs=30,
        )
        training_main.configure_hspm_epoch(args, model, 10)
        self.assertAlmostEqual(model.apbr_half.route_scale, 11.0 / 30.0)
        self.assertAlmostEqual(model.apbr_full.route_scale, 11.0 / 30.0)

    def test_oracle_tool_builds_v2_from_config(self):
        model = build_oracle_model(
            {
                "model": "CMUNeXt_HSPM_APBR_V2",
                "hspm_mixer_mode": "stable",
                "hspm_gamma_init": 0.05,
                "hspm_temperature": 0.5,
            }
        )
        self.assertIsInstance(model, CMUNeXt_HSPM_APBR_V2)

    def test_inference_entry_builds_v2_on_cpu(self):
        args = Namespace(
            model="CMUNeXt_HSPM_APBR_V2",
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

    def test_v1_state_dict_remains_strictly_compatible(self):
        v1 = cmunext_hspm_apbr(
            dims=(4, 8, 16, 20, 24),
            depths=(1, 1, 1, 1, 1),
            kernels=(3, 3, 3, 3, 3),
        )
        rebuilt_v1 = cmunext_hspm_apbr(
            dims=(4, 8, 16, 20, 24),
            depths=(1, 1, 1, 1, 1),
            kernels=(3, 3, 3, 3, 3),
        )
        rebuilt_v1.load_state_dict(v1.state_dict(), strict=True)


if __name__ == "__main__":
    unittest.main()
