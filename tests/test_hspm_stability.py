import sys
import unittest
from argparse import Namespace
from unittest import mock

import torch

with mock.patch.object(sys, "argv", ["main.py"]):
    import main as training_main

from src.network.conv_based.CMUNeXt_HSPM import (
    ConfidenceAwarePrototypeMixer,
    cmunext_hspm,
)
from src.utils.losses import HSPMLoss


class HSPMStabilityTests(unittest.TestCase):
    def _small_model(self, mixer_mode, backbone_mode="highres_only", fusion_mode="global"):
        return cmunext_hspm(
            dims=(4, 8, 16, 20, 24),
            depths=(1, 1, 1, 1, 1),
            kernels=(3, 3, 3, 3, 3),
            hspm_mixer_mode=mixer_mode,
            hspm_gamma_init=0.05 if mixer_mode != "legacy" else 0.1,
            hspm_gamma_max=0.3,
            hspm_temperature=0.5 if mixer_mode == "stable" else 0.1,
            hspm_prototype_dropout=0.1 if mixer_mode == "stable" else 0.0,
            hspm_backbone_mode=backbone_mode,
            hspm_fusion_gate_init=0.05,
            hspm_fusion_gate_max=0.3,
            hspm_fusion_mode=fusion_mode,
            hspm_small_area_threshold=0.05,
            hspm_small_area_temperature=0.02,
        )

    def test_modes_keep_output_contract(self):
        image = torch.randn(2, 3, 32, 32)
        for mode in ("legacy", "bounded", "stable"):
            model = self._small_model(mode).eval()
            with torch.no_grad():
                outputs = model(image)
            self.assertEqual(set(outputs), {"seg", "coarse", "uncertainty"})
            self.assertEqual(outputs["seg"].shape, (2, 1, 32, 32))
            self.assertEqual(outputs["coarse"].shape, (2, 1, 4, 4))

    def test_dual_path_keeps_output_contract(self):
        model = self._small_model("legacy", backbone_mode="dual_path").eval()
        with torch.no_grad():
            outputs = model(torch.randn(2, 3, 32, 32))
        self.assertEqual(set(outputs), {"seg", "coarse", "uncertainty"})
        self.assertEqual(outputs["seg"].shape, (2, 1, 32, 32))
        self.assertEqual(outputs["coarse"].shape, (2, 1, 4, 4))

    def test_size_aware_dual_path_keeps_output_contract(self):
        model = self._small_model(
            "legacy",
            backbone_mode="dual_path",
            fusion_mode="size_aware",
        ).eval()
        with torch.no_grad():
            outputs = model(torch.randn(2, 3, 32, 32))
        self.assertEqual(set(outputs), {"seg", "coarse", "uncertainty"})
        self.assertEqual(outputs["seg"].shape, (2, 1, 32, 32))
        self.assertIsNotNone(model.last_fusion_diagnostics)
        self.assertIn("spatial_gate", model.last_fusion_diagnostics)

    def test_explicit_global_fusion_matches_default_dual_path(self):
        default_model = self._small_model("legacy", backbone_mode="dual_path").eval()
        explicit_global = self._small_model(
            "legacy",
            backbone_mode="dual_path",
            fusion_mode="global",
        ).eval()
        explicit_global.load_state_dict(default_model.state_dict(), strict=True)
        image = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            default_outputs = default_model(image)
            explicit_outputs = explicit_global(image)
        self.assertTrue(torch.equal(default_outputs["seg"], explicit_outputs["seg"]))

    def test_dual_path_fusion_gate_initialization_and_bounds(self):
        model = self._small_model("legacy", backbone_mode="dual_path")
        self.assertAlmostEqual(model.effective_fusion_gate().item(), 0.05, places=5)
        for raw_gate in (-100.0, 0.0, 100.0):
            model.fusion_gate_raw.data.fill_(raw_gate)
            effective_gate = model.effective_fusion_gate().item()
            self.assertGreaterEqual(effective_gate, 0.0)
            self.assertLessEqual(effective_gate, 0.300001)

    def test_dual_path_rejects_invalid_fusion_gate_configuration(self):
        for gate_init, gate_max in ((0.0, 0.3), (0.3, 0.3), (0.4, 0.3)):
            with self.assertRaises(ValueError):
                cmunext_hspm(
                    dims=(4, 8, 16, 20, 24),
                    depths=(1, 1, 1, 1, 1),
                    kernels=(3, 3, 3, 3, 3),
                    hspm_backbone_mode="dual_path",
                    hspm_fusion_gate_init=gate_init,
                    hspm_fusion_gate_max=gate_max,
                )

    def test_size_aware_fusion_rejects_invalid_configuration(self):
        for kwargs in (
            {"hspm_fusion_mode": "unsupported"},
            {"hspm_small_area_threshold": 0.0},
            {"hspm_small_area_threshold": 1.0},
            {"hspm_small_area_temperature": 0.0},
        ):
            with self.assertRaises(ValueError):
                cmunext_hspm(
                    dims=(4, 8, 16, 20, 24),
                    depths=(1, 1, 1, 1, 1),
                    kernels=(3, 3, 3, 3, 3),
                    **kwargs,
                )

    def test_size_aware_spatial_gate_is_bounded_detached_and_size_sensitive(self):
        model = self._small_model(
            "legacy",
            backbone_mode="dual_path",
            fusion_mode="size_aware",
        )
        small_logits = torch.full((1, 1, 8, 8), -8.0, requires_grad=True)
        large_logits = torch.full((1, 1, 8, 8), 8.0, requires_grad=True)
        small_uncertainty = 4.0 * torch.sigmoid(small_logits) * (1.0 - torch.sigmoid(small_logits))
        large_uncertainty = 4.0 * torch.sigmoid(large_logits) * (1.0 - torch.sigmoid(large_logits))

        small_gate, smallness, _ = model.build_spatial_gate(small_logits, small_uncertainty)
        large_gate, largeness, _ = model.build_spatial_gate(large_logits, large_uncertainty)

        self.assertGreaterEqual(small_gate.min().item(), 0.0)
        self.assertLessEqual(small_gate.max().item(), 1.0)
        self.assertGreater(smallness.item(), largeness.item())
        self.assertFalse(small_gate.requires_grad)
        self.assertFalse(smallness.requires_grad)
        self.assertFalse(large_gate.requires_grad)

    def test_size_aware_large_lesion_prefers_uncertain_boundary(self):
        model = self._small_model(
            "legacy",
            backbone_mode="dual_path",
            fusion_mode="size_aware",
        )
        coarse_logits = torch.full((1, 1, 8, 8), 8.0)
        coarse_logits[:, :, 0, :] = 0.0
        coarse_prob = torch.sigmoid(coarse_logits)
        uncertainty = 4.0 * coarse_prob * (1.0 - coarse_prob)
        spatial_gate, smallness, _ = model.build_spatial_gate(coarse_logits, uncertainty)

        boundary_gate = spatial_gate[:, :, 0, :].mean()
        interior_gate = spatial_gate[:, :, 1:, :].mean()
        self.assertLess(smallness.item(), 0.01)
        self.assertGreater(boundary_gate.item(), interior_gate.item())

    def test_dual_path_final_loss_updates_both_paths_projection_and_gate(self):
        model = self._small_model("legacy", backbone_mode="dual_path").train()
        outputs = model(torch.randn(2, 3, 32, 32))
        loss = HSPMLoss(coarse_weight=0.3)(
            outputs,
            torch.randint(0, 2, (2, 1, 32, 32)).float(),
        )
        loss.backward()

        self.assertIsNotNone(model.encoder5.up.conv[0].weight.grad)
        self.assertIsNotNone(model.high_resolution_context.input_proj[0].weight.grad)
        self.assertIsNotNone(model.hspm_projection[0].weight.grad)
        self.assertIsNotNone(model.fusion_gate_raw.grad)

    def test_size_aware_final_loss_updates_both_paths_projection_and_gate(self):
        model = self._small_model(
            "legacy",
            backbone_mode="dual_path",
            fusion_mode="size_aware",
        ).train()
        outputs = model(torch.randn(2, 3, 32, 32))
        loss = HSPMLoss(coarse_weight=0.3)(
            outputs,
            torch.randint(0, 2, (2, 1, 32, 32)).float(),
        )
        loss.backward()

        self.assertIsNotNone(model.encoder5.up.conv[0].weight.grad)
        self.assertIsNotNone(model.high_resolution_context.input_proj[0].weight.grad)
        self.assertIsNotNone(model.hspm_projection[0].weight.grad)
        self.assertIsNotNone(model.fusion_gate_raw.grad)

    def test_dual_path_projection_is_signed_and_parameter_budget_is_bounded(self):
        model = cmunext_hspm(hspm_backbone_mode="dual_path").eval()
        with torch.no_grad():
            projected = model.hspm_projection(torch.randn(2, 256, 8, 8))
        self.assertLess(projected.min().item(), 0.0)
        self.assertGreater(projected.max().item(), 0.0)
        self.assertLessEqual(sum(parameter.numel() for parameter in model.parameters()), 3_500_000)

    def test_bounded_gamma_stays_in_range(self):
        mixer = ConfidenceAwarePrototypeMixer(
            channels=8,
            mixer_mode="stable",
            gamma_init=0.05,
            gamma_max=0.3,
        )
        self.assertAlmostEqual(mixer.effective_gamma().item(), 0.05, places=5)
        for raw_gamma in (-100.0, 0.0, 100.0):
            mixer.gamma.data.fill_(raw_gamma)
            effective_gamma = mixer.effective_gamma().item()
            self.assertGreaterEqual(effective_gamma, 0.0)
            self.assertLessEqual(effective_gamma, 0.300001)

    def test_stable_final_loss_does_not_update_coarse_head(self):
        mixer = ConfidenceAwarePrototypeMixer(
            channels=8,
            mixer_mode="stable",
            gamma_init=0.05,
            gamma_max=0.3,
            temperature=0.5,
        )
        feature = torch.randn(2, 8, 8, 8, requires_grad=True)
        enhanced, _, _ = mixer(feature)
        enhanced.mean().backward()
        self.assertIsNone(mixer.coarse_head.weight.grad)
        self.assertIsNone(mixer.coarse_head.bias.grad)

    def test_stable_training_and_inference_smoke(self):
        model = self._small_model("stable")
        criterion = HSPMLoss(coarse_weight=0.3)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        image = torch.randn(2, 3, 32, 32)
        target = torch.randint(0, 2, (2, 1, 32, 32)).float()

        model.train()
        model.prototype_mixer.set_prototype_scale(0.5)
        outputs = model(image)
        loss = criterion(outputs, target, coarse_weight=0.2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        model.prototype_mixer.set_prototype_scale(1.0)
        with torch.no_grad():
            inference_outputs = model(image)
        self.assertTrue(torch.isfinite(inference_outputs["seg"]).all())

    def test_dynamic_coarse_weight_override(self):
        criterion = HSPMLoss(coarse_weight=0.3)
        target = torch.randint(0, 2, (2, 1, 16, 16)).float()
        outputs = {
            "seg": torch.randn(2, 1, 16, 16),
            "coarse": torch.randn(2, 1, 4, 4),
        }
        final_only = criterion.seg_loss(outputs["seg"], target)
        override_loss = criterion(outputs, target, coarse_weight=0.0)
        self.assertTrue(torch.allclose(final_only, override_loss))

    def test_training_schedules(self):
        args = Namespace(
            model="CMUNeXt_HSPM",
            hspm_mixer_mode="stable",
            hspm_coarse_loss_weight=0.3,
            hspm_coarse_loss_final_weight=0.1,
            hspm_coarse_loss_decay_epochs=100,
            hspm_prototype_warmup_epochs=20,
        )
        self.assertAlmostEqual(training_main.get_hspm_coarse_weight(args, 0), 0.3)
        self.assertAlmostEqual(training_main.get_hspm_coarse_weight(args, 50), 0.2)
        self.assertAlmostEqual(training_main.get_hspm_coarse_weight(args, 100), 0.1)
        self.assertAlmostEqual(training_main.get_hspm_prototype_scale(args, 0), 0.0)
        self.assertAlmostEqual(training_main.get_hspm_prototype_scale(args, 10), 0.5)
        self.assertAlmostEqual(training_main.get_hspm_prototype_scale(args, 20), 1.0)

    def test_legacy_state_dict_strictly_loads_into_stable(self):
        legacy = self._small_model("legacy")
        stable = self._small_model("stable")
        stable.load_state_dict(legacy.state_dict(), strict=True)

    def test_highres_only_state_dict_contract_excludes_dual_path_modules(self):
        legacy = self._small_model("legacy")
        rebuilt = self._small_model("legacy")
        rebuilt.load_state_dict(legacy.state_dict(), strict=True)
        self.assertFalse(any(key.startswith("encoder5.") for key in legacy.state_dict()))
        self.assertNotIn("fusion_gate_raw", legacy.state_dict())

    def test_dual_path_state_dict_strictly_loads_across_fusion_modes(self):
        global_model = self._small_model("legacy", backbone_mode="dual_path", fusion_mode="global")
        size_aware_model = self._small_model(
            "legacy",
            backbone_mode="dual_path",
            fusion_mode="size_aware",
        )
        size_aware_model.load_state_dict(global_model.state_dict(), strict=True)
        self.assertNotIn("hspm_small_area_threshold", global_model.state_dict())
        self.assertNotIn("hspm_small_area_temperature", global_model.state_dict())

if __name__ == "__main__":
    unittest.main()
