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
from src.network.conv_based.CMUNeXt_HSPM_UBRD import cmunext_hspm_ubrd
from src.utils.losses import HSPMLoss


class HSPMStabilityTests(unittest.TestCase):
    def _small_model(self, mixer_mode):
        return cmunext_hspm(
            dims=(4, 8, 16, 20, 24),
            depths=(1, 1, 1, 1, 1),
            kernels=(3, 3, 3, 3, 3),
            hspm_mixer_mode=mixer_mode,
            hspm_gamma_init=0.05 if mixer_mode != "legacy" else 0.1,
            hspm_gamma_max=0.3,
            hspm_temperature=0.5 if mixer_mode == "stable" else 0.1,
            hspm_prototype_dropout=0.1 if mixer_mode == "stable" else 0.0,
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

    def test_ubrd_legacy_state_dict_contract_is_unchanged(self):
        legacy_ubrd = cmunext_hspm_ubrd(
            dims=(4, 8, 16, 20, 24),
            depths=(1, 1, 1, 1, 1),
            kernels=(3, 3, 3, 3, 3),
        )
        rebuilt_ubrd = cmunext_hspm_ubrd(
            dims=(4, 8, 16, 20, 24),
            depths=(1, 1, 1, 1, 1),
            kernels=(3, 3, 3, 3, 3),
        )
        rebuilt_ubrd.load_state_dict(legacy_ubrd.state_dict(), strict=True)


if __name__ == "__main__":
    unittest.main()
