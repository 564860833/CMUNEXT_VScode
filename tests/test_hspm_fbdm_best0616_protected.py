import sys
import unittest
from argparse import Namespace
from unittest import mock

import torch

with mock.patch.object(sys, "argv", ["main.py"]):
    import main as training_main

from src.network.conv_based.CMUNeXt_HSPM import cmunext_hspm
from src.network.conv_based.CMUNeXt_HSPM_FBDM_Best0616 import (
    cmunext_hspm_fbdm_best0616,
)
from src.utils.losses import HSPMFBDMLoss


SMALL_DIMS = (4, 8, 16, 20, 24)
SMALL_DEPTHS = (1, 1, 1, 1, 1)
SMALL_KERNELS = (3, 3, 3, 3, 3)


class ProtectedBest0616Tests(unittest.TestCase):
    def _model(self):
        return cmunext_hspm_fbdm_best0616(
            dims=SMALL_DIMS,
            depths=SMALL_DEPTHS,
            kernels=SMALL_KERNELS,
        )

    def _sample(self):
        image = torch.randn(2, 3, 32, 32)
        target = torch.zeros(2, 1, 32, 32)
        target[:, :, 8:24, 10:22] = 1.0
        return image, target

    def test_rng_state_and_hspm_initialization_match_standalone(self):
        torch.manual_seed(41)
        standalone = cmunext_hspm(
            dims=SMALL_DIMS,
            depths=SMALL_DEPTHS,
            kernels=SMALL_KERNELS,
            hspm_mode="full",
            hspm_mixer_mode="bounded",
            hspm_gamma_init=0.1,
            hspm_gamma_max=0.35,
            hspm_backbone_mode="dual_path",
            hspm_fusion_mode="global",
        )
        standalone_rng = torch.get_rng_state().clone()

        torch.manual_seed(41)
        combined = self._model()
        combined_rng = torch.get_rng_state().clone()

        self.assertTrue(torch.equal(standalone_rng, combined_rng))
        combined_state = combined.state_dict()
        for key, value in standalone.state_dict().items():
            self.assertTrue(torch.equal(value, combined_state[key]), key)

    def test_zero_and_disabled_correction_are_exact_identity(self):
        model = self._model().eval()
        image, _ = self._sample()

        with torch.no_grad():
            outputs = model(image)
        self.assertLess(
            (outputs["seg"] - outputs["base_seg"]).abs().max().item(),
            1e-6,
        )

        with torch.no_grad():
            model.fbdm_correction.correction_head.bias.fill_(10.0)
        model.set_fbdm_correction_schedule_scale(0.0)
        with torch.no_grad():
            outputs = model(image)
        self.assertLess(
            (outputs["seg"] - outputs["base_seg"]).abs().max().item(),
            1e-6,
        )

    def test_correction_schedule_and_hard_bound(self):
        args = Namespace(
            model=training_main.HSPM_FBDM_BEST0616_MODEL,
            fbdm_correction_start_epoch=40,
            fbdm_correction_warmup_epochs=40,
        )
        expected = {0: 0.0, 40: 0.0, 60: 0.5, 80: 1.0, 120: 1.0}
        for epoch, scale in expected.items():
            self.assertAlmostEqual(
                training_main.get_fbdm_correction_schedule_scale(args, epoch),
                scale,
            )

        model = self._model().eval()
        model.set_fbdm_correction_schedule_scale(1.0)
        with torch.no_grad():
            model.fbdm_correction.correction_head.bias.fill_(20.0)
            model.fbdm_correction.correction_scale_raw.fill_(20.0)
            outputs = model(self._sample()[0])
        self.assertLessEqual(outputs["logit_correction"].abs().max().item(), 0.10 + 1e-6)

    def test_edge_loss_does_not_reach_hspm(self):
        model = self._model().train()
        image, target = self._sample()
        outputs = model(image)
        _, components = HSPMFBDMLoss(
            coarse_weight=0.0,
            edge_weight=0.03,
            protected_refinement=True,
        )(outputs, target, return_components=True)

        components["edge_weighted"].backward()
        self.assertIsNotNone(model.fbdm1.edge_head.weight.grad)
        self.assertIsNone(model.encoder1.block[0][0].fn[0].weight.grad)
        self.assertIsNone(model.Conv_1x1.weight.grad)

    def test_refine_loss_only_updates_correction_branch(self):
        model = self._model().train()
        model.set_fbdm_correction_schedule_scale(1.0)
        image, target = self._sample()
        outputs = model(image)
        _, components = HSPMFBDMLoss(
            coarse_weight=0.0,
            edge_weight=0.0,
            protected_refinement=True,
            refine_weight=0.5,
            preserve_weight=1.0,
        )(outputs, target, return_components=True)

        (components["refine_weighted"] + components["preserve_weighted"]).backward()
        self.assertGreater(
            model.fbdm_correction.correction_head.weight.grad.abs().sum().item(),
            0.0,
        )
        self.assertIsNone(model.Conv_1x1.weight.grad)
        self.assertIsNone(model.Up_conv2.conv[0].weight.grad)

    def test_base_and_coarse_losses_train_hspm(self):
        model = self._model().train()
        image, target = self._sample()
        outputs = model(image)
        _, components = HSPMFBDMLoss(
            coarse_weight=0.1,
            edge_weight=0.0,
            protected_refinement=True,
        )(outputs, target, return_components=True)

        (components["seg"] + components["coarse_weighted"]).backward()
        self.assertIsNotNone(model.Conv_1x1.weight.grad)
        self.assertIsNotNone(model.prototype_mixer.coarse_head.weight.grad)


if __name__ == "__main__":
    unittest.main()
