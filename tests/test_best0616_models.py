import sys
import unittest
from argparse import Namespace
from unittest import mock

import torch

with mock.patch.object(sys, "argv", ["main.py"]):
    import main as training_main
import infer as inference_main

from src.network.conv_based.CMUNeXt_HSPM import cmunext_hspm
from src.network.conv_based.CMUNeXt_HSPM_Best0616 import cmunext_hspm_best0616
from src.network.conv_based.CMUNeXt_HSPM_Best0619 import cmunext_hspm_best0619


SMALL_DIMS = (4, 8, 16, 20, 24)
SMALL_DEPTHS = (1, 1, 1, 1, 1)
SMALL_KERNELS = (3, 3, 3, 3, 3)


class Best0616ModelTests(unittest.TestCase):
    def test_inference_can_select_base_logits(self):
        outputs = {
            "seg": torch.ones(1, 1, 4, 4),
            "base_seg": torch.zeros(1, 1, 4, 4),
        }
        self.assertTrue(
            torch.equal(
                inference_main.get_seg_logits(outputs, use_base_seg=True),
                outputs["base_seg"],
            )
        )

    def test_hspm_best0616_output_contract_and_fixed_structure(self):
        model = cmunext_hspm_best0616(
            dims=SMALL_DIMS,
            depths=SMALL_DEPTHS,
            kernels=SMALL_KERNELS,
        ).eval()

        with torch.no_grad():
            outputs = model(torch.randn(2, 3, 32, 32))

        self.assertEqual(set(outputs), {"seg", "coarse", "uncertainty"})
        self.assertEqual(outputs["seg"].shape, (2, 1, 32, 32))
        self.assertEqual(outputs["coarse"].shape, (2, 1, 4, 4))
        self.assertEqual(model.hspm_backbone_mode, "dual_path")
        self.assertEqual(model.hspm_fusion_mode, "global")
        self.assertEqual(model.prototype_mixer.mixer_mode, "legacy")
        self.assertAlmostEqual(model.effective_fusion_gate().item(), 0.05, places=5)

    def test_hspm_best0619_output_contract_and_fixed_structure(self):
        model = cmunext_hspm_best0619(
            dims=SMALL_DIMS,
            depths=SMALL_DEPTHS,
            kernels=SMALL_KERNELS,
        ).eval()

        with torch.no_grad():
            outputs = model(torch.randn(2, 3, 32, 32))

        self.assertEqual(set(outputs), {"seg", "coarse", "uncertainty"})
        self.assertEqual(outputs["seg"].shape, (2, 1, 32, 32))
        self.assertEqual(outputs["coarse"].shape, (2, 1, 4, 4))
        self.assertEqual(model.hspm_backbone_mode, "dual_path")
        self.assertEqual(model.hspm_fusion_mode, "global")
        self.assertEqual(model.prototype_mixer.mixer_mode, "bounded")
        self.assertEqual(model.prototype_mixer.gamma_max, 0.35)
        self.assertAlmostEqual(
            model.prototype_mixer.effective_gamma().item(),
            0.1,
            places=5,
        )
        self.assertAlmostEqual(model.effective_fusion_gate().item(), 0.05, places=5)

    def test_hspm_presets_are_applied_by_model_name(self):
        hspm_args = training_main.apply_best0616_presets(
            Namespace(model=training_main.HSPM_BEST0616_MODEL)
        )
        self.assertEqual(hspm_args.hspm_backbone_mode, "dual_path")
        self.assertEqual(hspm_args.hspm_fusion_mode, "global")
        self.assertEqual(hspm_args.hspm_mixer_mode, "legacy")
        self.assertEqual(hspm_args.hspm_coarse_loss_weight, 0.1)
        self.assertEqual(hspm_args.hspm_coarse_loss_final_weight, 0.02)
        self.assertEqual(hspm_args.hspm_coarse_loss_decay_epochs, 150)

        hspm_0619_args = training_main.apply_best0616_presets(
            Namespace(model=training_main.HSPM_BEST0619_MODEL)
        )
        self.assertEqual(hspm_0619_args.hspm_backbone_mode, "dual_path")
        self.assertEqual(hspm_0619_args.hspm_fusion_mode, "global")
        self.assertEqual(hspm_0619_args.hspm_mixer_mode, "bounded")
        self.assertEqual(hspm_0619_args.hspm_gamma_init, 0.1)
        self.assertEqual(hspm_0619_args.hspm_gamma_max, 0.35)
        self.assertEqual(hspm_0619_args.hspm_coarse_loss_weight, 0.1)
        self.assertEqual(hspm_0619_args.hspm_coarse_loss_final_weight, 0.02)
        self.assertEqual(hspm_0619_args.hspm_coarse_loss_decay_epochs, 150)

        inference_args = inference_main.apply_best0616_presets(
            Namespace(model=inference_main.HSPM_BEST0619_MODEL)
        )
        self.assertEqual(inference_args.hspm_mixer_mode, "bounded")
        self.assertEqual(inference_args.hspm_gamma_max, 0.35)

    def test_hspm_best0616_state_dict_matches_original_best_structure(self):
        original = cmunext_hspm(
            dims=SMALL_DIMS,
            depths=SMALL_DEPTHS,
            kernels=SMALL_KERNELS,
            hspm_backbone_mode="dual_path",
            hspm_fusion_mode="global",
            hspm_mixer_mode="legacy",
        )
        best = cmunext_hspm_best0616(
            dims=SMALL_DIMS,
            depths=SMALL_DEPTHS,
            kernels=SMALL_KERNELS,
        )

        best.load_state_dict(original.state_dict(), strict=True)

    def test_hspm_best0619_state_dict_matches_original_best_structure(self):
        original = cmunext_hspm(
            dims=SMALL_DIMS,
            depths=SMALL_DEPTHS,
            kernels=SMALL_KERNELS,
            hspm_backbone_mode="dual_path",
            hspm_fusion_mode="global",
            hspm_mixer_mode="bounded",
            hspm_gamma_max=0.35,
        )
        best = cmunext_hspm_best0619(
            dims=SMALL_DIMS,
            depths=SMALL_DEPTHS,
            kernels=SMALL_KERNELS,
        )

        best.load_state_dict(original.state_dict(), strict=True)


if __name__ == "__main__":
    unittest.main()
