import sys
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

import torch

with mock.patch.object(sys, "argv", ["main.py"]):
    import main as training_main

from src.network.conv_based.CMUNeXt_FBDM import cmunext_fbdm
from src.network.conv_based.CMUNeXt_FBDM_Best0616 import cmunext_fbdm_best0616
from src.network.conv_based.CMUNeXt_HSPM import cmunext_hspm
from src.network.conv_based.CMUNeXt_HSPM_Best0616 import cmunext_hspm_best0616
from src.network.conv_based.CMUNeXt_HSPM_FBDM import cmunext_hspm_fbdm
from src.network.conv_based.CMUNeXt_HSPM_FBDM_Best0616 import cmunext_hspm_fbdm_best0616
from src.utils.losses import HSPMFBDMLoss


SMALL_DIMS = (4, 8, 16, 20, 24)
SMALL_DEPTHS = (1, 1, 1, 1, 1)
SMALL_KERNELS = (3, 3, 3, 3, 3)


class Best0616ModelTests(unittest.TestCase):
    def test_fbdm_best0616_output_contract_and_fixed_gate(self):
        model = cmunext_fbdm_best0616(
            dims=SMALL_DIMS,
            depths=SMALL_DEPTHS,
            kernels=SMALL_KERNELS,
        ).eval()

        with torch.no_grad():
            outputs = model(torch.randn(2, 3, 32, 32))

        self.assertEqual(set(outputs), {"seg", "edge"})
        self.assertEqual(outputs["seg"].shape, (2, 1, 32, 32))
        self.assertEqual(outputs["edge"].shape, (2, 1, 32, 32))
        self.assertTrue(model.fbdm1.edge_aux_only)
        self.assertFalse(model.fbdm1.use_hspm_prior)
        self.assertAlmostEqual(model.fbdm1.effective_gamma().item(), 0.01, places=5)
        self.assertEqual(model.fbdm1.gate_max, 0.06)

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

    def test_hspm_fbdm_best0616_output_contract_and_fixed_structure(self):
        model = cmunext_hspm_fbdm_best0616(
            dims=SMALL_DIMS,
            depths=SMALL_DEPTHS,
            kernels=SMALL_KERNELS,
        ).eval()

        with torch.no_grad():
            outputs = model(torch.randn(2, 3, 32, 32))

        self.assertEqual(set(outputs), {"seg", "coarse", "uncertainty", "edge"})
        self.assertEqual(outputs["seg"].shape, (2, 1, 32, 32))
        self.assertEqual(outputs["coarse"].shape, (2, 1, 4, 4))
        self.assertEqual(outputs["edge"].shape, (2, 1, 32, 32))
        self.assertEqual(model.hspm_backbone_mode, "dual_path")
        self.assertEqual(model.hspm_fusion_mode, "global")
        self.assertEqual(model.prototype_mixer.mixer_mode, "legacy")
        self.assertAlmostEqual(model.effective_fusion_gate().item(), 0.05, places=5)
        self.assertTrue(model.fbdm1.edge_aux_only)
        self.assertFalse(model.fbdm1.use_hspm_prior)
        self.assertAlmostEqual(model.fbdm1.effective_gamma().item(), 0.01, places=5)
        self.assertEqual(model.fbdm1.gate_max, 0.06)
        self.assertIsNotNone(model.last_fusion_diagnostics)

    def test_hspm_fbdm_best0616_loss_backward(self):
        model = cmunext_hspm_fbdm_best0616(
            dims=SMALL_DIMS,
            depths=SMALL_DEPTHS,
            kernels=SMALL_KERNELS,
        ).eval()
        target = torch.zeros(2, 1, 32, 32)
        target[:, :, 8:24, 10:22] = 1.0

        outputs = model(torch.randn(2, 3, 32, 32))
        total, components = HSPMFBDMLoss(
            coarse_weight=0.1,
            edge_weight=0.03,
        )(outputs, target, return_components=True)

        self.assertTrue(torch.isfinite(total))
        self.assertIn("edge_weighted", components)
        total.backward()
        self.assertIsNotNone(model.Conv_1x1.weight.grad)
        self.assertIsNotNone(model.fbdm1.edge_head.weight.grad)

    def test_best0616_presets_are_applied_by_model_name(self):
        fbdm_args = training_main.apply_best0616_presets(
            Namespace(model=training_main.FBDM_BEST0616_MODEL)
        )
        self.assertTrue(fbdm_args.fbdm_edge_aux_only)
        self.assertEqual(fbdm_args.fbdm_gate_init, 0.01)
        self.assertEqual(fbdm_args.fbdm_gate_max, 0.06)
        self.assertEqual(fbdm_args.fbdm_residual_warmup_epochs, 40)
        self.assertEqual(fbdm_args.fbdm_edge_loss_weight, 0.03)
        self.assertEqual(fbdm_args.fbdm_edge_loss_final_weight, 0.003)
        self.assertEqual(fbdm_args.fbdm_edge_loss_decay_epochs, 150)

        hspm_args = training_main.apply_best0616_presets(
            Namespace(model=training_main.HSPM_BEST0616_MODEL)
        )
        self.assertEqual(hspm_args.hspm_backbone_mode, "dual_path")
        self.assertEqual(hspm_args.hspm_fusion_mode, "global")
        self.assertEqual(hspm_args.hspm_mixer_mode, "legacy")
        self.assertEqual(hspm_args.hspm_coarse_loss_weight, 0.1)
        self.assertEqual(hspm_args.hspm_coarse_loss_final_weight, 0.02)
        self.assertEqual(hspm_args.hspm_coarse_loss_decay_epochs, 150)

        hspm_fbdm_args = training_main.apply_best0616_presets(
            Namespace(model=training_main.HSPM_FBDM_BEST0616_MODEL)
        )
        self.assertTrue(hspm_fbdm_args.fbdm_edge_aux_only)
        self.assertEqual(hspm_fbdm_args.fbdm_gate_init, 0.01)
        self.assertEqual(hspm_fbdm_args.fbdm_gate_max, 0.06)
        self.assertEqual(hspm_fbdm_args.fbdm_residual_warmup_epochs, 40)
        self.assertEqual(hspm_fbdm_args.fbdm_edge_loss_weight, 0.03)
        self.assertEqual(hspm_fbdm_args.fbdm_edge_loss_final_weight, 0.003)
        self.assertEqual(hspm_fbdm_args.fbdm_edge_loss_decay_epochs, 150)
        self.assertEqual(hspm_fbdm_args.hspm_backbone_mode, "dual_path")
        self.assertEqual(hspm_fbdm_args.hspm_fusion_mode, "global")
        self.assertEqual(hspm_fbdm_args.hspm_mixer_mode, "legacy")
        self.assertEqual(hspm_fbdm_args.hspm_coarse_loss_weight, 0.1)
        self.assertEqual(hspm_fbdm_args.hspm_coarse_loss_final_weight, 0.02)
        self.assertEqual(hspm_fbdm_args.hspm_coarse_loss_decay_epochs, 150)

    def test_best0616_files_do_not_import_other_model_classes(self):
        for path in (
            Path("src/network/conv_based/CMUNeXt_FBDM_Best0616.py"),
            Path("src/network/conv_based/CMUNeXt_HSPM_Best0616.py"),
            Path("src/network/conv_based/CMUNeXt_HSPM_FBDM_Best0616.py"),
        ):
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("from src.network", source)
            self.assertNotIn("import src.network", source)

    def test_fbdm_best0616_state_dict_matches_original_best_structure(self):
        original = cmunext_fbdm(
            dims=SMALL_DIMS,
            depths=SMALL_DEPTHS,
            kernels=SMALL_KERNELS,
            fbdm_gate_init=0.01,
            fbdm_gate_max=0.06,
            fbdm_edge_aux_only=True,
        )
        best = cmunext_fbdm_best0616(
            dims=SMALL_DIMS,
            depths=SMALL_DEPTHS,
            kernels=SMALL_KERNELS,
        )

        best.load_state_dict(original.state_dict(), strict=True)

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

    def test_hspm_fbdm_best0616_state_dict_matches_original_best_structure(self):
        original = cmunext_hspm_fbdm(
            dims=SMALL_DIMS,
            depths=SMALL_DEPTHS,
            kernels=SMALL_KERNELS,
            hspm_backbone_mode="dual_path",
            hspm_fusion_mode="global",
            hspm_mixer_mode="legacy",
            fbdm_use_hspm_prior=False,
            fbdm_gate_init=0.01,
            fbdm_gate_max=0.06,
            fbdm_edge_aux_only=True,
        )
        best = cmunext_hspm_fbdm_best0616(
            dims=SMALL_DIMS,
            depths=SMALL_DEPTHS,
            kernels=SMALL_KERNELS,
        )

        best.load_state_dict(original.state_dict(), strict=True)


if __name__ == "__main__":
    unittest.main()
