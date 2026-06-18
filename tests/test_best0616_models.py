import sys
import unittest
import argparse
from argparse import Namespace
from pathlib import Path
from unittest import mock

import torch

with mock.patch.object(sys, "argv", ["main.py"]):
    import main as training_main
import infer as inference_main

from src.network.conv_based.CMUNeXt_FBDM import cmunext_fbdm
from src.network.conv_based.CMUNeXt_FBDM_Best0616 import cmunext_fbdm_best0616
from src.network.conv_based.CMUNeXt_HSPM import cmunext_hspm
from src.network.conv_based.CMUNeXt_HSPM_Best0616 import cmunext_hspm_best0616
from src.network.conv_based.CMUNeXt_HSPM_FBDM import cmunext_hspm_fbdm
from src.network.conv_based.CMUNeXt_HSPM_FBDM_Best0616 import cmunext_hspm_fbdm_best0616
from src.utils.losses import FBDMLoss, HSPMFBDMLoss, mask_to_edge


SMALL_DIMS = (4, 8, 16, 20, 24)
SMALL_DEPTHS = (1, 1, 1, 1, 1)
SMALL_KERNELS = (3, 3, 3, 3, 3)


class Best0616ModelTests(unittest.TestCase):
    def test_fbdm_stage_parser_accepts_supported_combinations(self):
        for parser_fn in (training_main.parse_fbdm_stages, inference_main.parse_fbdm_stages):
            self.assertEqual(parser_fn("0"), (0,))
            self.assertEqual(parser_fn("1"), (1,))
            self.assertEqual(parser_fn("0,1"), (0, 1))
            self.assertEqual(parser_fn("1,0"), (0, 1))
            self.assertEqual(parser_fn((1, 0)), (0, 1))

            for invalid in ("", "2", "0,2", "0,,1"):
                with self.assertRaises(argparse.ArgumentTypeError):
                    parser_fn(invalid)

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
        self.assertIsNone(model.x2_edge_head)

    def test_fbdm_best0616_stage_output_contracts(self):
        stage1_model = cmunext_fbdm_best0616(
            dims=SMALL_DIMS,
            depths=SMALL_DEPTHS,
            kernels=SMALL_KERNELS,
            fbdm_stages=(1,),
        ).eval()
        dual_model = cmunext_fbdm_best0616(
            dims=SMALL_DIMS,
            depths=SMALL_DEPTHS,
            kernels=SMALL_KERNELS,
            fbdm_stages=(0, 1),
        ).eval()

        with torch.no_grad():
            stage1_outputs = stage1_model(torch.randn(2, 3, 32, 32))
            dual_outputs = dual_model(torch.randn(2, 3, 32, 32))

        self.assertEqual(set(stage1_outputs), {"seg", "edge"})
        self.assertEqual(stage1_outputs["edge"].shape, (2, 1, 16, 16))
        self.assertIsNone(stage1_model.fbdm1)
        self.assertIsNotNone(stage1_model.x2_edge_head)

        self.assertEqual(set(dual_outputs), {"seg", "edge", "edge_x2"})
        self.assertEqual(dual_outputs["edge"].shape, (2, 1, 32, 32))
        self.assertEqual(dual_outputs["edge_x2"].shape, (2, 1, 16, 16))
        self.assertIsNotNone(dual_model.fbdm1)
        self.assertIsNotNone(dual_model.x2_edge_head)

    def test_fbdm_best0616_rejects_unsupported_stages(self):
        for stages in ((), (2,), (0, 2)):
            with self.assertRaises(ValueError):
                cmunext_fbdm_best0616(
                    dims=SMALL_DIMS,
                    depths=SMALL_DEPTHS,
                    kernels=SMALL_KERNELS,
                    fbdm_stages=stages,
                )

    def test_fbdm_multistage_loss_preserves_total_edge_weight(self):
        class SequenceLoss(torch.nn.Module):
            def __init__(self, values):
                super().__init__()
                self.values = iter(values)

            def forward(self, prediction, target):
                return prediction.sum() * 0.0 + next(self.values)

        target = torch.zeros(1, 1, 16, 16)
        outputs = {
            "seg": torch.zeros(1, 1, 16, 16),
            "edge": torch.zeros(1, 1, 16, 16),
            "edge_x2": torch.zeros(1, 1, 8, 8),
        }
        criterion = FBDMLoss(edge_weight=0.03, x2_edge_ratio=0.3)
        criterion.seg_loss = SequenceLoss((10.0, 2.0, 4.0))

        total, components = criterion(outputs, target, return_components=True)
        x1_weight = 0.03 / 1.3
        x2_weight = x1_weight * 0.3
        expected_edge = x1_weight * 2.0 + x2_weight * 4.0

        self.assertAlmostEqual(components["edge_weighted"].item(), expected_edge, places=6)
        self.assertAlmostEqual(total.item(), 10.0 + expected_edge, places=6)
        self.assertAlmostEqual(x1_weight + x2_weight, 0.03, places=8)

        single_stage_outputs = {
            "seg": torch.zeros(1, 1, 16, 16),
            "edge": torch.zeros(1, 1, 8, 8),
        }
        single_stage_criterion = FBDMLoss(edge_weight=0.03, x2_edge_ratio=0.3)
        single_stage_criterion.seg_loss = SequenceLoss((10.0, 4.0))
        _, single_stage_components = single_stage_criterion(
            single_stage_outputs,
            target,
            return_components=True,
        )
        self.assertAlmostEqual(single_stage_components["edge_weighted"].item(), 0.12, places=6)

    def test_fbdm_loss_resizes_mask_before_extracting_edge(self):
        target = torch.zeros(1, 1, 16, 16)
        target[:, :, 1, 1] = 1.0
        criterion = FBDMLoss(edge_kernel_size=3)

        actual = criterion._edge_target(target, (8, 8))
        resized_mask = torch.nn.functional.interpolate(target, size=(8, 8), mode="nearest")
        expected = mask_to_edge(resized_mask, kernel_size=3)
        resize_after_edge = torch.nn.functional.interpolate(
            mask_to_edge(target, kernel_size=3),
            size=(8, 8),
            mode="nearest",
        )

        self.assertTrue(torch.equal(actual, expected))
        self.assertFalse(torch.equal(actual, resize_after_edge))

    def test_fbdm_multistage_loss_backward_reaches_both_heads(self):
        model = cmunext_fbdm_best0616(
            dims=SMALL_DIMS,
            depths=SMALL_DEPTHS,
            kernels=SMALL_KERNELS,
            fbdm_stages=(0, 1),
        ).eval()
        target = torch.zeros(2, 1, 32, 32)
        target[:, :, 8:24, 10:22] = 1.0

        outputs = model(torch.randn(2, 3, 32, 32))
        loss = FBDMLoss(edge_weight=0.03, x2_edge_ratio=0.3)(outputs, target)
        loss.backward()

        self.assertIsNotNone(model.fbdm1.edge_head.weight.grad)
        self.assertIsNotNone(model.x2_edge_head.head[-1].weight.grad)
        self.assertIsNotNone(next(model.encoder2.parameters()).grad)

    def test_inference_builder_forwards_fbdm_stages(self):
        model = inference_main.build_model(
            Namespace(
                model=inference_main.FBDM_BEST0616_MODEL,
                num_classes=1,
                fbdm_stages=(1,),
            ),
            parser=mock.Mock(),
        )

        self.assertEqual(model.fbdm_stages, (1,))
        self.assertIsNone(model.fbdm1)
        self.assertIsNotNone(model.x2_edge_head)

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
