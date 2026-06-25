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
from src.network.conv_based.CMUNeXt_HSPM_Best0619 import cmunext_hspm_best0619
from src.network.conv_based.CMUNeXt_HSPM_FBDM import cmunext_hspm_fbdm
from src.network.conv_based.CMUNeXt_HSPM_FBDM_Best0616 import cmunext_hspm_fbdm_best0616
from src.utils.losses import FBDMLoss, HSPMFBDMLoss, mask_to_edge


SMALL_DIMS = (4, 8, 16, 20, 24)
SMALL_DEPTHS = (1, 1, 1, 1, 1)
SMALL_KERNELS = (3, 3, 3, 3, 3)


class Best0616ModelTests(unittest.TestCase):
    def test_inference_can_select_protected_base_logits(self):
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
        criterion.seg_loss = SequenceLoss((10.0,))
        criterion.edge_loss = SequenceLoss((2.0, 4.0))

        total, components = criterion(outputs, target, return_components=True)
        x1_weight = 0.03 / 1.3
        x2_weight = x1_weight * 0.3
        expected_edge = x1_weight * 2.0 + x2_weight * 4.0

        self.assertAlmostEqual(
            components["edge_raw"].item(),
            expected_edge / 0.03,
            places=6,
        )
        self.assertAlmostEqual(components["edge_weighted"].item(), expected_edge, places=6)
        self.assertAlmostEqual(total.item(), 10.0 + expected_edge, places=6)
        self.assertAlmostEqual(x1_weight + x2_weight, 0.03, places=8)

        single_stage_outputs = {
            "seg": torch.zeros(1, 1, 16, 16),
            "edge": torch.zeros(1, 1, 8, 8),
        }
        single_stage_criterion = FBDMLoss(edge_weight=0.03, x2_edge_ratio=0.3)
        single_stage_criterion.seg_loss = SequenceLoss((10.0,))
        single_stage_criterion.edge_loss = SequenceLoss((4.0,))
        _, single_stage_components = single_stage_criterion(
            single_stage_outputs,
            target,
            return_components=True,
        )
        self.assertAlmostEqual(single_stage_components["edge_raw"].item(), 4.0, places=6)
        self.assertAlmostEqual(single_stage_components["edge_weighted"].item(), 0.12, places=6)

    def test_fbdm_multistage_diagnostics_follow_loss_ratio(self):
        target = torch.zeros(1, 1, 16, 16)
        target[:, :, 4:12, 4:12] = 1.0
        outputs = {
            "seg": torch.zeros(1, 1, 16, 16),
            "edge": torch.zeros(1, 1, 16, 16),
            "edge_x2": torch.full((1, 1, 8, 8), 2.0),
        }
        ratio = 0.25
        criterion = FBDMLoss(edge_weight=0.03, x2_edge_ratio=ratio)

        criterion(outputs, target)
        diagnostics = criterion.get_edge_diagnostics()
        x1_target = criterion._edge_target(target, (16, 16))
        x2_target = criterion._edge_target(target, (8, 8))
        x1_ratio = 1.0 / (1.0 + ratio)
        x2_ratio = ratio / (1.0 + ratio)

        expected_target_ratio = (
            x1_ratio * x1_target.mean() + x2_ratio * x2_target.mean()
        )
        expected_probability = (
            x1_ratio * torch.tensor(0.5)
            + x2_ratio * torch.sigmoid(torch.tensor(2.0))
        )
        self.assertTrue(
            torch.allclose(diagnostics["edge_target_ratio"], expected_target_ratio)
        )
        self.assertTrue(
            torch.allclose(diagnostics["edge_prob_mean"], expected_probability)
        )
        self.assertAlmostEqual(
            diagnostics["edge_pred_positive_ratio"].item(),
            x2_ratio,
            places=6,
        )

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

    def test_stage1_only_fbdm_schedule_has_no_residual_gate(self):
        model = cmunext_fbdm_best0616(
            dims=SMALL_DIMS,
            depths=SMALL_DEPTHS,
            kernels=SMALL_KERNELS,
            fbdm_stages=(1,),
        )
        args = Namespace(
            model=training_main.FBDM_BEST0616_MODEL,
            fbdm_edge_loss_weight=0.03,
            fbdm_edge_loss_final_weight=0.003,
            fbdm_edge_loss_decay_epochs=150,
            fbdm_residual_warmup_epochs=40,
        )

        edge_weight, residual_scale, effective_gamma = training_main.configure_fbdm_epoch(
            args,
            model,
            epoch_num=20,
        )

        self.assertAlmostEqual(edge_weight, 0.0264)
        self.assertAlmostEqual(residual_scale, 0.5)
        self.assertEqual(effective_gamma, 0.0)

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

    def test_hspm_fbdm_best0616_output_contract_and_fixed_structure(self):
        model = cmunext_hspm_fbdm_best0616(
            dims=SMALL_DIMS,
            depths=SMALL_DEPTHS,
            kernels=SMALL_KERNELS,
        ).eval()
        model.set_fbdm_correction_schedule_scale(0.0)

        with torch.no_grad():
            outputs = model(torch.randn(2, 3, 32, 32))

        self.assertEqual(
            set(outputs),
            {
                "seg",
                "base_seg",
                "coarse",
                "uncertainty",
                "edge",
                "boundary_gate",
                "logit_correction",
            },
        )
        self.assertEqual(outputs["seg"].shape, (2, 1, 32, 32))
        self.assertTrue(torch.equal(outputs["seg"], outputs["base_seg"]))
        self.assertTrue(
            torch.equal(outputs["logit_correction"], torch.zeros_like(outputs["seg"]))
        )
        self.assertEqual(outputs["coarse"].shape, (2, 1, 4, 4))
        self.assertEqual(outputs["edge"].shape, (2, 1, 32, 32))
        self.assertEqual(model.hspm_backbone_mode, "dual_path")
        self.assertEqual(model.hspm_fusion_mode, "global")
        self.assertEqual(model.prototype_mixer.mixer_mode, "bounded")
        self.assertEqual(model.prototype_mixer.gamma_max, 0.35)
        self.assertAlmostEqual(model.effective_fusion_gate().item(), 0.05, places=5)
        self.assertTrue(model.fbdm1.edge_aux_only)
        self.assertTrue(model.fbdm1.use_hspm_prior)
        self.assertTrue(model.fbdm1.detach_hspm_prior)
        self.assertAlmostEqual(model.fbdm1.effective_gamma().item(), 0.01, places=5)
        self.assertEqual(model.fbdm1.gate_max, 0.06)
        self.assertEqual(model.fbdm_correction.correction_scale_max, 0.20)
        self.assertEqual(model.fbdm_correction.boundary_gate_floor, 0.20)
        self.assertEqual(
            model.fbdm1.last_boundary_feature.shape,
            (2, SMALL_DIMS[0], 32, 32),
        )
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
            protected_refinement=True,
            refine_weight=1.0,
            preserve_weight=1.0,
        )(outputs, target, return_components=True)

        self.assertTrue(torch.isfinite(total))
        self.assertIn("edge_weighted", components)
        self.assertIn("refine_weighted", components)
        self.assertIn("preserve_weighted", components)
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
        self.assertEqual(hspm_fbdm_args.hspm_mixer_mode, "bounded")
        self.assertEqual(hspm_fbdm_args.hspm_gamma_max, 0.35)
        self.assertEqual(hspm_fbdm_args.hspm_coarse_loss_weight, 0.1)
        self.assertEqual(hspm_fbdm_args.hspm_coarse_loss_final_weight, 0.02)
        self.assertEqual(hspm_fbdm_args.hspm_coarse_loss_decay_epochs, 150)
        self.assertEqual(hspm_fbdm_args.fbdm_edge_loss_type, "balanced_bce_dice")
        self.assertEqual(hspm_fbdm_args.fbdm_edge_kernel_size, 5)
        self.assertEqual(hspm_fbdm_args.fbdm_correction_scale_init, 0.05)
        self.assertEqual(hspm_fbdm_args.fbdm_correction_scale_max, 0.20)
        self.assertEqual(hspm_fbdm_args.fbdm_boundary_gate_floor, 0.20)
        self.assertEqual(hspm_fbdm_args.fbdm_correction_start_epoch, 40)
        self.assertEqual(hspm_fbdm_args.fbdm_correction_warmup_epochs, 40)
        self.assertEqual(hspm_fbdm_args.fbdm_refine_loss_weight, 1.0)
        self.assertEqual(hspm_fbdm_args.fbdm_preserve_loss_weight, 1.0)
        self.assertEqual(hspm_fbdm_args.fbdm_boundary_band_loss_weight, 0.10)
        self.assertEqual(hspm_fbdm_args.fbdm_boundary_band_loss_final_weight, 0.02)
        self.assertEqual(hspm_fbdm_args.fbdm_boundary_band_loss_decay_epochs, 150)
        self.assertEqual(hspm_fbdm_args.fbdm_lr_multiplier, 2.0)
        self.assertEqual(hspm_fbdm_args.fbdm_correction_lr_multiplier, 5.0)

        inference_args = inference_main.apply_best0616_presets(
            Namespace(model=training_main.HSPM_FBDM_BEST0616_MODEL)
        )
        self.assertEqual(inference_args.fbdm_edge_kernel_size, 5)
        self.assertEqual(inference_args.fbdm_correction_scale_init, 0.05)
        self.assertEqual(inference_args.fbdm_correction_scale_max, 0.20)
        self.assertEqual(inference_args.fbdm_boundary_gate_floor, 0.20)

    def test_edge_loss_cli_defaults_and_best0616_presets_are_compatible(self):
        self.assertEqual(training_main.args.fbdm_edge_loss_type, "legacy")
        self.assertEqual(training_main.args.fbdm_edge_pos_weight, 20.0)
        self.assertEqual(training_main.args.fbdm_edge_focal_alpha, 0.95)
        self.assertEqual(training_main.args.fbdm_edge_focal_gamma, 2.0)

        configured = training_main.apply_best0616_presets(
            Namespace(
                model=training_main.FBDM_BEST0616_MODEL,
                fbdm_edge_loss_type="focal_dice",
                fbdm_edge_pos_weight=12.0,
                fbdm_edge_focal_alpha=0.9,
                fbdm_edge_focal_gamma=1.5,
            )
        )
        self.assertEqual(configured.fbdm_edge_loss_type, "focal_dice")
        self.assertEqual(configured.fbdm_edge_pos_weight, 12.0)
        self.assertEqual(configured.fbdm_edge_focal_alpha, 0.9)
        self.assertEqual(configured.fbdm_edge_focal_gamma, 1.5)

    def test_training_criterion_receives_edge_loss_options(self):
        args = Namespace(
            model=training_main.FBDM_BEST0616_MODEL,
            fbdm_edge_loss_weight=0.03,
            fbdm_edge_kernel_size=3,
            fbdm_x2_edge_ratio=0.3,
            fbdm_edge_loss_type="balanced_bce_dice",
            fbdm_edge_pos_weight=15.0,
            fbdm_edge_focal_alpha=0.9,
            fbdm_edge_focal_gamma=1.5,
        )
        with mock.patch.object(torch.nn.Module, "cuda", lambda self: self):
            criterion = training_main.get_criterion(args)

        self.assertEqual(criterion.edge_loss.loss_type, "balanced_bce_dice")
        self.assertEqual(criterion.edge_loss.pos_weight.item(), 15.0)
        self.assertEqual(criterion.edge_loss.focal_alpha, 0.9)
        self.assertEqual(criterion.edge_loss.focal_gamma, 1.5)

    def test_best0616_files_do_not_import_other_model_classes(self):
        for path in (
            Path("src/network/conv_based/CMUNeXt_FBDM_Best0616.py"),
            Path("src/network/conv_based/CMUNeXt_HSPM_Best0616.py"),
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
        original = cmunext_hspm(
            dims=SMALL_DIMS,
            depths=SMALL_DEPTHS,
            kernels=SMALL_KERNELS,
            hspm_backbone_mode="dual_path",
            hspm_fusion_mode="global",
            hspm_mixer_mode="bounded",
            hspm_gamma_max=0.35,
        )
        best = cmunext_hspm_fbdm_best0616(
            dims=SMALL_DIMS,
            depths=SMALL_DEPTHS,
            kernels=SMALL_KERNELS,
        )

        incompatible = best.load_state_dict(original.state_dict(), strict=False)
        self.assertEqual(incompatible.unexpected_keys, [])
        self.assertTrue(incompatible.missing_keys)
        self.assertTrue(
            all(
                key.startswith(("fbdm1.", "fbdm_correction."))
                for key in incompatible.missing_keys
            )
        )


if __name__ == "__main__":
    unittest.main()
