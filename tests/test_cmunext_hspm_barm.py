import argparse
import sys
import unittest
from argparse import Namespace
from unittest import mock

import torch

with mock.patch.object(sys, "argv", ["main.py"]):
    import main as training_main
import infer as inference_main

from src.network.conv_based.CMUNeXt_HSPM_BARM import (
    BoundaryAwareRefinement,
    cmunext_hspm_barm,
    cmunext_hspm_barm_hfbypass,
)
from src.utils.losses import HSPMBARMLoss


SMALL_DIMS = (4, 8, 16, 20, 24)
SMALL_DEPTHS = (1, 1, 1, 1, 1)
SMALL_KERNELS = (3, 3, 3, 3, 3)


def _has_nonzero_grad(parameter):
    return parameter.grad is not None and parameter.grad.detach().abs().sum().item() > 0.0


def _small_model():
    return cmunext_hspm_barm(
        dims=SMALL_DIMS,
        depths=SMALL_DEPTHS,
        kernels=SMALL_KERNELS,
        hspm_mixer_mode="bounded",
        hspm_gamma_max=0.35,
        hspm_backbone_mode="dual_path",
        barm_gate_init=0.05,
        barm_gate_max=0.5,
    )


def _small_hfbypass_model():
    return cmunext_hspm_barm_hfbypass(
        dims=SMALL_DIMS,
        depths=SMALL_DEPTHS,
        kernels=SMALL_KERNELS,
        hspm_mixer_mode="bounded",
        hspm_gamma_max=0.35,
        hspm_backbone_mode="dual_path",
        barm_gate_init=0.05,
        barm_gate_max=0.5,
    )


class CMUNeXtHSPMBARMTests(unittest.TestCase):
    def test_output_contract_and_barm_warmup(self):
        model = _small_model().eval()
        model.set_warmup_scale(0.5)

        with torch.no_grad():
            outputs = model(torch.randn(2, 3, 32, 32))

        self.assertEqual(
            set(outputs),
            {"seg", "base_seg", "coarse", "uncertainty", "edge", "band", "logit_correction"},
        )
        for key in ("seg", "base_seg", "edge", "band", "logit_correction"):
            self.assertEqual(outputs[key].shape, (2, 1, 32, 32))
        self.assertEqual(outputs["coarse"].shape, (2, 1, 4, 4))
        self.assertEqual(outputs["uncertainty"].shape, (2, 1, 4, 4))
        self.assertAlmostEqual(model.barm.effective_gamma().item(), 0.025, places=5)
        self.assertIsNotNone(model.get_barm_diagnostics())
        self.assertIsNotNone(model.last_fusion_diagnostics)

    def test_hfbypass_output_contract_and_hf_feature_path(self):
        model = _small_hfbypass_model().eval()
        self.assertTrue(model.barm_hf_bypass)

        with torch.no_grad():
            outputs = model(torch.randn(2, 3, 32, 32))

        self.assertEqual(
            set(outputs),
            {"seg", "base_seg", "coarse", "uncertainty", "edge", "band", "logit_correction"},
        )
        for key in ("seg", "base_seg", "edge", "band", "logit_correction"):
            self.assertEqual(outputs[key].shape, (2, 1, 32, 32))
        self.assertIsNotNone(model.get_barm_diagnostics())

        barm = BoundaryAwareRefinement(
            channels=SMALL_DIMS[0],
            gate_init=0.05,
            gate_max=0.5,
        ).eval()
        feature = torch.randn(1, SMALL_DIMS[0], 32, 32)
        hf_feature = torch.randn(1, SMALL_DIMS[0], 32, 32)
        seg_logits = torch.randn(1, 1, 32, 32)
        with torch.no_grad():
            refined, edge, band, correction = barm(
                feature,
                seg_logits,
                hf_feature=hf_feature,
            )
        self.assertEqual(refined.shape, (1, 1, 32, 32))
        self.assertEqual(edge.shape, (1, 1, 32, 32))
        self.assertEqual(band.shape, (1, 1, 32, 32))
        self.assertEqual(correction.shape, (1, 1, 32, 32))
        self.assertIsNotNone(barm.last_diagnostics)

    def test_loss_backward_reaches_hspm_and_barm_heads(self):
        torch.manual_seed(0)
        model = _small_model().train()
        inputs = torch.randn(2, 3, 32, 32)
        target = torch.zeros(2, 1, 32, 32)
        target[:, :, 8:24, 10:22] = 1.0

        outputs = model(inputs)
        criterion = HSPMBARMLoss(
            base_weight=0.3,
            coarse_weight=0.1,
            boundary_weight=0.1,
            edge_weight=0.05,
            edge_band_width=2,
            edge_pos_weight=10.0,
        )
        total, components = criterion(outputs, target, coarse_weight=0.08, return_components=True)
        self.assertEqual(
            set(components),
            {"seg", "base_weighted", "coarse_weighted", "boundary_weighted", "edge_weighted", "total"},
        )
        total.backward()

        self.assertTrue(_has_nonzero_grad(model.Conv_1x1.weight))
        self.assertTrue(_has_nonzero_grad(model.prototype_mixer.coarse_head.weight))
        self.assertTrue(_has_nonzero_grad(model.barm.delta_head.weight))
        self.assertTrue(_has_nonzero_grad(model.barm.edge_head.weight))

    def test_residual_correction_loss_targets_barm_delta_only(self):
        torch.manual_seed(0)
        model = _small_model().train()
        inputs = torch.randn(2, 3, 32, 32)
        target = torch.zeros(2, 1, 32, 32)
        target[:, :, 8:24, 10:22] = 1.0

        outputs = model(inputs)
        criterion = HSPMBARMLoss(
            base_weight=0.3,
            coarse_weight=0.1,
            boundary_weight=0.1,
            edge_weight=0.05,
            correction_weight=0.05,
            correction_band_width=3,
            correction_margin=0.05,
        )
        total, components = criterion(outputs, target, coarse_weight=0.08, return_components=True)
        self.assertIn("correction_weighted", components)
        self.assertGreater(components["correction_weighted"].item(), 0.0)
        total.backward()
        self.assertTrue(_has_nonzero_grad(model.barm.delta_head.weight))

        base_seg = torch.zeros(1, 1, 16, 16, requires_grad=True)
        logit_correction = torch.zeros(1, 1, 16, 16, requires_grad=True)
        synthetic_target = torch.zeros(1, 1, 16, 16)
        synthetic_target[:, :, 4:12, 4:12] = 1.0
        synthetic_outputs = {
            "seg": base_seg + logit_correction,
            "base_seg": base_seg,
            "coarse": torch.zeros(1, 1, 4, 4, requires_grad=True),
            "edge": torch.zeros(1, 1, 16, 16, requires_grad=True),
            "logit_correction": logit_correction,
        }
        _, synthetic_components = criterion(
            synthetic_outputs,
            synthetic_target,
            coarse_weight=0.08,
            return_components=True,
        )
        synthetic_components["correction_weighted"].backward()

        self.assertTrue(_has_nonzero_grad(logit_correction))
        self.assertTrue(base_seg.grad is None or base_seg.grad.detach().abs().sum().item() == 0.0)

    def test_training_and_inference_entrypoints_register_hspm_barm(self):
        for model_name in ("CMUNeXt_HSPM_BARM", "CMUNeXt_HSPM_BARM_HFBypass"):
            self.assertIn(model_name, training_main.HSPM_BARM_MODELS)
            self.assertIn(model_name, training_main.HSPM_MODELS)
            self.assertIn(model_name, training_main.BARM_MODELS)
            self.assertIn(model_name, inference_main.HSPM_BARM_MODELS)
            self.assertIn(model_name, inference_main.BASE_SEG_MODELS)

        args = Namespace(
            model="CMUNeXt_HSPM_BARM_HFBypass",
            num_classes=1,
            hspm_mode="full",
            hspm_mixer_mode="bounded",
            hspm_gamma_init=0.1,
            hspm_gamma_max=0.35,
            hspm_temperature=0.1,
            hspm_prototype_dropout=0.0,
            hspm_backbone_mode="dual_path",
            hspm_fusion_gate_init=0.05,
            hspm_fusion_gate_max=0.3,
            hspm_fusion_mode="global",
            hspm_small_area_threshold=0.05,
            hspm_small_area_temperature=0.02,
            barm_gate_init=0.05,
            barm_gate_max=0.5,
            barm_hf_keep_init=0.3,
        )
        model = inference_main.build_model(args, argparse.ArgumentParser()).eval()
        self.assertTrue(model.barm_hf_bypass)
        with torch.no_grad():
            outputs = model(torch.randn(1, 3, 32, 32))
        self.assertEqual(outputs["seg"].shape, (1, 1, 32, 32))
        self.assertEqual(
            inference_main.get_seg_logits(outputs, use_base_seg=True).shape,
            (1, 1, 32, 32),
        )

        loss_args = Namespace(model="CMUNeXt_HSPM_BARM_HFBypass")
        loss_output = training_main.compute_loss(
            loss_args,
            HSPMBARMLoss(),
            outputs,
            torch.zeros(1, 1, 32, 32),
            sampled_batch={},
            aux_weight=0.1,
        )
        self.assertIsInstance(loss_output, tuple)


if __name__ == "__main__":
    unittest.main()
