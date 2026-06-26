import argparse
import sys
import unittest
from argparse import Namespace
from unittest import mock

import torch

with mock.patch.object(sys, "argv", ["main.py"]):
    import main as training_main
import infer as inference_main

from src.network.conv_based.CMUNeXt_BARM import BoundaryAwareRefinement, cmunext_barm
from src.utils.losses import CMUNeXtBARMLoss


SMALL_DIMS = (4, 8, 16, 20, 24)
SMALL_DEPTHS = (1, 1, 1, 1, 1)
SMALL_KERNELS = (3, 3, 3, 3, 3)


def _has_nonzero_grad(parameter):
    return parameter.grad is not None and parameter.grad.detach().abs().sum().item() > 0.0


class CMUNeXtBARMTests(unittest.TestCase):
    def test_output_contract_and_warmup_scale(self):
        model = cmunext_barm(
            dims=SMALL_DIMS,
            depths=SMALL_DEPTHS,
            kernels=SMALL_KERNELS,
            barm_gate_init=0.02,
            barm_gate_max=0.2,
        ).eval()
        model.set_warmup_scale(0.5)

        with torch.no_grad():
            outputs = model(torch.randn(2, 3, 32, 32))

        self.assertEqual(
            set(outputs),
            {"seg", "seg_coarse", "edge", "band", "logit_correction"},
        )
        for key in ("seg", "seg_coarse", "edge", "band", "logit_correction"):
            self.assertEqual(outputs[key].shape, (2, 1, 32, 32))

        self.assertAlmostEqual(model.barm.residual_scale.item(), 0.5, places=6)
        self.assertAlmostEqual(model.barm.effective_gamma().item(), 0.01, places=5)
        diagnostics = model.get_barm_diagnostics()
        self.assertIsNotNone(diagnostics)
        self.assertIn("prediction_flip_ratio", diagnostics)

    def test_barm_loss_backward_reaches_refinement_heads(self):
        torch.manual_seed(0)
        model = cmunext_barm(
            dims=SMALL_DIMS,
            depths=SMALL_DEPTHS,
            kernels=SMALL_KERNELS,
        ).train()
        inputs = torch.randn(2, 3, 32, 32)
        target = torch.zeros(2, 1, 32, 32)
        target[:, :, 8:24, 10:22] = 1.0

        outputs = model(inputs)
        criterion = CMUNeXtBARMLoss(
            coarse_weight=0.3,
            boundary_weight=0.2,
            edge_weight=0.1,
            edge_band_width=2,
            edge_pos_weight=10.0,
        )
        total, components = criterion(outputs, target, return_components=True)
        self.assertEqual(
            set(components),
            {"seg", "coarse_weighted", "boundary_weighted", "edge_weighted", "total"},
        )
        total.backward()

        self.assertTrue(_has_nonzero_grad(model.Conv_1x1.weight))
        self.assertTrue(_has_nonzero_grad(model.barm.delta_head.weight))
        self.assertTrue(_has_nonzero_grad(model.barm.edge_head.weight))

    def test_hf_energy_backward_is_finite_for_flat_feature(self):
        barm = BoundaryAwareRefinement(
            channels=SMALL_DIMS[0],
            gate_init=0.02,
            gate_max=0.2,
        ).train()
        feature = torch.zeros(2, SMALL_DIMS[0], 32, 32)
        seg_logits = torch.randn(2, 1, 32, 32)

        refined, edge, _, _ = barm(feature, seg_logits)
        loss = refined.square().mean() + edge.square().mean()
        loss.backward()

        self.assertIsNotNone(barm.hf_atten.grad)
        self.assertTrue(torch.isfinite(barm.hf_atten.grad).all())

    def test_training_and_inference_entrypoints_register_barm(self):
        self.assertIn("CMUNeXt_BARM", training_main.BARM_MODELS)
        self.assertIn("CMUNeXt_BARM", inference_main.BARM_MODELS)

        args = Namespace(
            model="CMUNeXt_BARM",
            num_classes=1,
            barm_gate_init=0.02,
            barm_gate_max=0.2,
            barm_hf_keep_init=0.3,
        )
        model = inference_main.build_model(args, argparse.ArgumentParser())
        self.assertEqual(model(torch.randn(1, 3, 32, 32))["seg"].shape, (1, 1, 32, 32))

        loss_output = training_main.compute_loss(
            Namespace(model="CMUNeXt_BARM"),
            CMUNeXtBARMLoss(),
            model(torch.randn(1, 3, 32, 32)),
            torch.zeros(1, 1, 32, 32),
            sampled_batch={},
        )
        self.assertIsInstance(loss_output, tuple)


if __name__ == "__main__":
    unittest.main()
