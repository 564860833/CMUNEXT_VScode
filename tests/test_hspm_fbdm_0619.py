import copy
import sys
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

import torch

with mock.patch.object(sys, "argv", ["main.py"]):
    import main as training_main

import infer as inference_main
from src.network.conv_based.CMUNeXt_HSPM import cmunext_hspm
from src.network.conv_based.CMUNeXt_FBDM_Best0616 import (
    FBDM as Best0616FBDM,
    LightweightEdgeHead as Best0616LightweightEdgeHead,
)
from src.network.conv_based.CMUNeXt_HSPM_FBDM_0619 import (
    cmunext_hspm_fbdm_0619,
)
from src.utils.losses import HSPMFBDMLoss


SMALL_DIMS = (4, 8, 16, 20, 24)
SMALL_DEPTHS = (1, 1, 1, 1, 1)
SMALL_KERNELS = (3, 3, 3, 3, 3)


def _small_hspm():
    return cmunext_hspm(
        dims=SMALL_DIMS,
        depths=SMALL_DEPTHS,
        kernels=SMALL_KERNELS,
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
    )


def _small_combined():
    return cmunext_hspm_fbdm_0619(
        dims=SMALL_DIMS,
        depths=SMALL_DEPTHS,
        kernels=SMALL_KERNELS,
    )


class HSPMFBDM0619Tests(unittest.TestCase):
    def test_model_file_is_self_contained(self):
        source = Path(
            "src/network/conv_based/CMUNeXt_HSPM_FBDM_0619.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("from src.network", source)
        self.assertNotIn("import src.network", source)

    def test_output_contract_and_fixed_configuration(self):
        model = _small_combined().eval()
        with torch.no_grad():
            outputs = model(torch.randn(2, 3, 32, 32))

        self.assertEqual(
            set(outputs),
            {"seg", "coarse", "uncertainty", "edge", "edge_x2"},
        )
        self.assertEqual(outputs["seg"].shape, (2, 1, 32, 32))
        self.assertEqual(outputs["coarse"].shape, (2, 1, 4, 4))
        self.assertEqual(outputs["edge"].shape, (2, 1, 32, 32))
        self.assertEqual(outputs["edge_x2"].shape, (2, 1, 16, 16))
        self.assertEqual(model.hspm_backbone_mode, "dual_path")
        self.assertEqual(model.prototype_mixer.mixer_mode, "bounded")
        self.assertAlmostEqual(model.prototype_mixer.gamma_max, 0.35)
        self.assertTrue(model.fbdm1.edge_aux_only)
        self.assertFalse(model.fbdm1.use_hspm_prior)
        self.assertAlmostEqual(model.fbdm1.effective_gamma().item(), 0.01, places=5)

    def test_auxiliary_heads_do_not_change_hspm_outputs(self):
        base = _small_hspm().eval()
        combined = _small_combined().eval()
        incompatible = combined.load_state_dict(base.state_dict(), strict=False)
        self.assertEqual(incompatible.unexpected_keys, [])
        self.assertTrue(
            all(
                key.startswith(("fbdm1.", "x2_edge_head."))
                for key in incompatible.missing_keys
            )
        )

        image = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            base_outputs = base(image)
            combined_outputs = combined(image)
        for name in ("seg", "coarse", "uncertainty"):
            self.assertTrue(torch.equal(base_outputs[name], combined_outputs[name]))

    def test_auxiliary_initialization_preserves_rng_state(self):
        torch.manual_seed(123)
        _small_hspm()
        base_next_random = torch.rand(8)

        torch.manual_seed(123)
        _small_combined()
        combined_next_random = torch.rand(8)

        self.assertTrue(torch.equal(base_next_random, combined_next_random))

    def test_multiscale_joint_loss_uses_normalized_edge_weights(self):
        class SequenceLoss(torch.nn.Module):
            def __init__(self, values):
                super().__init__()
                self.values = iter(values)

            def forward(self, prediction, target):
                return prediction.sum() * 0.0 + next(self.values)

        outputs = {
            "seg": torch.zeros(1, 1, 16, 16),
            "coarse": torch.zeros(1, 1, 2, 2),
            "edge": torch.zeros(1, 1, 16, 16),
            "edge_x2": torch.zeros(1, 1, 8, 8),
        }
        target = torch.zeros(1, 1, 16, 16)
        criterion = HSPMFBDMLoss(
            coarse_weight=0.1,
            edge_weight=0.03,
            x2_edge_ratio=0.3,
        )
        criterion.seg_loss = SequenceLoss((10.0, 2.0))
        criterion.edge_loss = SequenceLoss((3.0, 5.0))

        total, components = criterion(outputs, target, return_components=True)
        expected_edge_raw = 3.0 / 1.3 + 0.3 * 5.0 / 1.3
        expected_total = 10.0 + 0.1 * 2.0 + 0.03 * expected_edge_raw
        self.assertAlmostEqual(components["edge_raw"].item(), expected_edge_raw, places=6)
        self.assertAlmostEqual(total.item(), expected_total, places=6)

    def test_backward_reaches_all_required_paths(self):
        model = _small_combined().train()
        image = torch.randn(2, 3, 32, 32)
        target = torch.zeros(2, 1, 32, 32)
        target[:, :, 8:24, 10:22] = 1.0
        outputs = model(image)
        loss = HSPMFBDMLoss(
            coarse_weight=0.1,
            edge_weight=0.03,
            edge_kernel_size=3,
            edge_loss_type="balanced_bce_dice",
            edge_pos_weight=20.0,
            x2_edge_ratio=0.3,
        )(outputs, target)
        loss.backward()

        self.assertIsNotNone(model.Conv_1x1.weight.grad)
        self.assertIsNotNone(model.prototype_mixer.coarse_head.weight.grad)
        self.assertIsNotNone(model.fbdm1.edge_head.weight.grad)
        self.assertIsNotNone(model.x2_edge_head.head[-1].weight.grad)
        self.assertIsNotNone(next(model.encoder1.parameters()).grad)
        self.assertIsNotNone(next(model.encoder2.parameters()).grad)

    def test_training_and_inference_registration_and_presets(self):
        args = copy.deepcopy(training_main.args)
        args.model = training_main.HSPM_FBDM_0619_MODEL
        training_main.apply_best0616_presets(args)

        self.assertIn(args.model, training_main.HSPM_FBDM_MODELS)
        self.assertNotIn(args.model, training_main.HSPM_FBDM_CORRECTION_MODELS)
        self.assertEqual(args.hspm_mixer_mode, "bounded")
        self.assertEqual(args.hspm_gamma_max, 0.35)
        self.assertEqual(args.train_file_dir, "busi_train3.txt")
        self.assertEqual(args.val_file_dir, "busi_val3.txt")
        self.assertEqual(args.base_lr, 0.01)
        self.assertEqual(args.batch_size, 8)
        self.assertEqual(args.epoch, 300)
        self.assertEqual(args.seed, 41)
        self.assertFalse(args.use_extra_aug)
        self.assertEqual(args.sampling_mode, "uniform")
        self.assertEqual(args.val_threshold, 0.5)
        self.assertEqual(args.fbdm_stages, (0, 1))
        self.assertEqual(args.fbdm_edge_loss_type, "balanced_bce_dice")
        self.assertEqual(args.fbdm_edge_kernel_size, 3)
        self.assertEqual(args.fbdm_x2_edge_ratio, 0.3)
        self.assertEqual(args.fbdm_edge_loss_weight, 0.03)
        self.assertEqual(args.fbdm_edge_loss_final_weight, 0.003)

        with mock.patch.object(torch.nn.Module, "cuda", lambda self: self):
            built_model = training_main.get_model(args)
            criterion = training_main.get_criterion(args)
        self.assertEqual(type(built_model).__name__, "CMUNeXt_HSPM_FBDM_0619")
        self.assertEqual(criterion.edge_loss.loss_type, "balanced_bce_dice")
        self.assertEqual(criterion.x2_edge_ratio, 0.3)
        optimizer = training_main.build_optimizer(args, built_model)
        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertEqual(optimizer.defaults["lr"], 0.01)
        self.assertEqual(optimizer.defaults["momentum"], 0.9)
        self.assertEqual(optimizer.defaults["weight_decay"], 0.0001)

        inference_args = inference_main.apply_best0616_presets(
            Namespace(model=inference_main.HSPM_FBDM_0619_MODEL)
        )
        inference_args.num_classes = 1
        inferred_model = inference_main.build_model(inference_args, parser=mock.Mock())
        self.assertEqual(type(inferred_model).__name__, "CMUNeXt_HSPM_FBDM_0619")

    def test_state_dict_round_trip(self):
        source = _small_combined()
        restored = _small_combined()
        restored.load_state_dict(source.state_dict(), strict=True)

    def test_inlined_fbdm_components_remain_best0616_compatible(self):
        model = _small_combined()
        reference_fbdm = Best0616FBDM(channels=SMALL_DIMS[0])
        reference_x2_head = Best0616LightweightEdgeHead(channels=SMALL_DIMS[1])

        model.fbdm1.load_state_dict(reference_fbdm.state_dict(), strict=True)
        model.x2_edge_head.load_state_dict(reference_x2_head.state_dict(), strict=True)


if __name__ == "__main__":
    unittest.main()
