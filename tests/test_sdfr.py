import os
import sys
import tempfile
import unittest
from argparse import Namespace
from unittest import mock

import cv2
import numpy as np
import torch
from albumentations import Compose, Resize

with mock.patch.object(sys, "argv", ["main.py"]):
    import main as training_main

import infer as inference_main
from src.dataloader.dataset import MedicalDataSets
from src.network.conv_based.CMUNeXt_HSPM_SDFR import (
    CMUNeXt_HSPM_SDFR,
    SignedDistanceBoundaryRefinement,
    cmunext_hspm_sdfr,
)
from src.utils.losses import SDFRLoss
from src.utils.sdf import compute_normalized_sdf


class SDFRTests(unittest.TestCase):
    def _small_model(self):
        return cmunext_hspm_sdfr(
            dims=(4, 8, 16, 20, 24),
            depths=(1, 1, 1, 1, 1),
            kernels=(3, 3, 3, 3, 3),
            hspm_mixer_mode="stable",
            hspm_gamma_init=0.05,
            hspm_gamma_max=0.3,
            hspm_temperature=0.5,
        )

    def test_sdf_sign_range_and_special_masks(self):
        mask = np.zeros((32, 32), dtype=np.float32)
        mask[8:24, 8:24] = 1.0
        sdf = compute_normalized_sdf(mask, truncation_ratio=0.08)
        self.assertEqual(sdf.shape, mask.shape)
        self.assertGreater(sdf[16, 16], 0.0)
        self.assertLess(sdf[0, 0], 0.0)
        self.assertGreaterEqual(sdf.min(), -1.0)
        self.assertLessEqual(sdf.max(), 1.0)
        self.assertGreater(abs(sdf[16, 16]), abs(sdf[8, 8]))

        empty = compute_normalized_sdf(np.zeros((8, 8), dtype=np.float32))
        full = compute_normalized_sdf(np.ones((8, 8), dtype=np.float32))
        self.assertTrue(np.all(empty == -1.0))
        self.assertTrue(np.all(full == 1.0))

    def test_sdf_supports_multiple_regions(self):
        mask = np.zeros((32, 32), dtype=np.float32)
        mask[3:8, 3:8] = 1.0
        mask[20:28, 20:28] = 1.0
        sdf = compute_normalized_sdf(mask)
        self.assertGreater(sdf[5, 5], 0.0)
        self.assertGreater(sdf[24, 24], 0.0)
        self.assertLess(sdf[15, 15], 0.0)

    def test_dataset_generates_sdf_after_transform(self):
        with tempfile.TemporaryDirectory(dir=".") as temp_dir:
            os.makedirs(os.path.join(temp_dir, "images"))
            os.makedirs(os.path.join(temp_dir, "masks", "0"))
            with open(os.path.join(temp_dir, "train.txt"), "w", encoding="utf-8") as file:
                file.write("case\n")
            with open(os.path.join(temp_dir, "val.txt"), "w", encoding="utf-8") as file:
                file.write("case\n")

            image = np.zeros((16, 16, 3), dtype=np.uint8)
            mask = np.zeros((16, 16), dtype=np.uint8)
            mask[4:12, 4:12] = 255
            cv2.imwrite(os.path.join(temp_dir, "images", "case.png"), image)
            cv2.imwrite(os.path.join(temp_dir, "masks", "0", "case.png"), mask)

            dataset = MedicalDataSets(
                base_dir=temp_dir,
                split="train",
                transform=Compose([Resize(32, 32)]),
                train_file_dir="train.txt",
                val_file_dir="val.txt",
                return_sdf=True,
            )
            sample = dataset[0]
            label = sample["label"][0] > 0.5
            sdf = sample["sdf"][0]
            self.assertEqual(label.shape, sdf.shape)
            self.assertTrue(np.all(sdf[label] > 0.0))
            self.assertTrue(np.all(sdf[~label] < 0.0))

    def test_output_contract_and_zero_initialization(self):
        model = self._small_model().eval()
        self.assertEqual(model.sdfr.sdf_head.weight.count_nonzero().item(), 0)
        self.assertEqual(model.sdfr.refine[-1].weight.count_nonzero().item(), 0)
        with torch.no_grad():
            outputs = model(torch.randn(2, 3, 32, 32))
        self.assertEqual(set(outputs), {"seg", "coarse", "uncertainty", "sdf"})
        self.assertEqual(outputs["seg"].shape, (2, 1, 32, 32))
        self.assertEqual(outputs["coarse"].shape, (2, 1, 4, 4))
        self.assertEqual(outputs["sdf"].shape, (2, 1, 32, 32))
        self.assertTrue(torch.equal(outputs["sdf"], torch.zeros_like(outputs["sdf"])))

    def test_schedule_zero_keeps_feature_unchanged(self):
        module = SignedDistanceBoundaryRefinement(channels=8).eval()
        module.set_schedule_scale(0.0)
        feature = torch.randn(2, 8, 16, 16)
        with torch.no_grad():
            refined, _ = module(feature)
        self.assertTrue(torch.equal(refined, feature))

    def test_refine_scale_is_bounded(self):
        module = SignedDistanceBoundaryRefinement(channels=8)
        self.assertAlmostEqual(module.effective_refine_scale().item(), 0.05, places=5)
        for value in (-100.0, 0.0, 100.0):
            module.refine_scale_raw.data.fill_(value)
            scale = module.effective_refine_scale().item()
            self.assertGreaterEqual(scale, 0.0)
            self.assertLessEqual(scale, 0.300001)

    def test_segmentation_loss_does_not_update_sdf_head(self):
        model = self._small_model().train()
        model.set_sdfr_refine_schedule_scale(1.0)
        outputs = model(torch.randn(2, 3, 32, 32))
        outputs["seg"].mean().backward()
        self.assertIsNone(model.sdfr.sdf_head.weight.grad)
        self.assertIsNone(model.sdfr.sdf_features[0].weight.grad)

    def test_sdf_loss_updates_sdf_head_and_shared_decoder(self):
        model = self._small_model().train()
        outputs = model(torch.randn(2, 3, 32, 32))
        target = torch.randint(0, 2, (2, 1, 32, 32)).float()
        target_sdf = torch.randn(2, 1, 32, 32).clamp(-1.0, 1.0)
        SDFRLoss()(outputs, target, target_sdf).backward()
        self.assertIsNotNone(model.sdfr.sdf_head.weight.grad)
        self.assertGreater(model.sdfr.sdf_head.weight.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(model.Up_conv2.conv[0].weight.grad)

    def test_sdfr_loss_components_sum_to_total(self):
        outputs = self._small_model().eval()(torch.randn(2, 3, 32, 32))
        target = torch.randint(0, 2, (2, 1, 32, 32)).float()
        target_sdf = torch.randn(2, 1, 32, 32).clamp(-1.0, 1.0)
        total, components = SDFRLoss()(outputs, target, target_sdf, return_components=True)
        expected = (
            components["seg"]
            + components["coarse_weighted"]
            + components["sdf_weighted"]
        )
        self.assertTrue(torch.allclose(total, expected))
        self.assertTrue(torch.allclose(total, components["total"]))

    def test_two_stage_training_schedule(self):
        args = Namespace(
            sdfr_sdf_loss_weight=0.2,
            sdfr_sdf_warmup_epochs=10,
            sdfr_refine_start_epoch=10,
            sdfr_refine_warmup_epochs=30,
        )
        expected = {
            0: (0.0, 0.0),
            5: (0.1, 0.0),
            10: (0.2, 0.0),
            25: (0.2, 0.5),
            40: (0.2, 1.0),
        }
        for epoch, (sdf_weight, refine_scale) in expected.items():
            self.assertAlmostEqual(training_main.get_sdfr_sdf_weight(args, epoch), sdf_weight)
            self.assertAlmostEqual(
                training_main.get_sdfr_refine_schedule_scale(args, epoch),
                refine_scale,
            )

    def test_training_schedule_configures_model(self):
        model = self._small_model()
        args = Namespace(
            model="CMUNeXt_HSPM_SDFR",
            hspm_mixer_mode="stable",
            hspm_coarse_loss_weight=0.1,
            hspm_coarse_loss_final_weight=0.02,
            hspm_coarse_loss_decay_epochs=150,
            hspm_prototype_warmup_epochs=20,
            sdfr_refine_start_epoch=10,
            sdfr_refine_warmup_epochs=30,
        )
        training_main.configure_hspm_epoch(args, model, 25)
        self.assertAlmostEqual(model.sdfr.schedule_scale, 0.5)

    def test_training_and_inference_entries_build_sdfr(self):
        args = Namespace(**vars(training_main.args))
        args.model = "CMUNeXt_HSPM_SDFR"
        args.hspm_mixer_mode = "stable"
        args.hspm_gamma_init = 0.05
        args.hspm_temperature = 0.5
        with mock.patch.object(torch.nn.Module, "cuda", lambda module: module):
            training_model = training_main.get_model(args)
            criterion = training_main.get_criterion(args)
        self.assertIsInstance(training_model, CMUNeXt_HSPM_SDFR)
        self.assertIsInstance(criterion, SDFRLoss)

        inference_model = inference_main.build_model(args, parser=None)
        self.assertIsInstance(inference_model, CMUNeXt_HSPM_SDFR)

    def test_training_compute_loss_uses_dynamic_sdf_weight(self):
        args = Namespace(model="CMUNeXt_HSPM_SDFR")
        model = self._small_model().eval()
        outputs = model(torch.randn(2, 3, 32, 32))
        target = torch.randint(0, 2, (2, 1, 32, 32)).float()
        sampled_batch = {"sdf": torch.randn(2, 1, 32, 32).clamp(-1.0, 1.0)}
        _, components = training_main.compute_loss(
            args,
            SDFRLoss(),
            outputs,
            target,
            sampled_batch=sampled_batch,
            aux_weight=0.1,
            sdf_weight=0.0,
        )
        self.assertEqual(components["sdf_weighted"].item(), 0.0)

    def test_state_dict_strictly_reloads(self):
        model = self._small_model()
        rebuilt = self._small_model()
        rebuilt.load_state_dict(model.state_dict(), strict=True)


if __name__ == "__main__":
    unittest.main()
