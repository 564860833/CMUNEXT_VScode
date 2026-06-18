import unittest

import torch
import torch.nn.functional as F

from src.network.conv_based.CMUNeXt import cmunext
from src.network.conv_based.CMUNeXt_FBDM import (
    cmunext_fbdm,
    cmunext_fbdm_l,
    cmunext_fbdm_s,
)
from src.utils.losses import BCEDiceLoss, EdgeSupervisionLoss, FBDMLoss


class CMUNeXtFBDMTests(unittest.TestCase):
    def _small_model(self, **kwargs):
        return cmunext_fbdm(
            dims=(4, 8, 16, 20, 24),
            depths=(1, 1, 1, 1, 1),
            kernels=(3, 3, 3, 3, 3),
            **kwargs,
        )

    def test_output_contract_is_fbdm_only(self):
        model = self._small_model().eval()
        with torch.no_grad():
            outputs = model(torch.randn(2, 3, 32, 32))

        self.assertEqual(set(outputs), {"seg", "edge"})
        self.assertNotIn("coarse", outputs)
        self.assertNotIn("uncertainty", outputs)
        self.assertFalse(hasattr(model, "prototype_mixer"))
        self.assertEqual(outputs["seg"].shape, (2, 1, 32, 32))
        self.assertEqual(outputs["edge"].shape, (2, 1, 32, 32))
        self.assertFalse(model.fbdm1.use_hspm_prior)
        self.assertFalse(model.fbdm1.edge_aux_only)
        self.assertAlmostEqual(model.fbdm1.effective_gamma().item(), 0.03, places=5)

    def test_edge_aux_only_output_contract_is_fbdm_only(self):
        model = self._small_model(fbdm_edge_aux_only=True).eval()
        with torch.no_grad():
            outputs = model(torch.randn(2, 3, 32, 32))

        self.assertTrue(model.fbdm1.edge_aux_only)
        self.assertEqual(set(outputs), {"seg", "edge"})
        self.assertEqual(outputs["seg"].shape, (2, 1, 32, 32))
        self.assertEqual(outputs["edge"].shape, (2, 1, 32, 32))

    def test_factories_smoke_test(self):
        for factory in (cmunext_fbdm_s, cmunext_fbdm_l):
            model = factory().eval()
            with torch.no_grad():
                outputs = model(torch.randn(1, 3, 32, 32))
            self.assertEqual(outputs["seg"].shape, (1, 1, 32, 32))
            self.assertEqual(outputs["edge"].shape, (1, 1, 32, 32))

    def test_fbdm_loss_backward(self):
        target = torch.zeros(2, 1, 16, 16)
        target[:, :, 4:12, 5:11] = 1.0
        outputs = {
            "seg": torch.randn(2, 1, 16, 16, requires_grad=True),
            "edge": torch.randn(2, 1, 16, 16, requires_grad=True),
        }

        loss = FBDMLoss(edge_weight=0.05)(outputs, target)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

        self.assertIsNotNone(outputs["seg"].grad)
        self.assertIsNotNone(outputs["edge"].grad)

    def test_edge_supervision_legacy_matches_existing_loss(self):
        logits = torch.randn(2, 1, 8, 8)
        target = torch.zeros_like(logits)
        target[:, :, 2:6, 3:5] = 1.0

        expected = BCEDiceLoss()(logits, target)
        actual = EdgeSupervisionLoss(loss_type="legacy")(logits, target)

        self.assertTrue(torch.allclose(actual, expected))

    def test_balanced_edge_loss_matches_manual_formula(self):
        logits = torch.tensor([[[[0.0, 1.0], [-1.0, 0.5]]]])
        target = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
        pos_weight = 3.0

        actual = EdgeSupervisionLoss(
            loss_type="balanced_bce_dice",
            pos_weight=pos_weight,
        )(logits, target)
        bce = F.binary_cross_entropy_with_logits(
            logits,
            target,
            pos_weight=torch.tensor([pos_weight]),
        )
        probability = torch.sigmoid(logits).reshape(1, -1)
        flat_target = target.reshape(1, -1)
        intersection = (probability * flat_target).sum(dim=1)
        dice = 1.0 - (
            (2.0 * intersection + 1e-5)
            / (probability.sum(dim=1) + flat_target.sum(dim=1) + 1e-5)
        ).mean()

        self.assertTrue(torch.allclose(actual, 0.5 * bce + 0.5 * dice))

    def test_focal_edge_loss_matches_manual_formula(self):
        logits = torch.tensor([[[[0.0, 1.0], [-1.0, 0.5]]]])
        target = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
        alpha = 0.9
        gamma = 2.0

        actual = EdgeSupervisionLoss(
            loss_type="focal_dice",
            focal_alpha=alpha,
            focal_gamma=gamma,
        )(logits, target)
        elementwise_bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        probability = torch.sigmoid(logits)
        p_t = target * probability + (1.0 - target) * (1.0 - probability)
        alpha_t = target * alpha + (1.0 - target) * (1.0 - alpha)
        focal = (alpha_t * (1.0 - p_t).pow(gamma) * elementwise_bce).mean()
        flat_probability = probability.reshape(1, -1)
        flat_target = target.reshape(1, -1)
        intersection = (flat_probability * flat_target).sum(dim=1)
        dice = 1.0 - (
            (2.0 * intersection + 1e-5)
            / (flat_probability.sum(dim=1) + flat_target.sum(dim=1) + 1e-5)
        ).mean()

        self.assertTrue(torch.allclose(actual, 0.5 * focal + 0.5 * dice))

    def test_edge_loss_variants_backpropagate_on_sparse_targets(self):
        positive_gradients = {}
        for loss_type in ("legacy", "balanced_bce_dice", "focal_dice"):
            logits = torch.zeros(1, 1, 8, 8, requires_grad=True)
            target = torch.zeros_like(logits)
            target[:, :, 3, 4] = 1.0
            loss = EdgeSupervisionLoss(loss_type=loss_type, pos_weight=20.0)(logits, target)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            positive_gradients[loss_type] = logits.grad[0, 0, 3, 4].abs().item()
            self.assertGreater(positive_gradients[loss_type], 0.0)

        self.assertGreater(
            positive_gradients["balanced_bce_dice"],
            positive_gradients["legacy"],
        )

    def test_edge_supervision_rejects_invalid_options(self):
        invalid_kwargs = (
            {"loss_type": "unknown"},
            {"pos_weight": 0.0},
            {"focal_alpha": 0.0},
            {"focal_alpha": 1.0},
            {"focal_gamma": -1.0},
        )
        for kwargs in invalid_kwargs:
            with self.assertRaises(ValueError):
                EdgeSupervisionLoss(**kwargs)

    def test_baseline_cmunext_output_contract_is_unchanged(self):
        model = cmunext(
            dims=(4, 8, 16, 20, 24),
            depths=(1, 1, 1, 1, 1),
            kernels=(3, 3, 3, 3, 3),
        ).eval()
        with torch.no_grad():
            outputs = model(torch.randn(2, 3, 32, 32))

        self.assertIsInstance(outputs, torch.Tensor)
        self.assertEqual(outputs.shape, (2, 1, 32, 32))


if __name__ == "__main__":
    unittest.main()
