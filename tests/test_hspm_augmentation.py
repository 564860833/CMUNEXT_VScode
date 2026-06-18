import sys
import unittest
from argparse import Namespace
from unittest import mock

import numpy as np
from albumentations import (
    GaussianBlur,
    HorizontalFlip,
    MultiplicativeNoise,
    OneOf,
    RandomBrightnessContrast,
    RandomGamma,
    RandomRotate90,
    Resize,
    ShiftScaleRotate,
)
from albumentations.augmentations import transforms

with mock.patch.object(sys, "argv", ["main.py"]):
    import main as training_main


class HSPMAugmentationTests(unittest.TestCase):
    @staticmethod
    def _args(use_extra_aug):
        return Namespace(use_extra_aug=use_extra_aug)

    @staticmethod
    def _types(compose):
        return [type(transform) for transform in compose.transforms]

    def test_basic_augmentation_remains_default_without_extra_aug(self):
        augmentation = training_main.build_train_transform(self._args(False), 64)

        self.assertEqual(
            self._types(augmentation),
            [RandomRotate90, transforms.Flip, Resize, transforms.Normalize],
        )

    def test_extra_augmentation_profile_argument_is_removed(self):
        args = training_main.parser.parse_args([])
        self.assertFalse(hasattr(args, "extra_aug_profile"))

    def test_extra_augmentation_uses_mild_image_style_perturbations(self):
        augmentation = training_main.build_train_transform(self._args(True), 64)
        top_level_types = self._types(augmentation)
        style_augmentations = augmentation.transforms[2]
        style_types = self._types(style_augmentations)

        self.assertEqual(
            top_level_types,
            [
                HorizontalFlip,
                ShiftScaleRotate,
                OneOf,
                MultiplicativeNoise,
                GaussianBlur,
                Resize,
                transforms.Normalize,
            ],
        )
        self.assertEqual(
            style_types,
            [RandomBrightnessContrast, RandomGamma],
        )

        affine = augmentation.transforms[1]
        multiplicative_noise = augmentation.transforms[3]
        self.assertEqual(affine.shift_limit_x, (-0.04, 0.04))
        self.assertEqual(affine.shift_limit_y, (-0.04, 0.04))
        self.assertEqual(affine.scale_limit, (0.92, 1.08))
        self.assertEqual(affine.rotate_limit, (-12, 12))
        self.assertAlmostEqual(affine.p, 0.4)
        self.assertEqual(style_augmentations.transforms[0].brightness_limit, (-0.1, 0.1))
        self.assertEqual(style_augmentations.transforms[0].contrast_limit, (-0.1, 0.1))
        self.assertEqual(style_augmentations.transforms[1].gamma_limit, (90, 110))
        self.assertAlmostEqual(style_augmentations.p, 0.35)
        self.assertEqual(multiplicative_noise.multiplier, (0.97, 1.03))
        self.assertFalse(multiplicative_noise.per_channel)
        self.assertTrue(multiplicative_noise.elementwise)
        self.assertAlmostEqual(multiplicative_noise.p, 0.20)
        self.assertEqual(augmentation.transforms[4].blur_limit, (3, 3))
        self.assertAlmostEqual(augmentation.transforms[4].p, 0.05)

    def test_extra_augmentation_keeps_mask_binary_and_resizes_image_and_mask(self):
        augmentation = training_main.build_train_transform(self._args(True), 64)
        image = np.full((40, 52, 3), 128, dtype=np.uint8)
        mask = np.zeros((40, 52), dtype=np.uint8)
        mask[10:30, 18:38] = 255

        result = augmentation(image=image, mask=mask)

        self.assertEqual(result["image"].shape, (64, 64, 3))
        self.assertEqual(result["mask"].shape, (64, 64))
        self.assertTrue(set(np.unique(result["mask"])).issubset({0, 255}))


if __name__ == "__main__":
    unittest.main()
