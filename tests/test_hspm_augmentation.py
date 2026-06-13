import sys
import unittest
from argparse import Namespace
from unittest import mock

import numpy as np
from albumentations import (
    GaussNoise,
    GaussianBlur,
    GridDistortion,
    MultiplicativeNoise,
    OneOf,
    RandomBrightnessContrast,
    RandomGamma,
    RandomRotate90,
    Resize,
)
from albumentations.augmentations import transforms

with mock.patch.object(sys, "argv", ["main.py"]):
    import main as training_main


class HSPMAugmentationTests(unittest.TestCase):
    @staticmethod
    def _args(use_extra_aug, profile="legacy"):
        return Namespace(
            use_extra_aug=use_extra_aug,
            extra_aug_profile=profile,
        )

    @staticmethod
    def _types(compose):
        return [type(transform) for transform in compose.transforms]

    def test_basic_augmentation_remains_default_without_extra_aug(self):
        augmentation = training_main.build_train_transform(self._args(False), 64)

        self.assertEqual(
            self._types(augmentation),
            [RandomRotate90, transforms.Flip, Resize, transforms.Normalize],
        )

    def test_extra_augmentation_profile_defaults_to_legacy(self):
        args = training_main.parser.parse_args([])

        self.assertEqual(args.extra_aug_profile, "legacy")

    def test_legacy_profile_matches_existing_extra_augmentation(self):
        augmentation = training_main.build_train_transform(self._args(True), 64)

        self.assertEqual(
            self._types(augmentation),
            [
                RandomRotate90,
                transforms.Flip,
                GridDistortion,
                OneOf,
                GaussianBlur,
                Resize,
                transforms.Normalize,
            ],
        )
        self.assertEqual(
            self._types(augmentation.transforms[3]),
            [RandomBrightnessContrast, RandomGamma, GaussNoise],
        )
        grid_distortion = augmentation.transforms[2]
        style_augmentations = augmentation.transforms[3]
        blur = augmentation.transforms[4]
        self.assertEqual(grid_distortion.num_steps, 5)
        self.assertEqual(grid_distortion.distort_limit, (-0.05, 0.05))
        self.assertAlmostEqual(grid_distortion.p, 0.15)
        self.assertAlmostEqual(style_augmentations.p, 0.3)
        self.assertEqual(style_augmentations.transforms[0].brightness_limit, (-0.15, 0.15))
        self.assertEqual(style_augmentations.transforms[0].contrast_limit, (-0.15, 0.15))
        self.assertEqual(style_augmentations.transforms[1].gamma_limit, (85, 115))
        self.assertEqual(style_augmentations.transforms[2].var_limit, (10.0, 40.0))
        self.assertEqual(blur.blur_limit, (3, 5))
        self.assertAlmostEqual(blur.p, 0.15)

    def test_hspm_safe_uses_only_mild_image_style_perturbations(self):
        augmentation = training_main.build_train_transform(
            self._args(True, "hspm_safe"),
            64,
        )
        top_level_types = self._types(augmentation)
        style_augmentations = augmentation.transforms[2]
        style_types = self._types(style_augmentations)

        self.assertEqual(
            top_level_types,
            [
                RandomRotate90,
                transforms.Flip,
                OneOf,
                GaussianBlur,
                Resize,
                transforms.Normalize,
            ],
        )
        self.assertNotIn(GridDistortion, top_level_types)
        self.assertNotIn(GaussNoise, style_types)
        self.assertEqual(
            style_types,
            [RandomBrightnessContrast, RandomGamma, MultiplicativeNoise],
        )

        multiplicative_noise = style_augmentations.transforms[2]
        self.assertEqual(style_augmentations.transforms[0].brightness_limit, (-0.1, 0.1))
        self.assertEqual(style_augmentations.transforms[0].contrast_limit, (-0.1, 0.1))
        self.assertEqual(style_augmentations.transforms[1].gamma_limit, (90, 110))
        self.assertEqual(multiplicative_noise.multiplier, (0.95, 1.05))
        self.assertFalse(multiplicative_noise.per_channel)
        self.assertTrue(multiplicative_noise.elementwise)
        self.assertEqual(augmentation.transforms[3].blur_limit, (3, 3))
        self.assertAlmostEqual(augmentation.transforms[3].p, 0.05)

    def test_hspm_safe_keeps_mask_binary_and_resizes_image_and_mask(self):
        augmentation = training_main.build_train_transform(
            self._args(True, "hspm_safe"),
            64,
        )
        image = np.full((40, 52, 3), 128, dtype=np.uint8)
        mask = np.zeros((40, 52), dtype=np.uint8)
        mask[10:30, 18:38] = 255

        result = augmentation(image=image, mask=mask)

        self.assertEqual(result["image"].shape, (64, 64, 3))
        self.assertEqual(result["mask"].shape, (64, 64))
        self.assertTrue(set(np.unique(result["mask"])).issubset({0, 255}))

    def test_hspm_safe_requires_extra_augmentation_flag(self):
        with self.assertRaisesRegex(ValueError, "requires --use_extra_aug"):
            training_main.build_train_transform(
                self._args(False, "hspm_safe"),
                64,
            )


if __name__ == "__main__":
    unittest.main()
