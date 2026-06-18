import os
import sys
import tempfile
import unittest
from argparse import Namespace
from unittest import mock

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.dataloader.dataset import (
    TARGETED_SAMPLING_GROUPS,
    TrackingWeightedRandomSampler,
    build_targeted_sampling_metadata,
)

with mock.patch.object(sys, "argv", ["main.py"]):
    import main as training_main


class TargetedSamplingTests(unittest.TestCase):
    @staticmethod
    def _write_mask(base_dir, case_name, foreground_pixels, extension='.png'):
        mask_directory = os.path.join(base_dir, 'masks', '0')
        os.makedirs(mask_directory, exist_ok=True)
        mask = np.zeros((10, 20), dtype=np.uint8)
        mask.flat[:foreground_pixels] = 255
        path = os.path.join(mask_directory, case_name + extension)
        if not cv2.imwrite(path, mask):
            raise RuntimeError(f'Failed to write test mask: {path}')

    def _build_six_group_dataset(self, base_dir):
        cases_and_pixels = [
            ('benign small', 9),
            ('benign medium', 10),
            ('benign large', 41),
            ('malignant small', 9),
            ('malignant medium', 40),
            ('malignant large', 41),
        ]
        for case_name, foreground_pixels in cases_and_pixels:
            self._write_mask(base_dir, case_name, foreground_pixels)
        return [case_name for case_name, _ in cases_and_pixels]

    def test_area_boundaries_groups_and_weights(self):
        with tempfile.TemporaryDirectory(dir='.') as base_dir:
            sample_list = self._build_six_group_dataset(base_dir)
            metadata = build_targeted_sampling_metadata(base_dir, sample_list)

        self.assertEqual(metadata.groups, TARGETED_SAMPLING_GROUPS)
        self.assertEqual(metadata.weights.tolist(), [1.0, 1.0, 1.0, 1.2, 1.8, 1.3])
        self.assertEqual(metadata.original_counts, {group: 1 for group in TARGETED_SAMPLING_GROUPS})
        self.assertAlmostEqual(sum(metadata.expected_counts.values()), len(sample_list))

    def test_sampler_is_fixed_length_tracked_and_reproducible(self):
        weights = torch.tensor([1.0, 1.2, 1.8, 1.3], dtype=torch.double)
        groups = (
            'benign_small',
            'malignant_small',
            'malignant_medium',
            'malignant_large',
        )

        def sampled_indices(seed):
            generator = torch.Generator().manual_seed(seed)
            sampler = TrackingWeightedRandomSampler(
                weights,
                num_samples=452,
                replacement=True,
                generator=generator,
                groups=groups,
            )
            indices = list(sampler)
            return indices, sampler

        first_indices, first_sampler = sampled_indices(41)
        repeated_indices, _ = sampled_indices(41)
        other_indices, _ = sampled_indices(42)

        self.assertEqual(len(first_indices), 452)
        self.assertEqual(first_indices, repeated_indices)
        self.assertNotEqual(first_indices, other_indices)
        self.assertEqual(sum(first_sampler.last_group_counts().values()), 452)

        loader = DataLoader(
            TensorDataset(torch.arange(4)),
            batch_size=8,
            sampler=TrackingWeightedRandomSampler(
                weights,
                num_samples=452,
                replacement=True,
                generator=torch.Generator().manual_seed(41),
                groups=groups,
            ),
        )
        self.assertEqual(len(loader), 57)

    def test_sampling_defaults_preserve_uniform_and_augmentation_independence(self):
        args = training_main.parser.parse_args([])
        self.assertEqual(args.sampling_mode, 'uniform')
        self.assertIsNone(
            training_main.build_train_sampler(args, Namespace(sample_list=[]))
        )

        targeted_args = training_main.parser.parse_args(['--sampling_mode', 'targeted'])
        self.assertFalse(targeted_args.use_extra_aug)

    def test_invalid_thresholds_and_weights_are_rejected(self):
        args = training_main.parser.parse_args([])
        args.targeted_small_area_threshold = 0.20
        args.targeted_large_area_threshold = 0.05
        with self.assertRaisesRegex(ValueError, 'area thresholds'):
            training_main.validate_sampling_args(args)

        args = training_main.parser.parse_args([])
        args.targeted_malignant_medium_weight = 0.0
        with self.assertRaisesRegex(ValueError, 'weights must be positive'):
            training_main.validate_sampling_args(args)

    def test_unknown_case_prefix_and_missing_mask_are_rejected(self):
        with tempfile.TemporaryDirectory(dir='.') as base_dir:
            self._write_mask(base_dir, 'unknown case', 10, extension='.jpg')
            with self.assertRaisesRegex(ValueError, 'case names starting'):
                build_targeted_sampling_metadata(base_dir, ['unknown case'])

            with self.assertRaisesRegex(FileNotFoundError, 'missing case'):
                build_targeted_sampling_metadata(base_dir, ['malignant missing case'])

    def test_duplicated_resampling_list_is_rejected(self):
        with tempfile.TemporaryDirectory(dir='.') as base_dir:
            self._write_mask(base_dir, 'malignant duplicated case', 20)
            with self.assertRaisesRegex(ValueError, 'non-duplicated training list'):
                build_targeted_sampling_metadata(
                    base_dir,
                    ['malignant duplicated case', 'malignant duplicated case'],
                )

    def test_tracking_sampler_requires_one_group_per_weight(self):
        with self.assertRaisesRegex(ValueError, 'groups must match'):
            TrackingWeightedRandomSampler(
                weights=[1.0, 2.0],
                num_samples=2,
                groups=['benign_small'],
            )


if __name__ == '__main__':
    unittest.main()
