import os
from collections import Counter
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import cv2
import numpy as np


SUPPORTED_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')
TARGETED_SAMPLING_GROUPS = (
    'benign_small',
    'benign_medium',
    'benign_large',
    'malignant_small',
    'malignant_medium',
    'malignant_large',
)


@dataclass(frozen=True)
class TargetedSamplingMetadata:
    weights: torch.Tensor
    groups: tuple
    original_counts: dict
    expected_counts: dict


def _find_case_file(directory, case_name):
    for extension in SUPPORTED_IMAGE_EXTENSIONS:
        path = os.path.join(directory, case_name + extension)
        if os.path.exists(path):
            return path
    return None


def _targeted_sampling_group(case_name, area_ratio, small_threshold, large_threshold):
    normalized_name = case_name.strip().lower()
    if normalized_name.startswith('benign'):
        pathology = 'benign'
    elif normalized_name.startswith('malignant'):
        pathology = 'malignant'
    else:
        raise ValueError(
            "Targeted sampling requires case names starting with 'benign' or 'malignant'; "
            f"got '{case_name}'."
        )

    if area_ratio < small_threshold:
        size = 'small'
    elif area_ratio <= large_threshold:
        size = 'medium'
    else:
        size = 'large'
    return f'{pathology}_{size}'


def build_targeted_sampling_metadata(
        base_dir,
        sample_list,
        small_threshold=0.05,
        large_threshold=0.20,
        malignant_small_weight=1.2,
        malignant_medium_weight=1.8,
        malignant_large_weight=1.3,
):
    if not 0.0 <= small_threshold < large_threshold <= 1.0:
        raise ValueError(
            'Targeted sampling area thresholds must satisfy '
            '0 <= small_threshold < large_threshold <= 1.'
        )

    malignant_weights = {
        'malignant_small': float(malignant_small_weight),
        'malignant_medium': float(malignant_medium_weight),
        'malignant_large': float(malignant_large_weight),
    }
    if any(weight <= 0.0 for weight in malignant_weights.values()):
        raise ValueError('Targeted sampling weights must be positive.')
    if len(set(sample_list)) != len(sample_list):
        raise ValueError(
            'Targeted sampling requires the original non-duplicated training list; '
            'duplicate case names were found.'
        )

    mask_directory = os.path.join(base_dir, 'masks', '0')
    groups = []
    weights = []
    for case_name in sample_list:
        mask_path = _find_case_file(mask_directory, case_name)
        if mask_path is None:
            raise FileNotFoundError(
                f"Mask file not found for case '{case_name}' in {mask_directory}. "
                'Tried .png, .jpg, and .jpeg.'
            )
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to read mask for case '{case_name}' from {mask_path}.")

        area_ratio = float(np.count_nonzero(mask > 127)) / float(mask.size)
        group = _targeted_sampling_group(
            case_name,
            area_ratio,
            small_threshold,
            large_threshold,
        )
        groups.append(group)
        weights.append(malignant_weights.get(group, 1.0))

    if not weights:
        raise ValueError('Targeted sampling requires at least one training sample.')

    original_counts = Counter(groups)
    total_weight = float(sum(weights))
    expected_counts = {
        group: len(weights)
        * sum(weight for weight, sample_group in zip(weights, groups) if sample_group == group)
        / total_weight
        for group in TARGETED_SAMPLING_GROUPS
    }
    return TargetedSamplingMetadata(
        weights=torch.as_tensor(weights, dtype=torch.double),
        groups=tuple(groups),
        original_counts={group: original_counts.get(group, 0) for group in TARGETED_SAMPLING_GROUPS},
        expected_counts=expected_counts,
    )


class TrackingWeightedRandomSampler(WeightedRandomSampler):
    def __init__(
            self,
            weights,
            num_samples,
            replacement=True,
            generator=None,
            groups=None,
    ):
        super().__init__(
            weights=weights,
            num_samples=num_samples,
            replacement=replacement,
            generator=generator,
        )
        if groups is None or len(groups) != len(weights):
            raise ValueError('Tracking sampler groups must match the number of weights.')
        self.groups = tuple(groups)
        self.last_indices = []

    def __iter__(self):
        indices = list(super().__iter__())
        self.last_indices = indices
        return iter(indices)

    def last_group_counts(self):
        counts = Counter(self.groups[index] for index in self.last_indices)
        return {group: counts.get(group, 0) for group in TARGETED_SAMPLING_GROUPS}


class MedicalDataSets(Dataset):
    def __init__(
            self,
            base_dir=None,
            split="train",
            transform=None,
            train_file_dir="train.txt",
            val_file_dir="val.txt",
            divide_image_by_255=False,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.train_list = []
        self.semi_list = []
        self.divide_image_by_255 = bool(divide_image_by_255)

        if self.split == "train":
            with open(os.path.join(self._base_dir, train_file_dir), "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        elif self.split == "val":
            with open(os.path.join(self._base_dir, val_file_dir), "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        print("total {}  {} samples".format(len(self.sample_list), self.split))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):

        case = self.sample_list[idx]

        # 自动查找图像文件的扩展名 (png, jpg, jpeg)
        image_path = _find_case_file(os.path.join(self._base_dir, 'images'), case)

        if image_path is None:
            raise FileNotFoundError(
                f"Image file not found for case '{case}' in {os.path.join(self._base_dir, 'images')}. "
                "Tried .png, .jpg, and .jpeg.")

        # 自动查找掩码文件的扩展名 (png, jpg, jpeg)
        # 保持原始代码中 'masks/0' 的路径结构
        mask_path = _find_case_file(os.path.join(self._base_dir, 'masks', '0'), case)

        if mask_path is None:
            raise FileNotFoundError(
                f"Mask file not found for case '{case}' in {os.path.join(self._base_dir, 'masks', '0')}. "
                "Tried .png, .jpg, and .jpeg.")

        # 使用找到的路径加载图像和掩码
        image = cv2.imread(image_path)
        label = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)[..., None]

        augmented = self.transform(image=image, mask=label)
        image = augmented['image']
        label = augmented['mask']

        image = image.astype('float32')
        if self.divide_image_by_255:
            image = image / 255
        image = image.transpose(2, 0, 1)

        label = label.astype('float32') / 255
        label = label.transpose(2, 0, 1)

        sample = {
            "image": image,
            "label": label,
            "idx": idx,
            "case_name": case,
            "image_path": image_path,
            "mask_path": mask_path,
        }
        return sample
