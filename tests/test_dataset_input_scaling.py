import os
import tempfile
import unittest

import cv2
import numpy as np
from albumentations import Compose

from src.dataloader.dataset import MedicalDataSets


class DatasetInputScalingTests(unittest.TestCase):
    def _build_dataset(self, base_dir, divide_image_by_255):
        return MedicalDataSets(
            base_dir=base_dir,
            split="val",
            transform=Compose([]),
            val_file_dir="val.txt",
            divide_image_by_255=divide_image_by_255,
        )

    def test_divide_image_by_255_is_opt_in(self):
        with tempfile.TemporaryDirectory(dir=".") as temp_dir:
            os.makedirs(os.path.join(temp_dir, "images"))
            os.makedirs(os.path.join(temp_dir, "masks", "0"))
            with open(os.path.join(temp_dir, "val.txt"), "w", encoding="utf-8") as file:
                file.write("case\n")

            image = np.full((2, 2, 3), 255, dtype=np.uint8)
            mask = np.zeros((2, 2), dtype=np.uint8)
            cv2.imwrite(os.path.join(temp_dir, "images", "case.png"), image)
            cv2.imwrite(os.path.join(temp_dir, "masks", "0", "case.png"), mask)

            current_scale = self._build_dataset(temp_dir, False)[0]["image"]
            official_cmunext_scale = self._build_dataset(temp_dir, True)[0]["image"]

            self.assertEqual(current_scale.dtype, np.float32)
            self.assertTrue(np.all(current_scale == 255.0))
            self.assertTrue(np.all(official_cmunext_scale == 1.0))


if __name__ == "__main__":
    unittest.main()
