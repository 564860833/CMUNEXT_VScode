import cv2
import numpy as np


def compute_normalized_sdf(mask, truncation_ratio=0.08):
    """Build a truncated signed distance field with positive values inside."""
    if truncation_ratio <= 0:
        raise ValueError("truncation_ratio must be positive.")

    mask = np.asarray(mask)
    if mask.ndim == 3:
        if mask.shape[-1] != 1:
            raise ValueError("mask must be 2D or have a singleton channel dimension.")
        mask = mask[..., 0]
    if mask.ndim != 2:
        raise ValueError("mask must be 2D or have a singleton channel dimension.")

    foreground = mask > 0.5
    if not foreground.any():
        return np.full(mask.shape, -1.0, dtype=np.float32)
    if foreground.all():
        return np.full(mask.shape, 1.0, dtype=np.float32)

    foreground_u8 = foreground.astype(np.uint8)
    background_u8 = (~foreground).astype(np.uint8)
    inside_distance = cv2.distanceTransform(
        foreground_u8,
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE,
    )
    outside_distance = cv2.distanceTransform(
        background_u8,
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE,
    )
    signed_distance = inside_distance - outside_distance
    truncation_distance = float(truncation_ratio) * min(mask.shape)
    return np.clip(
        signed_distance / max(truncation_distance, 1e-6),
        -1.0,
        1.0,
    ).astype(np.float32)
