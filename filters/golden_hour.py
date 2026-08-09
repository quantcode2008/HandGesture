"""
golden_hour.py - Filter 4: Golden Hour (Warm glow on highlights).

Technique: Warm tone boost + Gaussian bright-pass bloom layer screen-blended.
"""

import cv2
import numpy as np

# Golden warm matrix (boost R & G, reduce B)
_GOLDEN_MATRIX = np.array([
    [0.75, 0.05, 0.00],  # Blue channel
    [0.00, 1.05, 0.05],  # Green channel
    [0.05, 0.15, 1.20],  # Red channel
], dtype=np.float32)


def apply_golden_hour(frame: np.ndarray) -> np.ndarray:
    """Apply Golden Hour warm highlight glow filter."""
    warm = cv2.transform(frame, _GOLDEN_MATRIX)

    # Extract bright regions for bloom layer (threshold bright highlights > 140)
    gray = cv2.cvtColor(warm, cv2.COLOR_BGR2GRAY)
    _, bright_mask = cv2.threshold(gray, 140, 255, cv2.THRESH_TOZERO)

    # Blur bright regions to create soft glow bloom
    bloom = cv2.GaussianBlur(bright_mask, (31, 31), 11)
    bloom_bgr = cv2.cvtColor(bloom, cv2.COLOR_GRAY2BGR).astype(np.float32)

    # Screen blend: 1 - (1 - A)*(1 - B)
    warm_f = warm.astype(np.float32) / 255.0
    bloom_f = (bloom_bgr / 255.0) * 0.4  # Bloom strength 40%

    blended = 1.0 - (1.0 - warm_f) * (1.0 - bloom_f)
    return np.clip(blended * 255.0, 0, 255).astype(np.uint8)
