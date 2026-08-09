"""
arctic_blue.py - Filter 3: Arctic Blue (Cold, crisp, blue-shifted).

Technique: Color matrix boosting B/G and cutting R -> contrast boost.
"""

import cv2
import numpy as np

# 3x3 cold blue color matrix for BGR transformation
_ARCTIC_MATRIX = np.array([
    [1.25, 0.10, 0.00],  # Blue channel boost
    [0.05, 1.05, 0.00],  # Green channel
    [0.00, 0.05, 0.75],  # Red channel cut
], dtype=np.float32)

# Contrast boost LUT
_CONTRAST_LUT = np.zeros((256, 1), dtype=np.uint8)
for i in range(256):
    # Midtone contrast stretch
    val = (i - 128) * 1.15 + 128
    _CONTRAST_LUT[i] = int(np.clip(val, 0, 255))


def apply_arctic_blue(frame: np.ndarray) -> np.ndarray:
    """Apply Arctic Blue cold crisp filter to frame."""
    transformed = cv2.transform(frame, _ARCTIC_MATRIX)
    # Apply contrast boost
    return cv2.LUT(transformed, _CONTRAST_LUT)
