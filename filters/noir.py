"""
noir.py - Filter 1: Noir (High-contrast black & white, filmic).

Technique: Grayscale conversion -> S-curve contrast LUT -> film grain -> 3-channel BGR.
"""

import cv2
import numpy as np

# Precomputed S-curve contrast LUT
_NOIR_LUT = np.zeros((256, 1), dtype=np.uint8)
for i in range(256):
    # S-curve sigmoidal contrast adjustment
    x = i / 255.0
    s = 1.0 / (1.0 + np.exp(-10.0 * (x - 0.5)))
    _NOIR_LUT[i] = int(np.clip(s * 255.0, 0, 255))


def apply_noir(frame: np.ndarray) -> np.ndarray:
    """Apply Noir high-contrast filmic black & white filter to frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    contrast_gray = cv2.LUT(gray, _NOIR_LUT)

    # Convert back to 3-channel BGR
    bgr = cv2.cvtColor(contrast_gray, cv2.COLOR_GRAY2BGR)

    # Add subtle filmic grain
    h, w, _ = frame.shape
    noise = np.random.randint(-12, 13, (h, w, 1), dtype=np.int16)
    bgr_int = bgr.astype(np.int16) + noise
    return np.clip(bgr_int, 0, 255).astype(np.uint8)
