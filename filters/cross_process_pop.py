"""
cross_process_pop.py - Filter 9: Cross-Process Pop (Punchy color-shifted tones).

Technique: Independent channel S-curve LUTs + HSV saturation boost (~30%).
"""

import cv2
import numpy as np

# Per-channel S-curve lookup tables for Cross-Process effect
_LUT_B = np.zeros((256, 1), dtype=np.uint8)
_LUT_G = np.zeros((256, 1), dtype=np.uint8)
_LUT_R = np.zeros((256, 1), dtype=np.uint8)

for i in range(256):
    x = i / 255.0
    # Blue: boosted shadows, compressed highlights
    b = 1.0 / (1.0 + np.exp(-6.0 * (x - 0.4)))
    # Green: gentle contrast S-curve
    g = 1.0 / (1.0 + np.exp(-7.0 * (x - 0.5)))
    # Red: boosted highlights, darker shadows
    r = 1.0 / (1.0 + np.exp(-8.0 * (x - 0.6)))

    _LUT_B[i] = int(np.clip(b * 255.0, 0, 255))
    _LUT_G[i] = int(np.clip(g * 255.0, 0, 255))
    _LUT_R[i] = int(np.clip(r * 255.0, 0, 255))


def apply_cross_process_pop(frame: np.ndarray) -> np.ndarray:
    """Apply Cross-Process Pop vibrant color-shift filter to frame."""
    b, g, r = cv2.split(frame)

    b_processed = cv2.LUT(b, _LUT_B)
    g_processed = cv2.LUT(g, _LUT_G)
    r_processed = cv2.LUT(r, _LUT_R)

    merged = cv2.merge([b_processed, g_processed, r_processed])

    # Boost saturation ~30% in HSV space
    hsv = cv2.cvtColor(merged, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.30, 0, 255)

    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
