"""
infrared_dream.py - Filter 10: Infrared Dream (Surreal false-color & foliage glow).

Technique: False-color channel matrix remap (push Red to Blue/Cyan & Green/Red to Red/Pink)
          + Gaussian bright-pass bloom layer.
"""

import cv2
import numpy as np

# False-color Infrared matrix transformation for BGR
_INFRARED_MATRIX = np.array([
    [0.90, 0.20, 0.00],  # Output B
    [0.10, 0.30, 0.80],  # Output G (shift red/foliage)
    [0.00, 0.10, 1.25],  # Output R (boost red highlights)
], dtype=np.float32)


def apply_infrared_dream(frame: np.ndarray) -> np.ndarray:
    """Apply Infrared Dream surreal false-color filter to frame."""
    false_color = cv2.transform(frame, _INFRARED_MATRIX)

    # Add soft infrared highlight bloom
    gray = cv2.cvtColor(false_color, cv2.COLOR_BGR2GRAY)
    _, bright = cv2.threshold(gray, 130, 255, cv2.THRESH_TOZERO)
    bloom = cv2.GaussianBlur(bright, (25, 25), 9)
    bloom_bgr = cv2.cvtColor(bloom, cv2.COLOR_GRAY2BGR)

    # Screen blend bloom
    f_fc = false_color.astype(np.float32) / 255.0
    f_bl = (bloom_bgr.astype(np.float32) / 255.0) * 0.35

    blended = 1.0 - (1.0 - f_fc) * (1.0 - f_bl)
    return np.clip(blended * 255.0, 0, 255).astype(np.uint8)
