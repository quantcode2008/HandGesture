"""
cyberpunk_duotone.py - Filter 5: Cyberpunk Duotone (Magenta-shadow / Cyan-highlight).

Technique: Grayscale luminance -> 3-channel BGR -> custom 2-color gradient LUT.
Dark pixels map to Deep Magenta (R=180, G=10, B=160),
Bright pixels map to Electric Cyan (R=0, G=240, B=255).
"""

import cv2
import numpy as np

# Precomputed 256x1 3-channel (BGR) LUT for Cyberpunk Duotone
_CYBERPUNK_LUT = np.zeros((256, 1, 3), dtype=np.uint8)

# Color 1 (Shadows): Deep Magenta/Violet (B=160, G=10, R=180)
# Color 2 (Highlights): Electric Cyan (B=255, G=240, R=0)
for i in range(256):
    t = i / 255.0
    t_sig = 1.0 / (1.0 + np.exp(-8.0 * (t - 0.5)))

    b = int(np.clip(160.0 * (1.0 - t_sig) + 255.0 * t_sig, 0, 255))
    g = int(np.clip(10.0 * (1.0 - t_sig) + 240.0 * t_sig, 0, 255))
    r = int(np.clip(180.0 * (1.0 - t_sig) + 0.0 * t_sig, 0, 255))

    _CYBERPUNK_LUT[i, 0] = [b, g, r]


def apply_cyberpunk_duotone(frame: np.ndarray) -> np.ndarray:
    """Apply Cyberpunk Duotone (Magenta / Cyan) gradient filter to frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bgr_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return cv2.LUT(bgr_gray, _CYBERPUNK_LUT)
