"""
sepia_classic.py - Filter 6: Sepia Classic (Traditional warm sepia tone).

Technique: Standard 3x3 sepia matrix transformation via cv2.transform.
"""

import cv2
import numpy as np

# Standard Sepia transformation matrix for BGR format
# Output B = 0.131*R + 0.534*G + 0.272*B
# Output G = 0.168*R + 0.686*G + 0.349*B
# Output R = 0.189*R + 0.769*G + 0.393*B
_SEPIA_MATRIX = np.array([
    [0.272, 0.534, 0.131],  # Blue channel coefficients
    [0.349, 0.686, 0.168],  # Green channel coefficients
    [0.393, 0.769, 0.189],  # Red channel coefficients
], dtype=np.float32)


def apply_sepia_classic(frame: np.ndarray) -> np.ndarray:
    """Apply Sepia Classic warm tone transformation to frame."""
    return cv2.transform(frame, _SEPIA_MATRIX)
