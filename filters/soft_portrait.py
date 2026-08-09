"""
soft_portrait.py - Filter 7: Soft Portrait (Smooth skin & gentle glow).

Technique: Bilateral edge-preserving smoothing blended ~60/40 with original + warmth lift.
"""

import cv2
import numpy as np


def apply_soft_portrait(frame: np.ndarray) -> np.ndarray:
    """Apply Soft Portrait skin smoothing and gentle glow filter."""
    # Fast bilateral filter for skin smoothing
    smooth = cv2.bilateralFilter(frame, d=7, sigmaColor=45, sigmaSpace=45)

    # Blend 60% smooth + 40% original to maintain sharp details while softening skin
    blended = cv2.addWeighted(smooth, 0.60, frame, 0.40, 0)

    # Slight warmth and brightness lift (+4 to R, +2 to G)
    lift_matrix = np.array([
        [1.00, 0.00, 0.00],  # B
        [0.00, 1.02, 0.00],  # G
        [0.00, 0.00, 1.05],  # R
    ], dtype=np.float32)

    return cv2.transform(blended, lift_matrix)
