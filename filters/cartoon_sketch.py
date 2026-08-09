"""
cartoon_sketch.py - Filter 8: Cartoon Sketch (Flattened color + bold outlines).

Technique: Bilateral filter color quantization + adaptive threshold edge mask.
"""

import cv2
import numpy as np


def apply_cartoon_sketch(frame: np.ndarray) -> np.ndarray:
    """Apply Cartoon Sketch color flattening and ink outline filter."""
    # 1. Color flattening using fast bilateral filtering
    color_flat = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)

    # 2. Edge detection: Median blur -> Adaptive thresholding
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_gray = cv2.medianBlur(gray, 7)

    edges = cv2.adaptiveThreshold(
        blurred_gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        blockSize=9, C=2,
    )

    # Convert 1-channel edge mask to 3-channel BGR mask
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # 3. Combine flattened colors with ink outlines using bitwise AND
    return cv2.bitwise_and(color_flat, edges_bgr)
