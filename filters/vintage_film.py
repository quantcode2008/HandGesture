"""
vintage_film.py - Filter 2: Vintage Film (Warm, faded, slightly desaturated).

Technique: 3x3 warm color matrix -> desaturate ~15% in HSV -> radial vignette.
"""

import cv2
import numpy as np

# 3x3 warm color matrix for BGR transformation (boost R, slightly cut B)
_VINTAGE_MATRIX = np.array([
    [0.85, 0.05, 0.00],  # Blue channel
    [0.00, 0.95, 0.05],  # Green channel
    [0.00, 0.10, 1.05],  # Red channel
], dtype=np.float32)

# Cached radial vignette mask
_VIGNETTE_MASK = None
_CACHED_SHAPE = None


def _get_vignette(h: int, w: int) -> np.ndarray:
    global _VIGNETTE_MASK, _CACHED_SHAPE
    if _CACHED_SHAPE != (h, w):
        kernel_x = cv2.getGaussianKernel(w, w * 0.5)
        kernel_y = cv2.getGaussianKernel(h, h * 0.5)
        kernel = kernel_y * kernel_x.T
        vignette = kernel / kernel.max()
        # Scale to range [0.55, 1.0] for soft edge darkening
        _VIGNETTE_MASK = (0.55 + 0.45 * vignette)[:, :, np.newaxis].astype(np.float32)
        _CACHED_SHAPE = (h, w)
    return _VIGNETTE_MASK


def apply_vintage_film(frame: np.ndarray) -> np.ndarray:
    """Apply Vintage Film filter with warm tones, desaturation, and vignette."""
    # Warm color matrix
    transformed = cv2.transform(frame, _VINTAGE_MATRIX)

    # Desaturate ~15% in HSV
    hsv = cv2.cvtColor(transformed, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= 0.85  # Cut saturation
    desaturated = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Multiply vignette mask
    h, w, _ = frame.shape
    vignette = _get_vignette(h, w)
    output = desaturated.astype(np.float32) * vignette

    return np.clip(output, 0, 255).astype(np.uint8)
