"""
smoothing.py - Rolling-buffer and exponential smoothing helpers.

Used to smooth landmark coordinates, angles, and scale factors.
"""

from typing import List, Optional, Tuple, Union
import numpy as np


class ExponentialSmoothing:
    """Exponential moving average (EMA) filter for scalar or 2D/3D points."""

    def __init__(self, alpha: float = 0.5):
        """
        Parameters
        ----------
        alpha : float
            Smoothing factor between 0.0 (max smoothing / lag) and 1.0 (no smoothing).
        """
        self.alpha = float(alpha)
        self.value: Optional[np.ndarray] = None

    def update(self, val: Union[float, Tuple[float, ...], np.ndarray]) -> np.ndarray:
        """Update filter with a new observation and return smoothed value."""
        arr = np.array(val, dtype=np.float64)
        if self.value is None:
            self.value = arr
        else:
            self.value = self.alpha * arr + (1.0 - self.alpha) * self.value
        return self.value

    def reset(self) -> None:
        """Reset internal filter state."""
        self.value = None


class RollingBuffer:
    """Fixed-size rolling buffer for computing moving averages."""

    def __init__(self, window_size: int = 5):
        self.window_size = int(window_size)
        self.buffer: List[np.ndarray] = []

    def add(self, val: Union[float, Tuple[float, ...], np.ndarray]) -> np.ndarray:
        """Add a value to the buffer and return the current mean."""
        arr = np.array(val, dtype=np.float64)
        self.buffer.append(arr)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        return np.mean(self.buffer, axis=0)

    def reset(self) -> None:
        """Clear buffer."""
        self.buffer.clear()
