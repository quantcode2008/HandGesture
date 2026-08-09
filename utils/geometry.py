"""
geometry.py - Distance, angle, midpoint, and rotation-matrix helpers.

Used by gesture_classifier.py (Phase 2) and effects modules (Phase 3+).
All functions work with objects that have .x, .y, .z attributes
(MediaPipe NormalizedLandmark) or plain (x, y, z) tuples/lists.
"""

import math
from typing import Any, List, Tuple, Union
import numpy as np


PointLike = Union[Any, Tuple[float, float], Tuple[float, float, float]]


def _coords(point: PointLike) -> Tuple[float, float, float]:
    """Extract (x, y, z) from a landmark-like object or a tuple/list."""
    if hasattr(point, "x"):
        return float(point.x), float(point.y), float(getattr(point, "z", 0.0))
    x = float(point[0])
    y = float(point[1])
    z = float(point[2]) if len(point) > 2 else 0.0
    return x, y, z


def euclidean(a: PointLike, b: PointLike) -> float:
    """3D Euclidean distance between two landmarks (or 2D if z is absent)."""
    ax, ay, az = _coords(a)
    bx, by, bz = _coords(b)
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)


def euclidean_2d(a: PointLike, b: PointLike) -> float:
    """2D Euclidean distance (ignores z) between two landmarks."""
    ax, ay, _ = _coords(a)
    bx, by, _ = _coords(b)
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def midpoint(a: PointLike, b: PointLike) -> Tuple[float, float, float]:
    """Return the (x, y, z) midpoint between two landmarks."""
    ax, ay, az = _coords(a)
    bx, by, bz = _coords(b)
    return ((ax + bx) / 2.0, (ay + by) / 2.0, (az + bz) / 2.0)


def angle_between(a: PointLike, b: PointLike) -> float:
    """Return the angle in radians from a to b (atan2-based, 2D)."""
    ax, ay, _ = _coords(a)
    bx, by, _ = _coords(b)
    return math.atan2(by - ay, bx - ax)


def rotation_matrix_2d(angle_rad: float) -> np.ndarray:
    """Return a 2x2 rotation matrix for the given angle (radians)."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def compute_palm_center(landmarks: List[Any]) -> Tuple[float, float]:
    """
    Compute normalized (x, y) palm center as average of landmarks 0, 5, 9, 13, 17
    per PRD Section 11.1.
    """
    indices = [0, 5, 9, 13, 17]
    xs = [landmarks[i].x for i in indices]
    ys = [landmarks[i].y for i in indices]
    return (sum(xs) / len(indices), sum(ys) / len(indices))
