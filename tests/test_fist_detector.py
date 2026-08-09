"""
test_fist_detector.py - Unit tests for SnapFrame fist detector.
"""

import sys
import os
import unittest
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gestures.fist_detector import is_fist, is_finger_extended


class FakeLandmark:
    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z


def _make_hand(fist: bool = False):
    wrist = FakeLandmark(0.5, 0.8)
    mcp_dist = 0.3
    ext_dist = 0.5
    curl_dist = 0.25

    lm = [None] * 21
    lm[0] = wrist

    angles = [150, 120, 100, 80, 60]

    def pt(dist, angle_deg):
        rad = math.radians(angle_deg)
        return FakeLandmark(wrist.x + dist * math.cos(rad), wrist.y + dist * math.sin(rad))

    # Thumb (1-4)
    lm[1] = pt(0.15, 150)
    lm[2] = pt(0.3, 150)
    lm[3] = pt(0.35, 150)
    lm[4] = pt(curl_dist if fist else ext_dist, 150)

    # Index (5-8), Middle (9-12), Ring (13-16), Pinky (17-20)
    indices = [(5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]
    for f_idx, (mcp_i, pip_i, dip_i, tip_i) in enumerate(indices):
        ang = angles[f_idx + 1]
        lm[mcp_i] = pt(mcp_dist, ang)
        lm[pip_i] = pt(mcp_dist + 0.05, ang)
        lm[dip_i] = pt(mcp_dist + 0.10, ang)
        lm[tip_i] = pt(curl_dist if fist else ext_dist, ang)

    return lm


class TestFistDetector(unittest.TestCase):
    def test_closed_fist_detected(self):
        lm = _make_hand(fist=True)
        self.assertTrue(is_fist(lm))

    def test_open_palm_rejected(self):
        lm = _make_hand(fist=False)
        self.assertFalse(is_fist(lm))


if __name__ == "__main__":
    unittest.main()
