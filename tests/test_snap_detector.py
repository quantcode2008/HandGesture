"""
test_snap_detector.py - Unit tests for SnapFrame temporal snap detector.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gestures.snap_detector import SnapDetector, SnapEvent
import config


class FakeLandmark:
    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z


def _make_landmarks_with_thumb_middle_distance(norm_dist: float):
    """
    Construct landmarks where dist(thumb_tip[4], middle_tip[12]) / dist(wrist[0], middle_mcp[9]) = norm_dist.
    """
    wrist = FakeLandmark(0.5, 0.8)
    middle_mcp = FakeLandmark(0.5, 0.5)  # distance = 0.3

    lm = [FakeLandmark(0.5, 0.5) for _ in range(21)]
    lm[0] = wrist
    lm[9] = middle_mcp

    scale = 0.3
    target_dist = norm_dist * scale

    lm[4] = FakeLandmark(0.5, 0.5)
    lm[12] = FakeLandmark(0.5 + target_dist, 0.5)
    return lm


class TestSnapDetector(unittest.TestCase):
    def setUp(self):
        self.detector = SnapDetector()

    def test_fast_contact_release_fires_snap(self):
        """Fast contact -> release sequence SHOULD fire a SnapEvent."""
        t = 0.0
        # Frame 1-3: Contact phase (dist = 0.10 < SNAP_CONTACT_THRESHOLD)
        for _ in range(3):
            lm = _make_landmarks_with_thumb_middle_distance(0.10)
            evt = self.detector.update("Right", lm, t)
            self.assertIsNone(evt)
            t += 0.033

        # Frame 4: Rapid Release phase (dist = 0.50 > SNAP_RELEASE_THRESHOLD)
        lm = _make_landmarks_with_thumb_middle_distance(0.50)
        evt = self.detector.update("Right", lm, t)
        self.assertIsNotNone(evt)
        self.assertEqual(evt.hand_label, "Right")

    def test_slow_gradual_separation_rejected(self):
        """Slow gradual separation SHOULD NOT fire due to low velocity."""
        t = 0.0
        # Contact
        lm = _make_landmarks_with_thumb_middle_distance(0.10)
        self.detector.update("Right", lm, t)
        t += 0.033

        # Very slow release over 3 seconds
        for step in range(1, 10):
            d = 0.10 + step * 0.05
            lm = _make_landmarks_with_thumb_middle_distance(d)
            evt = self.detector.update("Right", lm, t)
            t += 0.3  # slow time delta
            self.assertIsNone(evt)

    def test_noisy_oscillation_rejected(self):
        """Noisy distance oscillation around 0.3 SHOULD NOT fire."""
        t = 0.0
        for _ in range(10):
            lm = _make_landmarks_with_thumb_middle_distance(0.30)
            evt = self.detector.update("Right", lm, t)
            self.assertIsNone(evt)
            t += 0.033


if __name__ == "__main__":
    unittest.main()
