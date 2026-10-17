"""
test_snap_detector.py - Unit tests for SnapFrame fast 2-phase snap detector.
Tests true snap sequences, single-frame jitter rejection, slow release rejection,
cooldown enforcement, and distance normalization invariance.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from gestures.snap_detector import (
    LandmarkSmoother,
    SnapDetector,
    SnapEvent,
    compute_normalized_thumb_middle_dist,
)


class FakeLandmark:
    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z


def _make_landmarks(norm_dist: float, scale: float = 0.3):
    """
    Construct synthetic landmarks where:
    dist(thumb_tip[4], middle_tip[12]) / dist(wrist[0], middle_mcp[9]) == norm_dist
    """
    wrist = FakeLandmark(0.5, 0.5 + scale)
    middle_mcp = FakeLandmark(0.5, 0.5)  # distance = scale

    lm = [FakeLandmark(0.5, 0.5) for _ in range(21)]
    lm[0] = wrist
    lm[9] = middle_mcp

    target_dist = norm_dist * scale
    lm[4] = FakeLandmark(0.5, 0.5)
    lm[12] = FakeLandmark(0.5 + target_dist, 0.5)
    return lm


class TestSnapDetector(unittest.TestCase):
    def setUp(self):
        self.detector = SnapDetector()

    def test_true_snap_triggers(self):
        """Phase 1: Pinch held for >=2 frames -> Phase 2: Explosive release -> FIRES."""
        t = 1.0
        # Idle frame
        evt = self.detector.update("Right", _make_landmarks(0.40), t)
        self.assertIsNone(evt)
        t += 0.033

        # Frame 1: Pinch contact (dist = 0.12 <= SNAP_PINCH_THRESHOLD)
        evt = self.detector.update("Right", _make_landmarks(0.12), t)
        self.assertIsNone(evt)
        t += 0.033

        # Frame 2: Pinch held (dist = 0.12, confirms >= 2 frames)
        evt = self.detector.update("Right", _make_landmarks(0.12), t)
        self.assertIsNone(evt)
        t += 0.033

        # Frame 3: Explosive release (dist = 0.55 > SNAP_RELEASE_THRESHOLD)
        evt = self.detector.update("Right", _make_landmarks(0.55), t)
        self.assertIsNotNone(evt)
        self.assertIsInstance(evt, SnapEvent)
        self.assertEqual(evt.hand_label, "Right")
        self.assertAlmostEqual(evt.timestamp, t, places=2)

    def test_single_frame_jitter_rejected(self):
        """Single-frame jitter into pinch threshold (< 2 frames) MUST NOT fire on release."""
        t = 1.0
        # Idle frame
        self.detector.update("Right", _make_landmarks(0.40), t)
        t += 0.033

        # Jitter: exactly 1 frame of low distance
        evt = self.detector.update("Right", _make_landmarks(0.12), t)
        self.assertIsNone(evt)
        t += 0.033

        # Immediately separates back to open hand
        evt = self.detector.update("Right", _make_landmarks(0.55), t)
        self.assertIsNone(evt, "Single-frame pinch jitter must not fire a snap event")

    def test_slow_pinch_separation_rejected(self):
        """Gradual finger separation with low velocity MUST NOT fire."""
        t = 1.0
        # Pinch held for 3 frames
        for _ in range(3):
            self.detector.update("Right", _make_landmarks(0.12), t)
            t += 0.033

        # Slow separation creeping up by 0.02 each frame
        curr_d = 0.12
        for _ in range(12):
            curr_d += 0.02
            evt = self.detector.update("Right", _make_landmarks(curr_d), t)
            t += 0.033
            self.assertIsNone(evt, f"Slow opening at dist {curr_d:.2f} should not trigger snap")

    def test_cooldown_prevents_double_trigger(self):
        """Cooldown period (~400ms) prevents duplicate snaps on the same hand."""
        t = 1.0
        # 1. Fire first snap
        self.detector.update("Right", _make_landmarks(0.12), t)
        t += 0.033
        self.detector.update("Right", _make_landmarks(0.12), t)
        t += 0.033
        first_evt = self.detector.update("Right", _make_landmarks(0.55), t)
        self.assertIsNotNone(first_evt)
        snap_time = t

        # 2. Attempt immediate second snap at t + 100ms (within 400ms cooldown)
        t = snap_time + 0.100
        self.detector.update("Right", _make_landmarks(0.12), t)
        t += 0.033
        self.detector.update("Right", _make_landmarks(0.12), t)
        t += 0.033
        second_evt = self.detector.update("Right", _make_landmarks(0.55), t)
        self.assertIsNone(second_evt, "Snap during cooldown period must be ignored")

        # 3. Attempt snap after cooldown expires (t > snap_time + 400ms)
        t = snap_time + 0.450
        self.detector.update("Right", _make_landmarks(0.12), t)
        t += 0.033
        self.detector.update("Right", _make_landmarks(0.12), t)
        t += 0.033
        third_evt = self.detector.update("Right", _make_landmarks(0.55), t)
        self.assertIsNotNone(third_evt, "Snap after cooldown expiration must succeed")

    def test_independent_hands(self):
        """Left and Right hands operate with completely independent state and cooldowns."""
        t = 1.0
        # Right hand fires snap
        self.detector.update("Right", _make_landmarks(0.12), t)
        t += 0.033
        self.detector.update("Right", _make_landmarks(0.12), t)
        t += 0.033
        r_evt = self.detector.update("Right", _make_landmarks(0.55), t)
        self.assertIsNotNone(r_evt)
        self.assertEqual(r_evt.hand_label, "Right")

        # Left hand pinches and snaps while Right hand is in cooldown
        t += 0.050
        self.detector.update("Left", _make_landmarks(0.12), t)
        t += 0.033
        self.detector.update("Left", _make_landmarks(0.12), t)
        t += 0.033
        l_evt = self.detector.update("Left", _make_landmarks(0.55), t)
        self.assertIsNotNone(l_evt)
        self.assertEqual(l_evt.hand_label, "Left")

    def test_hand_scale_normalization_invariance(self):
        """Normalized distance is invariant whether hand is close or far from camera."""
        # Close hand (scale = 0.6)
        lm_close = _make_landmarks(0.15, scale=0.6)
        dist_close = compute_normalized_thumb_middle_dist(lm_close)

        # Far hand (scale = 0.15)
        lm_far = _make_landmarks(0.15, scale=0.15)
        dist_far = compute_normalized_thumb_middle_dist(lm_far)

        self.assertAlmostEqual(dist_close, 0.15, places=4)
        self.assertAlmostEqual(dist_far, 0.15, places=4)

    def test_landmark_smoother_dampens_noise(self):
        """LandmarkSmoother reduces single-frame spike while tracking smoothly."""
        smoother = LandmarkSmoother(alpha=0.75)
        lm_base = [FakeLandmark(0.5, 0.5) for _ in range(21)]
        s1 = smoother.smooth("Right", lm_base)
        self.assertAlmostEqual(s1[4].x, 0.5, places=4)

        # Spike on landmark 4 to 0.9
        lm_spike = [FakeLandmark(0.5, 0.5) for _ in range(21)]
        lm_spike[4] = FakeLandmark(0.9, 0.5)
        s2 = smoother.smooth("Right", lm_spike)
        # Expected: 0.75 * 0.9 + 0.25 * 0.5 = 0.80 (< 0.9)
        self.assertAlmostEqual(s2[4].x, 0.80, places=4)

    def test_debug_telemetry(self):
        """get_debug_info provides distance, velocity, and state for HUD."""
        self.detector.update("Right", _make_landmarks(0.35), 1.0)
        info = self.detector.get_debug_info()
        self.assertIn("Right", info)
        self.assertIn("Left", info)
        self.assertEqual(info["Right"]["state"], "IDLE")
        self.assertAlmostEqual(info["Right"]["dist"], 0.35, places=2)


if __name__ == "__main__":
    unittest.main()
