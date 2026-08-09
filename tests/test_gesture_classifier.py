"""
test_gesture_classifier.py - Unit tests for gesture classification (PRD Section 16).

Uses synthetic landmark fixtures for known poses:
  - Web-shooter (index + pinky extended, middle + ring curled)
  - Open palm (all fingers extended)
  - Fist (all fingers curled)
  - Thumbs-up (only thumb extended)
  - Victory / peace sign (index + middle extended)

Each fixture is a list of 21 simple objects with .x, .y, .z attributes
arranged so the distance-ratio classifier produces the expected result.
"""

import sys
import os
import unittest

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gestures.gesture_classifier import (
    is_finger_extended,
    is_web_shooter_pose,
)


class FakeLandmark:
    """Minimal stand-in for mediapipe NormalizedLandmark."""

    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"LM({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"


# --- Fixture Helpers -------------------------------------------------------
# We place the wrist (landmark 0) at the origin and arrange finger tips
# and MCP joints at controlled distances so the distance-ratio test
# produces deterministic results.
#
# "Extended" finger: tip is far from wrist (ratio > 1.3 x MCP distance)
# "Curled" finger:   tip is close to wrist (ratio < 1.3 x MCP distance)
#
# Landmark layout (indices from PRD Section 10.1):
#   0: WRIST
#   1-4:   THUMB (CMC, MCP, IP, TIP)
#   5-8:   INDEX (MCP, PIP, DIP, TIP)
#   9-12:  MIDDLE (MCP, PIP, DIP, TIP)
#   13-16: RING (MCP, PIP, DIP, TIP)
#   17-20: PINKY (MCP, PIP, DIP, TIP)


def _make_landmarks(
    index_extended=False,
    middle_extended=False,
    ring_extended=False,
    pinky_extended=False,
    thumb_extended=False,
):
    """Build a 21-landmark array with the specified finger states.

    MCP joints are placed at distance 0.3 from wrist.
    Extended tips are at distance 0.5 (ratio ~= 1.67, well above 1.3 threshold).
    Curled tips are at distance 0.25 (ratio ~= 0.83, well below 1.3 threshold).
    """
    wrist = FakeLandmark(0.5, 0.8)  # center-bottom of "image"

    mcp_dist = 0.3
    ext_tip_dist = 0.5   # 0.5 / 0.3 = 1.67 > 1.3 -> extended
    curl_tip_dist = 0.25  # 0.25 / 0.3 = 0.83 < 1.3 -> curled

    # Helper: place a point at a given distance along an angle from wrist
    import math

    def pt(dist, angle_deg):
        rad = math.radians(angle_deg)
        return FakeLandmark(
            wrist.x + dist * math.cos(rad),
            wrist.y + dist * math.sin(rad),
        )

    # Angles (spread fingers across an arc for realism)
    # Thumb: ~150 deg, Index: ~120 deg, Middle: ~100 deg, Ring: ~80 deg, Pinky: ~60 deg
    angles = {
        "thumb": 150,
        "index": 120,
        "middle": 100,
        "ring": 80,
        "pinky": 60,
    }

    def finger_landmarks(name, is_extended):
        """Return [MCP, PIP, DIP, TIP] landmarks for one finger."""
        angle = angles[name]
        mcp = pt(mcp_dist, angle)
        pip = pt(mcp_dist + 0.05, angle)
        dip = pt(mcp_dist + 0.10, angle)
        tip_dist = ext_tip_dist if is_extended else curl_tip_dist
        tip = pt(tip_dist, angle)
        return [mcp, pip, dip, tip]

    # Build the 21 landmarks in order
    lm = [None] * 21
    lm[0] = wrist

    # Thumb (indices 1-4): CMC, MCP, IP, TIP
    thumb_angle = angles["thumb"]
    lm[1] = pt(0.15, thumb_angle)  # CMC
    lm[2] = pt(mcp_dist, thumb_angle)  # MCP
    lm[3] = pt(mcp_dist + 0.05, thumb_angle)  # IP
    thumb_tip_dist = ext_tip_dist if thumb_extended else curl_tip_dist
    lm[4] = pt(thumb_tip_dist, thumb_angle)  # TIP

    # Index (5-8)
    idx_lms = finger_landmarks("index", index_extended)
    lm[5], lm[6], lm[7], lm[8] = idx_lms

    # Middle (9-12)
    mid_lms = finger_landmarks("middle", middle_extended)
    lm[9], lm[10], lm[11], lm[12] = mid_lms

    # Ring (13-16)
    ring_lms = finger_landmarks("ring", ring_extended)
    lm[13], lm[14], lm[15], lm[16] = ring_lms

    # Pinky (17-20)
    pinky_lms = finger_landmarks("pinky", pinky_extended)
    lm[17], lm[18], lm[19], lm[20] = pinky_lms

    return lm


# --- Tests ------------------------------------------------------------------


class TestFingerExtension(unittest.TestCase):
    """Test the low-level is_finger_extended function."""

    def test_extended_finger_returns_true(self):
        lm = _make_landmarks(index_extended=True)
        self.assertTrue(is_finger_extended(lm, tip_idx=8, mcp_idx=5))

    def test_curled_finger_returns_false(self):
        lm = _make_landmarks(index_extended=False)
        self.assertFalse(is_finger_extended(lm, tip_idx=8, mcp_idx=5))

    def test_each_finger_independently(self):
        """Each finger's extension state should be independent of others."""
        lm = _make_landmarks(
            index_extended=True,
            middle_extended=False,
            ring_extended=True,
            pinky_extended=False,
        )
        self.assertTrue(is_finger_extended(lm, 8, 5))
        self.assertFalse(is_finger_extended(lm, 12, 9))
        self.assertTrue(is_finger_extended(lm, 16, 13))
        self.assertFalse(is_finger_extended(lm, 20, 17))


class TestWebShooterPose(unittest.TestCase):
    """Test the web-shooter gesture detector against known poses."""

    def test_web_shooter_pose_detected(self):
        """Index + pinky extended, middle + ring curled -> TRUE."""
        lm = _make_landmarks(
            index_extended=True,
            middle_extended=False,
            ring_extended=False,
            pinky_extended=True,
        )
        self.assertTrue(is_web_shooter_pose(lm))

    def test_web_shooter_with_thumb_tucked(self):
        """Same as above but with thumb curled - should still detect."""
        lm = _make_landmarks(
            index_extended=True,
            middle_extended=False,
            ring_extended=False,
            pinky_extended=True,
            thumb_extended=False,
        )
        self.assertTrue(is_web_shooter_pose(lm))

    def test_web_shooter_with_thumb_extended(self):
        """Thumb extended doesn't prevent detection (ILoveYou variant)."""
        lm = _make_landmarks(
            index_extended=True,
            middle_extended=False,
            ring_extended=False,
            pinky_extended=True,
            thumb_extended=True,
        )
        self.assertTrue(is_web_shooter_pose(lm))

    def test_open_palm_rejected(self):
        """All fingers extended -> NOT web-shooter."""
        lm = _make_landmarks(
            index_extended=True,
            middle_extended=True,
            ring_extended=True,
            pinky_extended=True,
            thumb_extended=True,
        )
        self.assertFalse(is_web_shooter_pose(lm))

    def test_fist_rejected(self):
        """All fingers curled -> NOT web-shooter."""
        lm = _make_landmarks(
            index_extended=False,
            middle_extended=False,
            ring_extended=False,
            pinky_extended=False,
            thumb_extended=False,
        )
        self.assertFalse(is_web_shooter_pose(lm))

    def test_thumbs_up_rejected(self):
        """Only thumb extended (all fingers curled) -> NOT web-shooter."""
        lm = _make_landmarks(
            index_extended=False,
            middle_extended=False,
            ring_extended=False,
            pinky_extended=False,
            thumb_extended=True,
        )
        self.assertFalse(is_web_shooter_pose(lm))

    def test_victory_sign_rejected(self):
        """Index + middle extended, rest curled -> NOT web-shooter."""
        lm = _make_landmarks(
            index_extended=True,
            middle_extended=True,
            ring_extended=False,
            pinky_extended=False,
        )
        self.assertFalse(is_web_shooter_pose(lm))

    def test_pointing_rejected(self):
        """Only index extended -> NOT web-shooter (pinky is required)."""
        lm = _make_landmarks(
            index_extended=True,
            middle_extended=False,
            ring_extended=False,
            pinky_extended=False,
        )
        self.assertFalse(is_web_shooter_pose(lm))

    def test_pinky_only_rejected(self):
        """Only pinky extended -> NOT web-shooter (index is required)."""
        lm = _make_landmarks(
            index_extended=False,
            middle_extended=False,
            ring_extended=False,
            pinky_extended=True,
        )
        self.assertFalse(is_web_shooter_pose(lm))

    def test_middle_ring_extended_rejects(self):
        """All four fingers extended (middle+ring not curled) -> NOT web-shooter."""
        lm = _make_landmarks(
            index_extended=True,
            middle_extended=True,
            ring_extended=True,
            pinky_extended=True,
        )
        self.assertFalse(is_web_shooter_pose(lm))


if __name__ == "__main__":
    unittest.main()
