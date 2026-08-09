"""
test_web_overlay.py - Unit tests for Web Overlay pattern rendering and compositing.
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from effects.web_overlay import create_web_pattern_canvas, render_web_overlay


class FakeLandmark:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class TestWebOverlay(unittest.TestCase):
    def test_create_pattern_canvas(self):
        canvas = create_web_pattern_canvas(200, 120)
        self.assertEqual(canvas.shape, (120, 200, 4))
        self.assertEqual(canvas.dtype, np.uint8)

    def test_render_web_overlay_single_hand(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        landmarks = [FakeLandmark(0.5, 0.5) for _ in range(21)]
        # Make wrist (0) and middle mcp (9) distinct for orientation
        landmarks[0] = FakeLandmark(0.5, 0.8)
        landmarks[9] = FakeLandmark(0.5, 0.4)

        result = render_web_overlay(frame.copy(), [landmarks], [True])
        self.assertEqual(result.shape, (720, 1280, 3))
        # Overlay should have modified non-zero pixels
        self.assertTrue(np.any(result > 0))

    def test_render_web_overlay_two_hands(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        hand1 = [FakeLandmark(0.3, 0.5) for _ in range(21)]
        hand2 = [FakeLandmark(0.7, 0.5) for _ in range(21)]

        result = render_web_overlay(frame.copy(), [hand1, hand2], [True, True])
        self.assertEqual(result.shape, (720, 1280, 3))
        self.assertTrue(np.any(result > 0))


if __name__ == "__main__":
    unittest.main()
