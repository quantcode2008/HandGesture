"""
test_glitch.py - Unit tests for GlitchEffect channel-shift and intensity envelope.
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from effects.glitch import GlitchEffect


class FakeLandmark:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class TestGlitchEffect(unittest.TestCase):
    def test_glitch_intensity_envelope(self):
        glitch = GlitchEffect()
        self.assertEqual(glitch.intensity, 0.0)

        # Trigger active
        glitch.update_intensity(is_active=True)
        self.assertEqual(glitch.intensity, 1.0)

        # Active decay
        glitch.update_intensity(is_active=True)
        self.assertLess(glitch.intensity, 1.0)

        # Inactive reset
        glitch.update_intensity(is_active=False)
        self.assertEqual(glitch.intensity, 0.0)

    def test_apply_glitch_effect(self):
        glitch = GlitchEffect()
        frame = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)
        orig = frame.copy()

        landmarks = [FakeLandmark(0.5, 0.5) for _ in range(21)]
        landmarks[0] = FakeLandmark(0.5, 0.8)
        landmarks[9] = FakeLandmark(0.5, 0.4)

        result = glitch.apply(frame.copy(), [landmarks], [True])
        self.assertEqual(result.shape, (720, 1280, 3))
        # Glitch effect should alter pixels in the active region
        self.assertFalse(np.array_equal(result, orig))


if __name__ == "__main__":
    unittest.main()
