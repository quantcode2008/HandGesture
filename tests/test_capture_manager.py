"""
test_capture_manager.py - Unit tests for CaptureManager (PRD Section 12).
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from capture.capture_manager import CaptureManager, _slugify_filter_name


class TestCaptureManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = os.path.join("tests", "temp_captures")
        self.manager = CaptureManager(capture_dir=self.temp_dir)
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def tearDown(self):
        # Cleanup created files
        if os.path.exists(self.temp_dir):
            for fname in os.listdir(self.temp_dir):
                os.remove(os.path.join(self.temp_dir, fname))
            os.rmdir(self.temp_dir)

    def test_slugify_filter_name(self):
        self.assertEqual(_slugify_filter_name("1. Noir"), "noir")
        self.assertEqual(_slugify_filter_name("2. Vintage Film"), "vintage_film")
        self.assertEqual(_slugify_filter_name("5. Cyberpunk Duotone"), "cyberpunk_duotone")

    def test_save_capture(self):
        saved_path = self.manager.save_capture(self.frame, "1. Noir", now=1.0)
        self.assertIsNotNone(saved_path)
        self.assertTrue(os.path.exists(saved_path))
        self.assertTrue(os.path.basename(saved_path).startswith("capture_noir_"))

    def test_render_feedback(self):
        output = self.manager.render_feedback(self.frame.copy(), now=1.0)
        self.assertEqual(output.shape, (480, 640, 3))
        self.assertEqual(output.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
