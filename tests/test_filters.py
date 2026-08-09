"""
test_filters.py - Unit tests for SnapFrame filter engine (PRD Section 17).
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from filters.registry import FILTER_REGISTRY


class TestFilters(unittest.TestCase):
    def setUp(self):
        self.h, self.w = 480, 640
        self.black_frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.white_frame = np.full((self.h, self.w, 3), 255, dtype=np.uint8)
        self.random_frame = np.random.randint(0, 256, (self.h, self.w, 3), dtype=np.uint8)

    def test_registry_count(self):
        """Verify registry contains exactly 10 filters."""
        self.assertEqual(len(FILTER_REGISTRY), 10)

    def test_filter_shapes_and_dtypes(self):
        """Verify all 10 filters return exact same shape and uint8 dtype."""
        for name, func in FILTER_REGISTRY:
            with self.subTest(filter_name=name):
                output = func(self.random_frame.copy())
                self.assertEqual(output.shape, (self.h, self.w, 3), f"{name} shape mismatch")
                self.assertEqual(output.dtype, np.uint8, f"{name} dtype mismatch")

    def test_filter_edge_case_inputs(self):
        """Verify all 10 filters handle all-black and all-white input frames without crashing."""
        for name, func in FILTER_REGISTRY:
            with self.subTest(filter_name=name):
                out_black = func(self.black_frame.copy())
                out_white = func(self.white_frame.copy())

                self.assertEqual(out_black.shape, (self.h, self.w, 3))
                self.assertEqual(out_white.shape, (self.h, self.w, 3))
                self.assertEqual(out_black.dtype, np.uint8)
                self.assertEqual(out_white.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
