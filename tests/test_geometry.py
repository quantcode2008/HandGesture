"""
test_geometry.py - Unit tests for geometry helpers.
"""

import math
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.geometry import (
    euclidean,
    euclidean_2d,
    midpoint,
    angle_between,
    rotation_matrix_2d,
)


class DummyPoint:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class TestGeometry(unittest.TestCase):
    def test_euclidean(self):
        p1 = (0.0, 0.0, 0.0)
        p2 = (3.0, 4.0, 0.0)
        self.assertAlmostEqual(euclidean(p1, p2), 5.0)

        dp1 = DummyPoint(1.0, 2.0, 3.0)
        dp2 = DummyPoint(4.0, 6.0, 3.0)
        self.assertAlmostEqual(euclidean(dp1, dp2), 5.0)

    def test_euclidean_2d(self):
        p1 = (0.0, 0.0, 10.0)
        p2 = (3.0, 4.0, 99.0)
        self.assertAlmostEqual(euclidean_2d(p1, p2), 5.0)

    def test_midpoint(self):
        p1 = (0.0, 0.0, 0.0)
        p2 = (2.0, 4.0, 6.0)
        m = midpoint(p1, p2)
        self.assertEqual(m, (1.0, 2.0, 3.0))

    def test_angle_between(self):
        p1 = (0.0, 0.0)
        p2 = (1.0, 0.0)
        self.assertAlmostEqual(angle_between(p1, p2), 0.0)

        p3 = (0.0, 1.0)
        self.assertAlmostEqual(angle_between(p1, p3), math.pi / 2.0)

    def test_rotation_matrix_2d(self):
        rot = rotation_matrix_2d(0.0)
        self.assertAlmostEqual(rot[0, 0], 1.0)
        self.assertAlmostEqual(rot[1, 1], 1.0)
        self.assertAlmostEqual(rot[0, 1], 0.0)
        self.assertAlmostEqual(rot[1, 0], 0.0)


if __name__ == "__main__":
    unittest.main()
