"""
test_filmstrip.py - Unit tests for Filmstrip thumbnail strip (Stretch Goal §19).
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui.filmstrip import FilmstripManager
import config


class TestFilmstripManager(unittest.TestCase):
    def setUp(self):
        self.manager = FilmstripManager()
        self.frame = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)

    def test_render_filmstrip(self):
        output = self.manager.render(self.frame.copy(), active_index=0)
        self.assertEqual(output.shape, (720, 1280, 3))
        self.assertEqual(output.dtype, np.uint8)
        self.assertTrue(len(self.manager.cached_thumbnails) == 10)

    def test_render_disabled(self):
        config.SHOW_FILMSTRIP = False
        output = self.manager.render(self.frame.copy(), active_index=0)
        self.assertTrue(np.array_equal(output, self.frame))
        config.SHOW_FILMSTRIP = True


if __name__ == "__main__":
    unittest.main()
