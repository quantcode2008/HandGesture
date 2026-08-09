"""
test_sound.py - Unit tests for SoundPlayer and WAV synthesizer.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from effects.sound import generate_thwip_wav, SoundPlayer


class TestSoundEffect(unittest.TestCase):
    def test_generate_thwip_wav(self):
        test_wav = os.path.join("models", "test_thwip.wav")
        res = generate_thwip_wav(test_wav, duration_sec=0.05)
        self.assertTrue(os.path.exists(res))
        self.assertGreater(os.path.getsize(res), 100)
        if os.path.exists(test_wav):
            os.remove(test_wav)

    def test_sound_player_trigger(self):
        player = SoundPlayer()
        # Should execute without throwing any exception
        player.play_thwip()


if __name__ == "__main__":
    unittest.main()
