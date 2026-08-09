"""
sound.py - Asynchronous synthesized web-shooter sound effect player.

Generates a custom "thwip" web-shoot audio file using Python standard library (wave + struct)
and plays it asynchronously in a background thread on gesture trigger without dropping FPS.
"""

import math
import os
import random
import struct
import sys
import threading
import wave
from typing import Optional

import config

# Windows-native audio support
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


def generate_thwip_wav(filepath: str, duration_sec: float = 0.15, sample_rate: int = 44100) -> str:
    """
    Synthesize a sci-fi web-shooter 'thwip' sound effect and write to WAV file.

    Audio Profile:
    - Initial 15ms noise burst (mechanical trigger snap)
    - Frequency sweep down from 1600 Hz to 240 Hz
    - Exponential amplitude envelope decay
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    num_samples = int(duration_sec * sample_rate)

    f_start = 1600.0
    f_end = 240.0

    samples = []
    phase = 0.0

    for i in range(num_samples):
        t = i / float(sample_rate)
        progress = t / duration_sec

        # Frequency exponential glide
        freq = f_start * math.pow(f_end / f_start, progress)
        phase += 2.0 * math.pi * freq / sample_rate

        # Sine wave tone
        tone = math.sin(phase)

        # Initial noise transient (first 15ms)
        noise = (random.random() * 2.0 - 1.0) if t < 0.015 else 0.0

        # Composite signal
        signal = 0.7 * tone + 0.3 * noise

        # Exponential decay envelope
        envelope = math.exp(-progress * 6.0)
        amplitude = signal * envelope

        # Clamp to 16-bit signed integer range [-32768, 32767]
        val = int(max(-1.0, min(1.0, amplitude)) * 32767.0)
        samples.append(val)

    # Write WAV file
    with wave.open(filepath, "wb") as wav_file:
        wav_file.setnchannels(1)       # Mono
        wav_file.setsampwidth(2)       # 16-bit
        wav_file.setframerate(sample_rate)
        packed_data = struct.pack(f"<{len(samples)}h", *samples)
        wav_file.writeframes(packed_data)

    return filepath


class SoundPlayer:
    """Non-blocking background audio player for gesture trigger events."""

    def __init__(self, wav_file: str = config.SOUND_EFFECT_FILE):
        self.wav_file = wav_file
        self._ensure_sound_file()

    def _ensure_sound_file(self):
        """Generate the sound effect WAV if missing."""
        if not os.path.exists(self.wav_file):
            try:
                generate_thwip_wav(self.wav_file)
            except Exception as e:
                print(f"[WARN] Failed to generate sound effect: {e}")

    def play_thwip(self):
        """Trigger sound effect playback asynchronously in a background thread."""
        if not config.ENABLE_SOUND_EFFECTS:
            return

        def _play_worker():
            try:
                if HAS_WINSOUND and os.path.exists(self.wav_file):
                    winsound.PlaySound(
                        self.wav_file,
                        winsound.SND_FILENAME | winsound.SND_ASYNC,
                    )
            except Exception as e:
                pass  # Graceful audio fallback

        thread = threading.Thread(target=_play_worker, daemon=True)
        thread.start()
