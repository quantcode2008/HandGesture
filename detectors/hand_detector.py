"""
hand_detector.py - HandLandmarker wrapper using the MediaPipe Tasks API in LIVE_STREAM mode.

Operates in LIVE_STREAM running mode with async callback processing, allowing the main
render pipeline to run at full camera frame rate without blocking on AI inference.
Tracks up to 2 hands simultaneously per PRD Section 8 and user requirements.
"""

import os
import sys
import threading
from typing import Optional
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarkerResult,
    RunningMode,
)

import config


class HandDetector:
    """Wraps MediaPipe Tasks HandLandmarker for asynchronous LIVE_STREAM use."""

    def __init__(self):
        model_path = config.HAND_LANDMARKER_MODEL
        if not os.path.isfile(model_path):
            print(
                f"[ERROR] Hand-landmarker model not found at:\n"
                f"  {model_path}\n\n"
                f"Download it with:\n"
                f"  curl -o models/hand_landmarker.task "
                f"https://storage.googleapis.com/mediapipe-models/"
                f"hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                file=sys.stderr,
            )
            sys.exit(1)

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.LIVE_STREAM,
            result_callback=self._on_result,
            num_hands=config.MAX_NUM_HANDS,  # 2
            min_hand_detection_confidence=config.MIN_HAND_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=config.MIN_HAND_PRESENCE_CONFIDENCE,
            min_tracking_confidence=config.MIN_HAND_TRACKING_CONFIDENCE,
        )
        self.landmarker = HandLandmarker.create_from_options(options)

        self._lock = threading.Lock()
        self._latest_result: Optional[HandLandmarkerResult] = None
        self._last_sent_timestamp_ms: int = 0
        self._first_result_event = threading.Event()

    def _on_result(
        self,
        result: HandLandmarkerResult,
        output_image: mp.Image,
        timestamp_ms: int,
    ):
        """Asynchronous callback invoked by MediaPipe when inference finishes."""
        with self._lock:
            self._latest_result = result
        self._first_result_event.set()

    def detect(self, rgb_frame, timestamp_ms: int) -> Optional[HandLandmarkerResult]:
        """Submit an RGB frame for asynchronous detection in LIVE_STREAM mode
        and immediately return the most recent available HandLandmarkerResult.

        Parameters
        ----------
        rgb_frame : np.ndarray
            The frame in RGB color order (not BGR).
        timestamp_ms : int
            Monotonically increasing timestamp in milliseconds.

        Returns
        -------
        HandLandmarkerResult or None
            Contains .hand_landmarks, .hand_world_landmarks, and .handedness.
        """
        # Ensure timestamp is strictly monotonically increasing as required by MediaPipe
        if timestamp_ms <= self._last_sent_timestamp_ms:
            timestamp_ms = self._last_sent_timestamp_ms + 1
        self._last_sent_timestamp_ms = timestamp_ms

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        self.landmarker.detect_async(mp_image, timestamp_ms)

        # On the very first frame, give up to 25ms to receive the first result
        if self._latest_result is None:
            self._first_result_event.wait(timeout=0.025)

        with self._lock:
            return self._latest_result

    def close(self):
        """Release the underlying landmarker resources."""
        self.landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
