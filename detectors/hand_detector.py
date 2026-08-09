"""
hand_detector.py - HandLandmarker wrapper using the MediaPipe Tasks API.

Uses mediapipe.tasks.python.vision.HandLandmarker (not the deprecated
mp.solutions.hands API) as required by PRD Section 8.

Operates in VIDEO running mode so we can pass timestamped frames from a
live webcam feed and get synchronous results back each frame.
"""

import os
import sys

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
    """Wraps MediaPipe Tasks HandLandmarker for synchronous per-frame use."""

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
            running_mode=RunningMode.VIDEO,
            num_hands=config.MAX_NUM_HANDS,
            min_hand_detection_confidence=config.MIN_HAND_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=config.MIN_HAND_PRESENCE_CONFIDENCE,
            min_tracking_confidence=config.MIN_HAND_TRACKING_CONFIDENCE,
        )
        self.landmarker = HandLandmarker.create_from_options(options)

    def detect(self, rgb_frame, timestamp_ms: int) -> HandLandmarkerResult:
        """Run detection on an RGB frame with a monotonically increasing timestamp.

        Parameters
        ----------
        rgb_frame : np.ndarray
            The frame in RGB color order (not BGR).
        timestamp_ms : int
            Monotonically increasing timestamp in milliseconds.

        Returns
        -------
        HandLandmarkerResult
            Contains .hand_landmarks (list of NormalizedLandmarkList),
            .hand_world_landmarks, and .handedness.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        return self.landmarker.detect_for_video(mp_image, timestamp_ms)

    def close(self):
        """Release the underlying landmarker resources."""
        self.landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
