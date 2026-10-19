"""
emotion_detector.py - Real-time facial emotion detection and filter auto-suggestion.

Uses MediaPipe FaceLandmarker (Tasks API) in LIVE_STREAM mode with face blendshapes.
Extracts facial action units and classifies emotions (Happy, Sad, Angry, Surprised,
Fear, Disgust, Neutral) using an extensible classifier interface and rule-based scorer.

Features:
- BaseEmotionClassifier interface: predict(frame, landmarks, blendshapes) -> (label, confidence).
- RuleBasedEmotionClassifier: tunable weighted scoring loaded from config.py.
- EmotionSmoother: temporal smoothing across ~10 frames to stop flickering.
- Asynchronous LIVE_STREAM pipeline with frame skipping (processes every 2nd frame)
  guaranteeing FPS stays well above 20 FPS.
- Top-left UI badge overlay (toggled with 'e').
- Telemetry integration for the SnapFrame Debug HUD.
- Filter auto-suggestions (e.g., Happy -> Golden Hour, Sad -> Noir).
"""

from abc import ABC, abstractmethod
from collections import deque
import os
import sys
import threading
from typing import Any, Dict, List, Optional, Tuple
import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    FaceLandmarkerResult,
    RunningMode,
)
import numpy as np

import config


class BaseEmotionClassifier(ABC):
    """Abstract interface for emotion classification algorithms.

    Allows interchangeable classifiers: rule-based blendshape models,
    trained neural networks (ONNX/Torch), or custom landmark classifiers.
    """

    @abstractmethod
    def predict(
        self,
        frame: Optional[np.ndarray] = None,
        landmarks: Optional[List[Any]] = None,
        blendshapes: Optional[Dict[str, float]] = None,
    ) -> Tuple[str, float]:
        """Predict emotion from frame, face landmarks, and/or blendshape scores.

        Parameters
        ----------
        frame : np.ndarray, optional
            The camera frame in BGR or RGB format.
        landmarks : list, optional
            478 normalized 3D face landmarks from MediaPipe FaceLandmarker.
        blendshapes : dict of str -> float, optional
            Dictionary mapping 52 blendshape category names to scores (0.0 to 1.0).

        Returns
        -------
        Tuple[str, float]
            (emotion_label, confidence_score) where confidence is in [0.0, 1.0].
            Example: ("Happy", 0.88)
        """
        pass


class RuleBasedEmotionClassifier(BaseEmotionClassifier):
    """Weighted rule-based emotion scorer evaluating facial blendshapes.

    All emotion rules, positive/negative blendshape weights, and confidence
    thresholds are loaded from config.py for zero-downtime tuning.
    """

    def __init__(
        self,
        rules: Optional[Dict[str, Any]] = None,
        min_confidence: Optional[float] = None,
        neutral_label: Optional[str] = None,
    ):
        self.rules = rules if rules is not None else getattr(config, "EMOTION_RULES", {})
        self.min_confidence = (
            min_confidence
            if min_confidence is not None
            else getattr(config, "EMOTION_MIN_CONFIDENCE", 0.28)
        )
        self.neutral_label = (
            neutral_label
            if neutral_label is not None
            else getattr(config, "EMOTION_NEUTRAL_LABEL", "Neutral")
        )

    def score_all(self, blendshapes: Dict[str, float]) -> Dict[str, float]:
        """Compute raw scores for each configured emotion given blendshapes."""
        scores: Dict[str, float] = {}
        if not blendshapes:
            return scores

        for emotion, rule in self.rules.items():
            s = 0.0
            for shape_name, weight in rule.get("positive", []):
                s += weight * blendshapes.get(shape_name, 0.0)
            for shape_name, weight in rule.get("negative", []):
                s -= weight * blendshapes.get(shape_name, 0.0)
            # Bound score between 0.0 and 1.0
            scores[emotion] = max(0.0, min(1.0, s))

        return scores

    def predict(
        self,
        frame: Optional[np.ndarray] = None,
        landmarks: Optional[List[Any]] = None,
        blendshapes: Optional[Dict[str, float]] = None,
    ) -> Tuple[str, float]:
        """Classify emotion and return (label, confidence)."""
        if not blendshapes:
            return self.neutral_label, 0.50

        scores = self.score_all(blendshapes)
        if not scores:
            return self.neutral_label, 0.50

        best_emotion, best_score = max(scores.items(), key=lambda kv: kv[1])
        if best_score < self.min_confidence:
            conf = max(0.50, min(1.0, 1.0 - best_score))
            return self.neutral_label, round(conf, 3)

        return best_emotion, round(best_score, 3)


class EmotionSmoother:
    """Temporal smoother over rolling window of predictions to eliminate flicker."""

    def __init__(self, window_size: Optional[int] = None):
        self.window_size = (
            window_size
            if window_size is not None
            else getattr(config, "EMOTION_SMOOTHING_WINDOW", 10)
        )
        self.history: deque = deque(maxlen=self.window_size)

    def add(self, emotion: str, confidence: float) -> Tuple[str, float]:
        """Add sample to window and return smoothed (emotion, confidence)."""
        self.history.append((emotion, confidence))
        return self.get_smoothed()

    def get_smoothed(self) -> Tuple[str, float]:
        """Compute weighted majority vote over historical predictions."""
        if not self.history:
            return getattr(config, "EMOTION_NEUTRAL_LABEL", "Neutral"), 0.50

        tally: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        for emo, conf in self.history:
            tally[emo] = tally.get(emo, 0.0) + conf
            counts[emo] = counts.get(emo, 0) + 1

        best_emo = max(tally.keys(), key=lambda k: tally[k])
        avg_conf = tally[best_emo] / max(1, counts[best_emo])
        return best_emo, round(avg_conf, 3)

    def reset(self):
        """Clear temporal history."""
        self.history.clear()


class EmotionDetector:
    """Asynchronous MediaPipe FaceLandmarker emotion detector with blendshape scoring."""

    def __init__(
        self,
        classifier: Optional[BaseEmotionClassifier] = None,
        model_path: Optional[str] = None,
    ):
        model = model_path or getattr(config, "FACE_LANDMARKER_MODEL", "models/face_landmarker.task")
        if not os.path.isfile(model):
            print(
                f"[ERROR] Face-landmarker model not found at:\n"
                f"  {model}\n\n"
                f"Download it with:\n"
                f"  curl -o models/face_landmarker.task "
                f"https://storage.googleapis.com/mediapipe-models/"
                f"face_landmarker/face_landmarker/float16/1/face_landmarker.task",
                file=sys.stderr,
            )
            sys.exit(1)

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model),
            running_mode=RunningMode.LIVE_STREAM,
            result_callback=self._on_result,
            output_face_blendshapes=True,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)

        self.classifier = classifier or RuleBasedEmotionClassifier()
        self.smoother = EmotionSmoother(
            window_size=getattr(config, "EMOTION_SMOOTHING_WINDOW", 10)
        )

        self.frame_interval = getattr(config, "EMOTION_FRAME_INTERVAL", 2)
        self.frame_count = 0

        self._lock = threading.Lock()
        self._last_sent_timestamp_ms = 0
        self._face_detected = False
        self._latest_blendshapes: Dict[str, float] = {}
        self._raw_emotion = getattr(config, "EMOTION_NEUTRAL_LABEL", "Neutral")
        self._raw_confidence = 0.50
        self._smoothed_emotion = getattr(config, "EMOTION_NEUTRAL_LABEL", "Neutral")
        self._smoothed_confidence = 0.50

        initial_meta = getattr(config, "EMOTION_METADATA", {}).get(
            self._smoothed_emotion, {}
        )
        self._suggested_filter: Optional[str] = initial_meta.get("filter", "Soft Portrait")

    def _on_result(
        self,
        result: FaceLandmarkerResult,
        output_image: mp.Image,
        timestamp_ms: int,
    ):
        """Asynchronous callback when MediaPipe face inference completes."""
        with self._lock:
            if result and result.face_blendshapes and len(result.face_blendshapes) > 0:
                self._face_detected = True
                shapes = {b.category_name: b.score for b in result.face_blendshapes[0]}
                self._latest_blendshapes = shapes

                landmarks = (
                    result.face_landmarks[0]
                    if (result.face_landmarks and len(result.face_landmarks) > 0)
                    else None
                )
                label, conf = self.classifier.predict(None, landmarks, shapes)
                self._raw_emotion = label
                self._raw_confidence = conf

                s_label, s_conf = self.smoother.add(label, conf)
                self._smoothed_emotion = s_label
                self._smoothed_confidence = s_conf

                meta = getattr(config, "EMOTION_METADATA", {}).get(s_label, {})
                self._suggested_filter = meta.get("filter", "Soft Portrait")
            else:
                self._face_detected = False
                s_label, s_conf = self.smoother.add(
                    getattr(config, "EMOTION_NEUTRAL_LABEL", "Neutral"), 0.50
                )
                self._smoothed_emotion = s_label
                self._smoothed_confidence = s_conf

    def detect(self, rgb_frame: np.ndarray, timestamp_ms: int) -> Tuple[str, float, Optional[str]]:
        """Submit frame every Nth interval asynchronously and return current smoothed prediction.

        Parameters
        ----------
        rgb_frame : np.ndarray
            Current camera frame in RGB format.
        timestamp_ms : int
            Monotonically increasing timestamp in milliseconds.

        Returns
        -------
        Tuple[str, float, Optional[str]]
            (smoothed_emotion_label, confidence, suggested_filter_name)
        """
        self.frame_count += 1
        # Process every Nth frame to keep FPS high
        if self.frame_count % self.frame_interval == 0:
            if timestamp_ms <= self._last_sent_timestamp_ms:
                timestamp_ms = self._last_sent_timestamp_ms + 1
            self._last_sent_timestamp_ms = timestamp_ms

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            self.landmarker.detect_async(mp_image, timestamp_ms)

        with self._lock:
            return self._smoothed_emotion, self._smoothed_confidence, self._suggested_filter

    def get_debug_info(self) -> Dict[str, Any]:
        """Return real-time emotion telemetry for the SnapFrame Debug HUD."""
        with self._lock:
            # Extract top 3 dominant blendshapes for HUD inspection
            top_shapes = []
            if self._latest_blendshapes:
                sorted_shapes = sorted(
                    self._latest_blendshapes.items(), key=lambda kv: kv[1], reverse=True
                )
                top_shapes = [f"{k}={v:.2f}" for k, v in sorted_shapes[:3] if v > 0.05]

            return {
                "face_detected": self._face_detected,
                "raw_emotion": self._raw_emotion,
                "raw_conf": self._raw_confidence,
                "emotion": self._smoothed_emotion,
                "confidence": self._smoothed_confidence,
                "suggested_filter": self._suggested_filter,
                "top_blendshapes": ", ".join(top_shapes) if top_shapes else "None",
            }

    def render_badge(
        self,
        frame: np.ndarray,
        emotion_label: str,
        confidence: float,
        suggested_filter: Optional[str] = None,
        filmstrip_visible: bool = True,
    ) -> np.ndarray:
        """Render a modern, semi-transparent pill overlay badge at the top-left."""
        h, w, _ = frame.shape
        meta = getattr(config, "EMOTION_METADATA", {}).get(emotion_label, {})
        display_label = meta.get("label", emotion_label)
        accent_color = meta.get("color", (0, 220, 255))

        with self._lock:
            face_found = self._face_detected

        if not face_found:
            text_line = "Face: Searching..."
            accent_color = (150, 150, 150)
        else:
            pct = int(confidence * 100)
            if suggested_filter:
                text_line = f"Mood: {display_label} {pct}%  |  Suggest: {suggested_filter}"
            else:
                text_line = f"Mood: {display_label} {pct}%"

        # Badge placement
        bx = 15
        by = 100 if filmstrip_visible else 20
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.50
        thick = 1

        (tw, th), _ = cv2.getTextSize(text_line, font, scale, thick)
        pad_x = 12
        pad_y = 7
        bw = tw + pad_x * 2 + 18  # extra space for accent dot
        bh = th + pad_y * 2 + 2

        # Draw semi-transparent dark container
        overlay = frame.copy()
        cv2.rectangle(
            overlay, (bx, by), (bx + bw, by + bh), (15, 15, 25), -1
        )
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

        # Draw subtle border with accent color
        cv2.rectangle(
            frame, (bx, by), (bx + bw, by + bh), accent_color, 1, cv2.LINE_AA
        )

        # Draw glowing status dot
        dot_x = bx + pad_x + 4
        dot_y = by + bh // 2
        cv2.circle(frame, (dot_x, dot_y), 4, accent_color, -1, cv2.LINE_AA)

        # Draw text
        text_x = dot_x + 10
        text_y = by + pad_y + th
        cv2.putText(
            frame,
            text_line,
            (text_x, text_y),
            font,
            scale,
            (255, 255, 255),
            thick,
            cv2.LINE_AA,
        )

        return frame

    def close(self):
        """Release FaceLandmarker resources."""
        self.landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
