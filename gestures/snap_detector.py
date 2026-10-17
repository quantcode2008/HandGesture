"""
snap_detector.py - Fast 2-phase temporal finger-snap motion detector (<100ms latency).

Detects a finger snap per hand using a robust 2-phase pattern:
1. "Pinch": Thumb-tip (landmark 4) and middle-fingertip (landmark 12) touch below
   SNAP_PINCH_THRESHOLD, normalized by hand size (wrist 0 to middle MCP 9).
   Must be held for at least SNAP_MIN_PINCH_FRAMES (>=2 frames) to reject single-frame jitter.
2. "Release": The normalized distance rapidly increases above SNAP_RELEASE_THRESHOLD.
   Triggers on release using velocity (normalized distance change per frame) over a short
   rolling window (~6-8 frames).

Features:
- Light EMA landmark filtering to dampen jitter without adding perceptible lag.
- Per-hand cooldown (~400ms) preventing duplicate triggers.
- Invariant to camera distance via hand-scale normalization.
- Rich debug telemetry for real-time HUD visualization.
"""

from collections import deque
from typing import Any, Dict, List, Optional
import config
from utils.geometry import euclidean

# Landmark indices
WRIST = 0
THUMB_TIP = 4
MIDDLE_MCP = 9
MIDDLE_TIP = 12


class SnapEvent:
    """Represents a confirmed finger snap event."""

    def __init__(self, hand_label: str, timestamp: float):
        self.hand_label = hand_label  # "Left" or "Right"
        self.timestamp = timestamp

    def __repr__(self):
        return f"SnapEvent(hand={self.hand_label}, t={self.timestamp:.3f})"


def compute_normalized_thumb_middle_dist(landmarks: List[Any]) -> float:
    """
    Compute thumb-tip (4) to middle-fingertip (12) distance, normalized by
    wrist (0) to middle-MCP (9) distance.
    This normalization makes snap detection invariant to hand distance from camera.
    """
    thumb_tip = landmarks[THUMB_TIP]
    middle_tip = landmarks[MIDDLE_TIP]
    wrist = landmarks[WRIST]
    middle_mcp = landmarks[MIDDLE_MCP]

    d_thumb_middle = euclidean(thumb_tip, middle_tip)
    d_hand_scale = euclidean(wrist, middle_mcp)

    if d_hand_scale < 1e-6:
        return 1.0

    return d_thumb_middle / d_hand_scale


class SmoothedLandmark:
    """Simple landmark object with x, y, z coordinates compatible with geometry functions."""
    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"SmoothedLandmark({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"


class LandmarkSmoother:
    """Lightweight EMA filter smoothing hand landmarks with low latency (<1 frame)."""

    def __init__(self, alpha: float = getattr(config, "LANDMARK_EMA_ALPHA", 0.75)):
        self.alpha = alpha
        self.prev_landmarks: Dict[str, List[SmoothedLandmark]] = {}

    def smooth(self, hand_label: str, landmarks: List[Any]) -> List[Any]:
        if not landmarks or len(landmarks) < 21:
            return landmarks

        alpha = getattr(config, "LANDMARK_EMA_ALPHA", self.alpha)
        one_minus_alpha = 1.0 - alpha

        if hand_label not in self.prev_landmarks or len(self.prev_landmarks[hand_label]) != len(landmarks):
            smoothed = [
                SmoothedLandmark(float(lm.x), float(lm.y), float(getattr(lm, "z", 0.0)))
                for lm in landmarks
            ]
            self.prev_landmarks[hand_label] = smoothed
            return smoothed

        prev = self.prev_landmarks[hand_label]
        smoothed = []
        for curr_lm, prev_lm in zip(landmarks, prev):
            sx = alpha * float(curr_lm.x) + one_minus_alpha * prev_lm.x
            sy = alpha * float(curr_lm.y) + one_minus_alpha * prev_lm.y
            sz = alpha * float(getattr(curr_lm, "z", 0.0)) + one_minus_alpha * prev_lm.z
            smoothed.append(SmoothedLandmark(sx, sy, sz))

        self.prev_landmarks[hand_label] = smoothed
        return smoothed

    def reset_hand(self, hand_label: str):
        if hand_label in self.prev_landmarks:
            del self.prev_landmarks[hand_label]

    def reset(self):
        self.prev_landmarks.clear()


class HandSnapTracker:
    """Tracks the 2-phase snap state machine for an individual hand."""

    def __init__(self, hand_label: str):
        self.hand_label = hand_label
        self.pinch_frames: int = 0
        self.is_pinched: bool = False
        self.min_pinch_dist: float = 1.0
        self.frames_since_pinch: int = 0
        self.cooldown_until: float = -100.0
        self.last_snap_time: float = -100.0
        self.last_velocity: float = 0.0
        self.current_dist: float = 1.0
        self.state: str = "IDLE"
        self.history: deque = deque(maxlen=getattr(config, "SNAP_RELEASE_WINDOW_FRAMES", 7))

    def reset(self):
        self.pinch_frames = 0
        self.is_pinched = False
        self.min_pinch_dist = 1.0
        self.frames_since_pinch = 0
        self.last_velocity = 0.0
        self.state = "IDLE"
        self.history.clear()


class SnapDetector:
    """Manages 2-phase snap detection with jitter rejection, EMA smoothing, and telemetry."""

    def __init__(self):
        self.smoother = LandmarkSmoother(alpha=getattr(config, "LANDMARK_EMA_ALPHA", 0.75))
        self.trackers: Dict[str, HandSnapTracker] = {
            "Left": HandSnapTracker("Left"),
            "Right": HandSnapTracker("Right"),
        }

    def _get_tracker(self, hand_label: str) -> HandSnapTracker:
        if hand_label not in self.trackers:
            self.trackers[hand_label] = HandSnapTracker(hand_label)
        return self.trackers[hand_label]

    def update(self, hand_label: str, landmarks: List[Any], timestamp_sec: float) -> Optional[SnapEvent]:
        """
        Process a new frame's landmarks for a hand and return a SnapEvent if a snap occurred.

        Parameters
        ----------
        hand_label : str
            "Left" or "Right"
        landmarks : list
            The 21 MediaPipe hand landmarks.
        timestamp_sec : float
            Current frame timestamp in seconds.

        Returns
        -------
        Optional[SnapEvent]
            SnapEvent if snap detected, else None.
        """
        # 1. Apply light EMA landmark smoothing
        smoothed = self.smoother.smooth(hand_label, landmarks)

        # 2. Compute normalized distance between thumb tip and middle tip
        norm_dist = compute_normalized_thumb_middle_dist(smoothed)

        tracker = self._get_tracker(hand_label)
        tracker.current_dist = norm_dist
        tracker.history.append((timestamp_sec, norm_dist))

        # 3. Check per-hand cooldown
        cooldown_sec = getattr(config, "SNAP_COOLDOWN_MS", 400) / 1000.0
        if timestamp_sec < tracker.cooldown_until:
            tracker.state = "COOLDOWN"
            tracker.last_velocity = 0.0
            return None

        # 4. Phase 1: Pinch detection
        pinch_thresh = getattr(config, "SNAP_PINCH_THRESHOLD", 0.22)
        min_pinch_frames = getattr(config, "SNAP_MIN_PINCH_FRAMES", 2)

        if norm_dist <= pinch_thresh:
            tracker.pinch_frames += 1
            if tracker.pinch_frames >= min_pinch_frames:
                tracker.is_pinched = True
                tracker.min_pinch_dist = min(tracker.min_pinch_dist, norm_dist)
                tracker.state = "PINCH_CONFIRMED"
            else:
                tracker.state = "PINCHING"
            tracker.frames_since_pinch = 0
            tracker.last_velocity = 0.0
            return None

        # 5. Phase 2: Release detection
        if tracker.is_pinched:
            tracker.frames_since_pinch += 1
            tracker.state = "RELEASE"

            release_window = getattr(config, "SNAP_RELEASE_WINDOW_FRAMES", 7)
            if tracker.frames_since_pinch <= release_window:
                # Velocity = normalized distance change per frame since pinch contact
                delta_dist = norm_dist - tracker.min_pinch_dist
                velocity = delta_dist / max(1, tracker.frames_since_pinch)
                tracker.last_velocity = velocity

                release_thresh = getattr(config, "SNAP_RELEASE_THRESHOLD", 0.38)
                velocity_thresh = getattr(config, "SNAP_VELOCITY_THRESHOLD", 0.045)

                if norm_dist >= release_thresh and velocity >= velocity_thresh:
                    # SNAP DETECTED! Trigger on release
                    tracker.cooldown_until = timestamp_sec + cooldown_sec
                    tracker.last_snap_time = timestamp_sec
                    tracker.state = "COOLDOWN"

                    # Reset tracking state for next gesture
                    tracker.is_pinched = False
                    tracker.pinch_frames = 0
                    tracker.frames_since_pinch = 0
                    tracker.min_pinch_dist = 1.0
                    tracker.history.clear()

                    return SnapEvent(hand_label, timestamp_sec)

                return None
            else:
                # Release window timed out without fast explosive separation (e.g. slow movement)
                tracker.reset()
                return None
        else:
            # Idle (or single-frame jitter that never achieved >= 2 frames pinch)
            tracker.pinch_frames = 0
            tracker.is_pinched = False
            tracker.frames_since_pinch = 0
            tracker.min_pinch_dist = 1.0
            tracker.state = "IDLE"
            tracker.last_velocity = 0.0
            return None

    def reset_hand(self, hand_label: str):
        """Clear tracking buffer and smoothing history when a hand leaves the camera frame."""
        if hand_label in self.trackers:
            self.trackers[hand_label].reset()
        self.smoother.reset_hand(hand_label)

    def get_hand_debug_info(self, hand_label: str) -> Dict[str, Any]:
        """Return snap detection telemetry for a specific hand."""
        tracker = self._get_tracker(hand_label)
        return {
            "dist": tracker.current_dist,
            "velocity": tracker.last_velocity,
            "state": tracker.state,
            "pinch_frames": tracker.pinch_frames,
            "is_pinched": tracker.is_pinched,
            "last_snap_time": tracker.last_snap_time,
        }

    def get_debug_info(self) -> Dict[str, Dict[str, Any]]:
        """Return snap detection telemetry for all tracked hands."""
        return {
            hand_label: self.get_hand_debug_info(hand_label)
            for hand_label in ("Right", "Left")
        }
