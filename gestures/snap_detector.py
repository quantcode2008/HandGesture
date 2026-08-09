"""
snap_detector.py - Temporal finger-snap motion detector.

Detects a finger snap per hand by tracking thumb-tip to middle-fingertip distance
over time per PRD Section 10.3:
- Contact phase: Thumb tip and middle fingertip touch (normalized distance < SNAP_CONTACT_THRESHOLD).
- Release phase: Thumb and middle finger rapidly separate (> SNAP_RELEASE_THRESHOLD)
  within SNAP_RELEASE_WINDOW_FRAMES.
- Velocity check: Separation speed exceeds SNAP_RELEASE_VELOCITY_THRESHOLD.
"""

from collections import deque
import time
from typing import Any, Dict, List, Optional, Tuple

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
    Compute thumb-tip to middle-fingertip distance, normalized by wrist-to-middle-MCP distance.
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


class SnapDetector:
    """Manages rolling buffers per hand and detects snap motion events."""

    def __init__(self):
        self.buffers: Dict[str, deque] = {
            "Left": deque(maxlen=config.SNAP_WINDOW_FRAMES),
            "Right": deque(maxlen=config.SNAP_WINDOW_FRAMES),
        }
        # Initialize last snap time to negative value so initial frames are not in cooldown
        self.last_snap_time: Dict[str, float] = {"Left": -100.0, "Right": -100.0}

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
        if hand_label not in self.buffers:
            self.buffers[hand_label] = deque(maxlen=config.SNAP_WINDOW_FRAMES)
            self.last_snap_time[hand_label] = -100.0

        # Check per-hand cooldown
        cooldown_sec = config.SNAP_COOLDOWN_MS / 1000.0
        if (timestamp_sec - self.last_snap_time[hand_label]) < cooldown_sec:
            return None

        norm_dist = compute_normalized_thumb_middle_dist(landmarks)
        buf = self.buffers[hand_label]
        buf.append((timestamp_sec, norm_dist))

        if len(buf) < 3:
            return None

        # Look for contact phase in buffer history (excluding the current sample)
        # Find the earliest sample in buffer where distance was below SNAP_CONTACT_THRESHOLD
        contact_idx = -1
        for idx in range(len(buf) - 1):
            if buf[idx][1] <= config.SNAP_CONTACT_THRESHOLD:
                contact_idx = idx
                break

        if contact_idx == -1:
            return None

        contact_time, contact_dist = buf[contact_idx]
        current_time, current_dist = buf[-1]

        frames_since_contact = (len(buf) - 1) - contact_idx

        # 1. Release window check
        if frames_since_contact > config.SNAP_RELEASE_WINDOW_FRAMES:
            return None

        # 2. Release threshold check
        if current_dist < config.SNAP_RELEASE_THRESHOLD:
            return None

        # 3. Separation velocity check
        dt = current_time - contact_time
        if dt < 1e-6:
            return None

        velocity = (current_dist - contact_dist) / dt
        if velocity < config.SNAP_RELEASE_VELOCITY_THRESHOLD:
            return None

        # Snap confirmed! Reset buffer for this hand & update cooldown
        buf.clear()
        self.last_snap_time[hand_label] = current_time
        return SnapEvent(hand_label, current_time)

    def reset_hand(self, hand_label: str):
        """Clear buffer when hand is lost."""
        if hand_label in self.buffers:
            self.buffers[hand_label].clear()
