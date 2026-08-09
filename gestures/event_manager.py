"""
event_manager.py - Gesture Event Manager.

Debounces and dispatches gesture events (Snap & Fist Capture) per PRD Section 10.4 & 10.5:
- Map snap events to ADVANCE / REVERSE filter commands with INVERT_HANDEDNESS calibration support.
- Edge-triggered Fist Capture state machine preventing multi-capture bursts.
"""

from enum import Enum, auto
import time
from typing import Any, Dict, List, Optional, Tuple

import config
from gestures.fist_detector import is_fist
from gestures.snap_detector import SnapDetector, SnapEvent


class FilterCommand(Enum):
    NONE = auto()
    ADVANCE = auto()   # Next filter (+1)
    REVERSE = auto()   # Previous filter (-1)


class FistState(Enum):
    IDLE = auto()
    ARMING = auto()
    CAPTURE = auto()
    COOLDOWN = auto()


class GestureEventManager:
    """Central event manager processing frame hand landmarks into high-level filter/capture events."""

    def __init__(self):
        self.snap_detector = SnapDetector()

        # Edge-triggered fist capture state
        self.fist_state = FistState.IDLE
        self.fist_arm_count = 0
        self.last_capture_time = 0.0

    def process_frame(
        self,
        hand_landmarks_list: List[List[Any]],
        handedness_list: List[Any],
        timestamp_sec: float,
    ) -> Tuple[FilterCommand, bool]:
        """
        Process per-frame hand tracking results and return (filter_command, trigger_capture).

        Parameters
        ----------
        hand_landmarks_list : list
            Landmarks per detected hand.
        handedness_list : list
            Handedness classification per detected hand.
        timestamp_sec : float
            Frame timestamp in seconds.

        Returns
        -------
        Tuple[FilterCommand, bool]
            (filter_command, trigger_capture)
            filter_command: ADVANCE, REVERSE, or NONE
            trigger_capture: True exactly on the single frame a capture should execute
        """
        cmd = FilterCommand.NONE
        trigger_capture = False

        seen_hands = set()
        any_fist = False

        if hand_landmarks_list:
            for idx, landmarks in enumerate(hand_landmarks_list):
                # Resolve handedness label
                if handedness_list and idx < len(handedness_list):
                    hand_label = handedness_list[idx][0].category_name
                else:
                    hand_label = "Right" if idx == 0 else "Left"

                # MediaPipe handedness on mirrored feed: swap Left<->Right for user perspective
                if hand_label == "Left":
                    hand_label = "Right"
                elif hand_label == "Right":
                    hand_label = "Left"

                seen_hands.add(hand_label)

                # 1. Snap Detection
                snap_event = self.snap_detector.update(hand_label, landmarks, timestamp_sec)
                if snap_event:
                    # Apply INVERT_HANDEDNESS calibration (§10.4)
                    is_right = (snap_event.hand_label == "Right")
                    if config.INVERT_HANDEDNESS:
                        is_right = not is_right

                    if is_right:
                        cmd = FilterCommand.ADVANCE
                        print(f"[EVENT] Snap [{snap_event.hand_label}] -> ADVANCE filter")
                    else:
                        cmd = FilterCommand.REVERSE
                        print(f"[EVENT] Snap [{snap_event.hand_label}] -> REVERSE filter")

                # 2. Fist Detection check
                if is_fist(landmarks):
                    any_fist = True

        # Clear snap buffers for hands not seen this frame
        for label in ["Left", "Right"]:
            if label not in seen_hands:
                self.snap_detector.reset_hand(label)

        # 3. Update Fist Capture State Machine (PRD §10.5)
        trigger_capture = self._update_fist_state(any_fist, timestamp_sec)

        return cmd, trigger_capture

    def _update_fist_state(self, is_fist_detected: bool, timestamp_sec: float) -> bool:
        """Update fist capture state machine and return True on capture edge."""
        trigger = False
        cooldown_sec = config.CAPTURE_COOLDOWN_MS / 1000.0

        if self.fist_state == FistState.IDLE:
            if is_fist_detected:
                self.fist_state = FistState.ARMING
                self.fist_arm_count = 1

        elif self.fist_state == FistState.ARMING:
            if is_fist_detected:
                self.fist_arm_count += 1
                if self.fist_arm_count >= config.FIST_HOLD_FRAMES:
                    self.fist_state = FistState.CAPTURE
                    self.last_capture_time = timestamp_sec
                    trigger = True
                    print("[EVENT] Fist confirmed -> CAPTURE FRAME!")
            else:
                self.fist_state = FistState.IDLE
                self.fist_arm_count = 0

        elif self.fist_state == FistState.CAPTURE:
            self.fist_state = FistState.COOLDOWN

        elif self.fist_state == FistState.COOLDOWN:
            elapsed = timestamp_sec - self.last_capture_time
            # Must release fist AND pass cooldown time before returning to IDLE (FR-6)
            if not is_fist_detected and elapsed >= cooldown_sec:
                self.fist_state = FistState.IDLE
                self.fist_arm_count = 0

        return trigger
