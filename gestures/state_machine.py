"""
state_machine.py - Per-hand gesture state machine (PRD Section 10.5).

States:  IDLE -> ARMING -> ACTIVE -> COOLDOWN -> IDLE

- ARMING:   gesture detected; hold counter increments each frame.
            Transitions to ACTIVE after GESTURE_HOLD_FRAMES consecutive
            positive frames.  Drops back to IDLE if gesture is lost.
- ACTIVE:   effect should render.  Tolerates up to MISS_TOLERANCE_FRAMES
            consecutive negative frames before transitioning to COOLDOWN.
- COOLDOWN: forced pause (COOLDOWN_MS) before the gesture can re-arm,
            preventing strobe/flicker at the detection boundary.
"""

import time
from enum import Enum, auto

import config


class State(Enum):
    IDLE = auto()
    ARMING = auto()
    ACTIVE = auto()
    COOLDOWN = auto()


class GestureStateMachine:
    """Debounced gesture state machine for a single hand."""

    def __init__(self, hand_label: str = ""):
        self.state = State.IDLE
        self.hand_label = hand_label  # e.g. "Left" or "Right", for logging

        # Counters
        self._arm_count = 0       # consecutive frames with gesture detected (ARMING)
        self._miss_count = 0      # consecutive frames without gesture (ACTIVE -> COOLDOWN)
        self._cooldown_start = 0  # time.monotonic() when cooldown began

    @property
    def is_active(self) -> bool:
        """True when the effect should be rendering."""
        return self.state == State.ACTIVE

    def update(self, gesture_detected: bool) -> State:
        """Advance the state machine by one frame.

        Parameters
        ----------
        gesture_detected : bool
            Whether the web-shooter pose was classified this frame.

        Returns
        -------
        State
            The new state after this frame's transition.
        """
        prev = self.state

        if self.state == State.IDLE:
            if gesture_detected:
                self.state = State.ARMING
                self._arm_count = 1

        elif self.state == State.ARMING:
            if gesture_detected:
                self._arm_count += 1
                if self._arm_count >= config.GESTURE_HOLD_FRAMES:
                    self.state = State.ACTIVE
                    self._miss_count = 0
            else:
                # Gesture lost during arming - reset
                self.state = State.IDLE
                self._arm_count = 0

        elif self.state == State.ACTIVE:
            if gesture_detected:
                self._miss_count = 0
            else:
                self._miss_count += 1
                if self._miss_count >= config.MISS_TOLERANCE_FRAMES:
                    self.state = State.COOLDOWN
                    self._cooldown_start = time.monotonic()

        elif self.state == State.COOLDOWN:
            elapsed_ms = (time.monotonic() - self._cooldown_start) * 1000
            if elapsed_ms >= config.COOLDOWN_MS:
                self.state = State.IDLE
                self._arm_count = 0
                self._miss_count = 0

        # Log transitions
        if self.state != prev:
            label = f"[{self.hand_label}] " if self.hand_label else ""
            print(f"  {label}{prev.name} -> {self.state.name}")

        return self.state

    def reset(self):
        """Force-reset to IDLE (e.g. when a hand disappears entirely)."""
        if self.state != State.IDLE:
            label = f"[{self.hand_label}] " if self.hand_label else ""
            print(f"  {label}{self.state.name} -> IDLE (hand lost)")
        self.state = State.IDLE
        self._arm_count = 0
        self._miss_count = 0
