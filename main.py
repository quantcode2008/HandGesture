"""
main.py - WebShooter FX (Complete Filter + Sound Effect Stretch Goal).

Real-time gesture-triggered AR "Web-Shooter" camera filter.
- Mirrored selfie webcam feed.
- MediaPipe Tasks API hand landmark tracking.
- Orientation-invariant rule-based web-shooter gesture classifier.
- Debounced state machine per hand (IDLE -> ARMING -> ACTIVE -> COOLDOWN).
- Synthesized "thwip" audio trigger sound effect (played asynchronously).
- Localized RGB chromatic-aberration glitch pulse effect.
- Cross-hatch web-pattern overlay (single-hand and two-hand tracking modes).
- Configurable Debug HUD (toggle with 'd' key or SHOW_DEBUG_HUD in config.py).

Controls:
  'd' - Toggle Debug HUD (landmarks, FPS, state info)
  'q' or ESC - Exit application
"""

import sys
import time
import cv2

import config
from camera import Camera
from detectors.hand_detector import HandDetector
from effects.glitch import GlitchEffect
from effects.sound import SoundPlayer
from effects.web_overlay import render_web_overlay
from gestures.gesture_classifier import is_web_shooter_pose
from gestures.state_machine import GestureStateMachine, State

# --- MediaPipe hand-skeleton connections (21-point topology) ----------------
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

# State -> color (BGR) for visual feedback in Debug HUD
STATE_COLORS = {
    State.IDLE:     (180, 180, 180),   # Grey
    State.ARMING:   (0, 200, 255),     # Amber/yellow
    State.ACTIVE:   (0, 255, 0),       # Green
    State.COOLDOWN: (255, 100, 100),   # Light blue
}

LANDMARK_RADIUS = 5
CONNECTION_THICKNESS = 2


def draw_landmarks_on_frame(frame, hand_landmarks_list, handedness_list, states):
    """Draw landmark skeleton and state labels on frame for Debug HUD mode."""
    h, w, _ = frame.shape

    for hand_idx, landmarks in enumerate(hand_landmarks_list):
        if hand_idx < len(states):
            color = STATE_COLORS.get(states[hand_idx], (180, 180, 180))
        else:
            color = (180, 180, 180)

        points = []
        for lm in landmarks:
            px = int(lm.x * w)
            py = int(lm.y * h)
            points.append((px, py))

        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], color, CONNECTION_THICKNESS, cv2.LINE_AA)

        for idx, (px, py) in enumerate(points):
            if idx in (4, 8, 12, 16, 20):
                cv2.circle(frame, (px, py), LANDMARK_RADIUS + 2, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, (px, py), LANDMARK_RADIUS, color, -1, cv2.LINE_AA)
            else:
                cv2.circle(frame, (px, py), LANDMARK_RADIUS, color, -1, cv2.LINE_AA)

        if handedness_list and hand_idx < len(handedness_list):
            hand_name = handedness_list[hand_idx][0].category_name
        else:
            hand_name = "?"

        state_name = states[hand_idx].name if hand_idx < len(states) else "?"
        label = f"{hand_name} | {state_name}"
        wrist = points[0]
        cv2.putText(
            frame, label,
            (wrist[0] - 20, wrist[1] - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
        )


def main():
    print("=" * 60)
    print("                WebShooter FX - AR Camera Filter")
    print("=" * 60)
    print("Controls:")
    print("  Press 'd' to toggle Debug HUD (landmarks, FPS, states)")
    print("  Press 'q' or ESC to exit\n")

    try:
        cam = Camera()
    except RuntimeError as e:
        print(f"[FATAL] {e}")
        return

    detector = HandDetector()
    glitch_effect = GlitchEffect()
    sound_player = SoundPlayer()

    state_machines = {
        "Left": GestureStateMachine("Left"),
        "Right": GestureStateMachine("Right"),
    }

    show_debug = config.SHOW_DEBUG_HUD

    fps_start = time.perf_counter()
    frame_count = 0
    display_fps = 0.0

    try:
        while True:
            ok, frame = cam.read()
            if not ok or frame is None:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = int(time.perf_counter() * 1000)

            result = detector.detect(rgb, timestamp_ms)

            seen_hands = set()
            hand_states = []

            if result.hand_landmarks:
                for hand_idx, landmarks in enumerate(result.hand_landmarks):
                    if result.handedness and hand_idx < len(result.handedness):
                        hand_name = result.handedness[hand_idx][0].category_name
                    else:
                        hand_name = "Right" if hand_idx == 0 else "Left"

                    if hand_name == "Left":
                        hand_name = "Right"
                    elif hand_name == "Right":
                        hand_name = "Left"

                    seen_hands.add(hand_name)
                    gesture = is_web_shooter_pose(landmarks)

                    sm = state_machines.get(hand_name)
                    if sm:
                        prev_state = sm.state
                        new_state = sm.update(gesture)
                        hand_states.append(new_state)

                        # Trigger sound effect on transition into ACTIVE
                        if prev_state != State.ACTIVE and new_state == State.ACTIVE:
                            sound_player.play_thwip()
                    else:
                        hand_states.append(State.IDLE)

            for name, sm in state_machines.items():
                if name not in seen_hands:
                    sm.reset()

            # ─── Layer Ordering (PRD Section 11.3) ───────────────────────────
            # 1. Base frame
            # 2. Glitch effect in-place
            # 3. Web overlay alpha-composited
            # 4. Debug HUD (if enabled)
            active_mask = [st == State.ACTIVE for st in hand_states]
            if result.hand_landmarks and any(active_mask):
                frame = glitch_effect.apply(frame, result.hand_landmarks, active_mask)
                frame = render_web_overlay(frame, result.hand_landmarks, active_mask)

            # Debug HUD overlay
            if show_debug:
                if result.hand_landmarks:
                    draw_landmarks_on_frame(
                        frame, result.hand_landmarks, result.handedness, hand_states
                    )

                frame_count += 1
                elapsed = time.perf_counter() - fps_start
                if elapsed >= 0.5:
                    display_fps = frame_count / elapsed
                    frame_count = 0
                    fps_start = time.perf_counter()

                cv2.putText(
                    frame, f"FPS: {display_fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA,
                )

                hands_found = len(result.hand_landmarks) if result.hand_landmarks else 0
                cv2.putText(
                    frame, f"Hands: {hands_found}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA,
                )

                for i, (name, sm) in enumerate(state_machines.items()):
                    state_color = STATE_COLORS.get(sm.state, (180, 180, 180))
                    cv2.putText(
                        frame, f"{name}: {sm.state.name}",
                        (10, 90 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2, cv2.LINE_AA,
                    )
            else:
                # Small clean indicator when debug HUD is off
                cv2.putText(
                    frame, "[Press 'd' for Debug HUD]",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA,
                )

            cv2.imshow("WebShooter FX", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("d"):
                show_debug = not show_debug
                print(f"[HUD] Debug HUD toggled {'ON' if show_debug else 'OFF'}")
            elif key == ord("q") or key == 27:
                break

    finally:
        detector.close()
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
