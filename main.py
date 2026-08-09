"""
main.py - SnapFrame (Complete Application + Filmstrip Carousel Stretch Goal).

Touchless gesture-controlled camera filter application with 10 visual filters.
- Snap Right Hand -> ADVANCE Filter (+1) with smooth cross-fade transition
- Snap Left Hand  -> REVERSE Filter (-1) with smooth cross-fade transition
- Make a Fist     -> CAPTURE Frame (saved to captures/ folder)
- Filmstrip Carousel -> Live top thumbnail bar showing all 10 filter previews

Controls:
  'f' - Toggle Filmstrip thumbnail carousel bar
  'd' - Toggle Debug HUD (landmarks, FPS, filter index, gesture events)
  'q' or ESC - Exit application
"""

import time
import cv2
import numpy as np

import config
from camera import Camera
from capture.capture_manager import CaptureManager
from detectors.hand_detector import HandDetector
from filters.registry import FILTER_REGISTRY
from gestures.event_manager import GestureEventManager, FilterCommand
from gestures.fist_detector import is_fist
from gestures.snap_detector import compute_normalized_thumb_middle_dist
from ui.filmstrip import FilmstripManager

# --- MediaPipe hand-skeleton connections (21-point topology) ----------------
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

LANDMARK_RADIUS = 5
CONNECTION_THICKNESS = 2


def draw_landmarks_on_frame(frame, hand_landmarks_list, handedness_list):
    """Draw landmark skeleton and handedness labels for Debug HUD mode."""
    h, w, _ = frame.shape

    for hand_idx, landmarks in enumerate(hand_landmarks_list):
        if handedness_list and hand_idx < len(handedness_list):
            hand_name = handedness_list[hand_idx][0].category_name
        else:
            hand_name = "?"

        if hand_name == "Left":
            hand_name = "Right"
        elif hand_name == "Right":
            hand_name = "Left"

        color = (255, 200, 0) if hand_name == "Right" else (147, 20, 255)

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

        norm_dist = compute_normalized_thumb_middle_dist(landmarks)
        fist_str = " [FIST]" if is_fist(landmarks) else ""
        label = f"{hand_name} | d={norm_dist:.2f}{fist_str}"
        wrist = points[0]
        cv2.putText(
            frame, label,
            (wrist[0] - 20, wrist[1] - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
        )


def draw_filter_banner(frame, filter_name: str, filter_idx: int, total_filters: int):
    """Draw semi-transparent bottom banner displaying current filter title (FR-8)."""
    h, w, _ = frame.shape
    banner_height = 55

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - banner_height), (w, h), (15, 15, 25), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

    title_text = f"FILTER [{filter_idx + 1}/{total_filters}]: {filter_name}"
    cv2.putText(
        frame, title_text,
        (20, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
    )

    help_text = "Snap Right/Left: Next/Prev | Fist: Capture | 'f': Filmstrip"
    cv2.putText(
        frame, help_text,
        (w - 490, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA,
    )


def main():
    print("=" * 60)
    print("             SnapFrame — Camera Filter Application")
    print("=" * 60)
    print("Gesture Controls:")
    print("  Snap Right Hand -> ADVANCE Filter (+1)")
    print("  Snap Left Hand  -> REVERSE Filter (-1)")
    print("  Make a Fist     -> CAPTURE Frame\n")
    print("Controls:")
    print("  Press 'f' to toggle Filmstrip Thumbnail Bar")
    print("  Press 'd' to toggle Debug HUD")
    print("  Press 'q' or ESC to exit\n")

    try:
        cam = Camera()
    except RuntimeError as e:
        print(f"[FATAL] {e}")
        return

    detector = HandDetector()
    event_manager = GestureEventManager()
    capture_manager = CaptureManager()
    filmstrip_manager = FilmstripManager()

    filter_index = 0
    num_filters = len(FILTER_REGISTRY)
    show_debug = config.SHOW_DEBUG_HUD

    # Cross-fade transition state
    old_filter_index = filter_index
    transition_start_time = 0.0
    transition_dur_sec = config.FILTER_TRANSITION_MS / 1000.0

    fps_start = time.perf_counter()
    frame_count = 0
    display_fps = 0.0

    last_event_text = "None"
    last_event_time = 0.0

    try:
        while True:
            ok, frame = cam.read()
            if not ok or frame is None:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            now = time.perf_counter()
            timestamp_ms = int(now * 1000)

            # 1. Run hand tracking
            result = detector.detect(rgb, timestamp_ms)

            # 2. Update gesture detectors & event manager
            cmd, trigger_capture = event_manager.process_frame(
                result.hand_landmarks if result else [],
                result.handedness if result else [],
                now,
            )

            # 3. Resolve snap events -> filter navigation with cross-fade (FR-5, §11.2)
            if cmd == FilterCommand.ADVANCE:
                old_filter_index = filter_index
                filter_index = (filter_index + 1) % num_filters
                transition_start_time = now
                last_event_text = f"SNAP RIGHT -> [{filter_index + 1}/10]"
                last_event_time = now
                print(f"[NAV] Filter ADVANCED -> [{filter_index + 1}/10]: {FILTER_REGISTRY[filter_index][0]}")

            elif cmd == FilterCommand.REVERSE:
                old_filter_index = filter_index
                filter_index = (filter_index - 1) % num_filters
                transition_start_time = now
                last_event_text = f"SNAP LEFT -> [{filter_index + 1}/10]"
                last_event_time = now
                print(f"[NAV] Filter REVERSED -> [{filter_index + 1}/10]: {FILTER_REGISTRY[filter_index][0]}")

            # 4. Filter Rendering with Cross-Fade Transition (§11.2)
            filter_name, filter_func = FILTER_REGISTRY[filter_index]
            dt_trans = now - transition_start_time

            if dt_trans < transition_dur_sec and old_filter_index != filter_index:
                alpha = np.clip(dt_trans / transition_dur_sec, 0.0, 1.0)
                old_func = FILTER_REGISTRY[old_filter_index][1]
                frame_new = filter_func(frame)
                frame_old = old_func(frame)
                filtered_frame = cv2.addWeighted(frame_new, alpha, frame_old, 1.0 - alpha, 0)
            else:
                filtered_frame = filter_func(frame)

            # 5. Handle Fist Capture trigger (FR-6, §12)
            if trigger_capture:
                last_event_text = "FIST -> CAPTURE!"
                last_event_time = now
                capture_manager.save_capture(filtered_frame, filter_name, now)

            # Create output frame for UI overlays
            output_frame = filtered_frame.copy()

            # 6. Render Filmstrip Thumbnail Carousel Bar (Stretch Goal §19)
            output_frame = filmstrip_manager.render(output_frame, filter_index)

            # 7. Render filter name banner (FR-8)
            draw_filter_banner(output_frame, filter_name, filter_index, num_filters)

            # 8. Render capture flash & badge feedback (FR-10)
            output_frame = capture_manager.render_feedback(output_frame, now)

            # 9. Render Debug HUD if enabled (FR-12)
            if show_debug:
                if result and result.hand_landmarks:
                    draw_landmarks_on_frame(output_frame, result.hand_landmarks, result.handedness)

                frame_count += 1
                elapsed = now - fps_start
                if elapsed >= 0.5:
                    display_fps = frame_count / elapsed
                    frame_count = 0
                    fps_start = now

                hud_y_start = 110 if config.SHOW_FILMSTRIP else 30

                cv2.putText(
                    output_frame, f"FPS: {display_fps:.1f}",
                    (10, hud_y_start),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA,
                )

                hands_found = len(result.hand_landmarks) if result and result.hand_landmarks else 0
                cv2.putText(
                    output_frame, f"Hands: {hands_found}",
                    (10, hud_y_start + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA,
                )

                cv2.putText(
                    output_frame, f"Index: {filter_index + 1}/{num_filters}",
                    (10, hud_y_start + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA,
                )

                if now - last_event_time < 1.5:
                    cv2.putText(
                        output_frame, f"Event: {last_event_text}",
                        (10, hud_y_start + 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA,
                    )
            else:
                top_hint_y = 100 if config.SHOW_FILMSTRIP else 30
                cv2.putText(
                    output_frame, "[Press 'd' for Debug HUD]",
                    (10, top_hint_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA,
                )

            # 10. Display frame
            cv2.imshow("SnapFrame", output_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("f"):
                config.SHOW_FILMSTRIP = not config.SHOW_FILMSTRIP
                print(f"[FILMSTRIP] Filmstrip carousel toggled {'ON' if config.SHOW_FILMSTRIP else 'OFF'}")
            elif key == ord("d"):
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
