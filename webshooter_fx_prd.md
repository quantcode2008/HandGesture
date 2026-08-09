# Product Requirements Document
## Gesture-Triggered AR "Web-Shooter" Camera Filter (working title: **WebShooter FX**)

---

## 0. Document Control

| Field | Value |
|---|---|
| Version | 1.0 |
| Status | Ready for build |
| Author | Misba |
| Intended builder | Claude Code |
| Target repo type | Personal / portfolio project |
| Last updated | 2026-08-09 |

**Note on origin:** This spec reconstructs the target behavior from a short reference clip (two representative frames: a cross-hatched "web mesh" overlay near the head, and a rotated RGB channel-shift/glitch effect over the face), not from the original creator's source code. Exact tuning values (colors, spacing, timing) below are sensible starting points, not ground truth — Phase 5 explicitly budgets time to eyeball-match them against the reference clip.

**IP note:** This is a personal, non-commercial fan-inspired filter (hand pose evokes Spider-Man's "web-shooter" gesture). It does not use any Marvel character models, logos, or likenesses — only a generic geometric overlay and image effect — but keep naming/branding generic in any public repo/README to stay clearly in "fan project" territory.

---

## 1. Overview

WebShooter FX is a real-time webcam filter, written in Python, that watches the user's hands through a laptop camera and fires a stylized visual effect — a web-pattern overlay plus a localized RGB chromatic-aberration "glitch" — whenever the user performs a specific hand gesture (index + pinky extended, middle + ring curled — the classic "web-shooter" pose). The effect is positioned, scaled, and rotated live based on the geometry of the user's hands, so it tracks naturally as they move.

This is a computer-vision + real-time-graphics project, intended as a portfolio/showcase piece demonstrating hand-tracking, gesture classification, and live video-effect compositing.

---

## 2. Problem Statement & Motivation

- Demonstrate practical, real-time computer vision skills (landmark detection, gesture classification, temporal smoothing) beyond CRUD/web-app work.
- Produce a shareable, visually impressive demo (video-clip-friendly) for a developer profile/portfolio.
- Build a reusable gesture → effect trigger pipeline that could later be extended to other gestures/effects.

---

## 3. Goals & Success Criteria

| Goal | Success Criteria |
|---|---|
| Reliable gesture detection | Correctly triggers on the web-shooter pose in >90% of deliberate attempts under normal indoor lighting, in informal testing |
| Low false-positive rate | Does not trigger on an open palm, fist, pointing, thumbs-up, or resting hand in casual use |
| Real-time performance | Sustains ≥ 20 FPS on a mid-range laptop CPU (no dedicated GPU required) at 1280×720 input |
| Visually matches reference | Overlay position/rotation tracks hand movement with no visible lag or jitter; glitch effect reads as an intentional "power-up" pulse, not a rendering bug |
| Clean handoff artifact | A recruiter/viewer can `pip install -r requirements.txt && python main.py` and see the effect working within minutes |

---

## 4. Non-Goals (v1 Scope Boundaries)

- ❌ Mobile app (iOS/Android) — desktop webcam only
- ❌ Multiplayer / multi-person simultaneous effects
- ❌ Sound effects (stretch goal, see §18)
- ❌ Video recording/export pipeline (stretch goal)
- ❌ Web/browser deployment (stretch goal — noted as a natural v2 port since MediaPipe has a JS/WASM equivalent)
- ❌ Training a custom gesture-recognition ML model — v1 uses either MediaPipe's built-in gesture classifier or a hand-written landmark-geometry rule, not a custom-trained network

---

## 5. Target Platform & User

- **Platform:** Windows / macOS / Linux desktop, standard USB or built-in webcam, Python 3.x (see §8 for version caveat)
- **User:** The developer themself, running it locally to capture a demo clip; secondarily, anyone cloning the repo to try it

---

## 6. User Stories

1. *As a user*, I open the app and see my live webcam feed with hand/face tracking running in the background (no visible clutter unless I enable debug mode).
2. *As a user*, when I hold up one or both hands in the web-shooter pose, I see a web-textured overlay and a glitch pulse appear, anchored to my hand position, within a couple of frames — not with a noticeable delay.
3. *As a user*, when I relax my hand, the effect fades out cleanly rather than cutting off abruptly or flickering.
4. *As a user*, if I make a similar-but-different gesture (e.g., a peace sign, a fist), nothing happens.
5. *As a developer*, I can tune detection thresholds and effect parameters from a single config file without touching detection/rendering logic.
6. *As a developer*, I can toggle a debug overlay (landmarks, FPS counter, current state) to diagnose detection issues.

---

## 7. System Architecture

### 7.1 High-Level Data Flow

```
Webcam Frame (BGR, cv2.VideoCapture)
        │
        ▼
Frame Preprocessing (resize/flip/color convert)
        │
        ▼
HandLandmarker.detect_async()  ──────►  up to 2 hands × 21 landmarks + handedness
        │
        ▼
Gesture Classifier  ──────►  per-hand boolean: is_web_shooter_pose
        │
        ▼
Gesture State Machine (per-session)  ──────►  IDLE | ARMING | ACTIVE | COOLDOWN
        │
        ├──(if ACTIVE)──► Web Overlay Renderer   ──┐
        │                                           ├──► Alpha Composite ──► Output Frame ──► cv2.imshow
        └──(if ACTIVE)──► Glitch Effect Renderer  ──┘
```

Face landmarks are **not required for v1 core** (the effect region is derived entirely from hand geometry — see §11). `FaceLandmarker` is an optional v1.1 add-on used only to mask the glitch effect precisely to the face contour instead of a rough rectangle. Keeping it out of v1 core reduces CPU load and dependency surface for the MVP.

### 7.2 Module Breakdown

| Module | Responsibility |
|---|---|
| `camera.py` | Opens webcam, yields frames, handles resolution/FPS config |
| `detectors/hand_detector.py` | Wraps `HandLandmarker`, returns normalized landmarks + handedness per hand |
| `detectors/face_detector.py` | (v1.1) Wraps `FaceLandmarker`, returns face oval contour for masking |
| `gestures/gesture_classifier.py` | Pure function(s): landmarks → gesture boolean/confidence |
| `gestures/state_machine.py` | IDLE/ARMING/ACTIVE/COOLDOWN transitions + timing |
| `effects/web_overlay.py` | Generates and composites the cross-hatch web pattern |
| `effects/glitch.py` | Generates and composites the RGB-shift chromatic aberration |
| `utils/geometry.py` | Distance, angle, midpoint, rotation-matrix helpers |
| `utils/smoothing.py` | Rolling-buffer / exponential smoothing helpers |
| `config.py` | All tunable constants in one place (§14) |
| `main.py` | Wires everything together; the render loop |

---

## 8. Tech Stack & Dependencies

| Component | Choice | Notes |
|---|---|---|
| Language | Python 3.x | Verify current MediaPipe PyPI package's supported Python range at install time — historically MediaPipe has lagged a version or two behind the newest CPython release, so don't assume the very latest Python is supported |
| CV / rendering | OpenCV (`opencv-python`) | Capture, drawing primitives, affine warps, channel ops |
| Hand & face tracking | `mediapipe` — **Tasks API**, not the legacy `mp.solutions.*` API | Google deprecated the legacy Solutions API (hands/face_mesh) in 2023 in favor of `mediapipe.tasks.python.vision`. Use `HandLandmarker` / `FaceLandmarker` / `GestureRecognizer` from `mediapipe.tasks.python.vision` |
| Math | `numpy` | Vector ops, channel shifting |

**requirements.txt**
```
opencv-python
mediapipe
numpy
```

**Model files required (downloaded separately, not bundled in the pip package):**

| Model | Download URL |
|---|---|
| Hand Landmarker | `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task` |
| Gesture Recognizer (optional, Approach A below) | see [MediaPipe Gesture Recognizer model page](https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer#models) |
| Face Landmarker (v1.1 only) | `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker_v2_with_blendshapes.task` |

Store these under `models/` (see §15) and load via `BaseOptions(model_asset_path=...)`. Verify URLs against MediaPipe's current model index before building, since Google occasionally revises bucket paths/versions.

---

## 9. Functional Requirements

| ID | Requirement | Acceptance Criteria |
|---|---|---|
| FR-1 | Capture live webcam video | App opens default camera, displays a live mirrored (selfie-view) feed at configurable resolution |
| FR-2 | Detect up to 2 hands per frame with 21 landmarks each | Landmarks available every frame hands are in view; degrades gracefully to 0/1 hands |
| FR-3 | Classify web-shooter gesture per hand | Returns a boolean or confidence score per detected hand, per frame |
| FR-4 | Debounce gesture into a stable trigger | Effect does not flicker on/off from single-frame misdetections (see §10.5 state machine) |
| FR-5 | Render web-pattern overlay anchored to hand geometry | Overlay position, rotation, and scale update every frame the effect is active |
| FR-6 | Render RGB chromatic-aberration glitch localized to the effect region | Effect is visibly confined to the target region, not the full frame |
| FR-7 | Effect entry/exit is smooth, not a hard cut | Fade/ease in on trigger, decay on release (§11.2) |
| FR-8 | All thresholds/parameters centralized in config | No magic numbers inside detection/rendering logic files |
| FR-9 | Optional debug HUD | Toggle (keypress or config flag) shows: raw landmarks, FPS counter, current state-machine state |
| FR-10 | Runs without crashing when no hand/face is visible | No exceptions on empty detection results; effect simply doesn't render |

---

## 10. Gesture Detection — Detailed Spec

### 10.1 Hand Landmark Reference (MediaPipe 21-point topology)

| # | Landmark | # | Landmark |
|---|---|---|---|
| 0 | WRIST | 11 | MIDDLE_FINGER_DIP |
| 1 | THUMB_CMC | 12 | MIDDLE_FINGER_TIP |
| 2 | THUMB_MCP | 13 | RING_FINGER_MCP |
| 3 | THUMB_IP | 14 | RING_FINGER_PIP |
| 4 | THUMB_TIP | 15 | RING_FINGER_DIP |
| 5 | INDEX_FINGER_MCP | 16 | RING_FINGER_TIP |
| 6 | INDEX_FINGER_PIP | 17 | PINKY_MCP |
| 7 | INDEX_FINGER_DIP | 18 | PINKY_PIP |
| 8 | INDEX_FINGER_TIP | 19 | PINKY_DIP |
| 9 | MIDDLE_FINGER_MCP | 20 | PINKY_TIP |
| 10 | MIDDLE_FINGER_PIP | | |

Each landmark has normalized `(x, y, z)` in `[0, 1]` relative to image width/height (z is relative depth from the wrist).

### 10.2 Two Implementation Approaches — pick one for v1

**Approach A — Use MediaPipe's built-in Gesture Recognizer (recommended for speed of build)**
MediaPipe ships a canned gesture classifier via the `GestureRecognizer` task with these categories: `Unknown, Closed_Fist, Open_Palm, Pointing_Up, Thumb_Down, Thumb_Up, Victory, ILoveYou`. The `ILoveYou` category (thumb + index + pinky extended, middle + ring curled) is a very close match to the web-shooter pose. Trade-off: it's a trained model, not tunable finger-by-finger, and technically expects the thumb extended too (the reference gesture is often performed with the thumb tucked in, which may or may not still classify as `ILoveYou` in practice — needs empirical testing).

**Approach B — Custom rule-based classifier from raw `HandLandmarker` output (recommended for precision/control)**
Write an explicit finger-extension classifier (below) directly against the 21 landmarks. More code, but exact control over what counts as a match, and no dependency on a second model bundle.

**Recommendation:** Build Approach B first (it's the more instructive/portfolio-relevant piece and gives full control over the thumb condition); keep Approach A noted in code comments as a drop-in alternative.

### 10.3 Finger Extension Algorithm (Approach B)

Naive approach (compare tip vs. PIP joint y-coordinate) breaks when the hand is rotated or upside-down. Use an **orientation-invariant distance-ratio** test instead: a finger is "extended" if its tip is meaningfully farther from the wrist than its own MCP (knuckle) joint is.

```python
def is_finger_extended(landmarks, tip_idx, mcp_idx, wrist_idx=0, ratio_threshold=1.3):
    tip = landmarks[tip_idx]
    mcp = landmarks[mcp_idx]
    wrist = landmarks[wrist_idx]
    d_tip_wrist = euclidean(tip, wrist)
    d_mcp_wrist = euclidean(mcp, wrist)
    return d_tip_wrist > d_mcp_wrist * ratio_threshold
```

Thumb is handled separately since it moves in a different plane — check whether the thumb tip is tucked toward the palm rather than extended outward:

```python
def is_thumb_tucked(landmarks, tolerance=...):
    thumb_tip = landmarks[4]
    index_mcp = landmarks[5]
    pinky_mcp = landmarks[17]
    # Tucked thumb sits closer to the index-side knuckle than a fully
    # extended thumb would.
    return euclidean(thumb_tip, index_mcp) < euclidean(pinky_mcp, index_mcp) * tolerance
```

Treat the thumb check as a **soft/optional** signal in v1 (thumb tracking is the least reliable of the five fingers) — don't hard-require it, or the gesture will feel finicky to perform.

### 10.4 Web-Shooter Gesture Definition

| Finger | Required state | Landmarks used |
|---|---|---|
| Index | Extended | tip 8, MCP 5 |
| Middle | Curled | tip 12, MCP 9 |
| Ring | Curled | tip 16, MCP 13 |
| Pinky | Extended | tip 20, MCP 17 |
| Thumb | Optional/soft check | tip 4, MCP 2, landmarks 5/17 |

```python
def is_web_shooter_pose(landmarks):
    index_ext  = is_finger_extended(landmarks, 8, 5)
    middle_ext = is_finger_extended(landmarks, 12, 9)
    ring_ext   = is_finger_extended(landmarks, 16, 13)
    pinky_ext  = is_finger_extended(landmarks, 20, 17)
    return index_ext and pinky_ext and not middle_ext and not ring_ext
```

### 10.5 Temporal Smoothing & State Machine

Raw per-frame classification is noisy — require a short hold before triggering, and a short grace period before releasing, so the effect doesn't strobe.

**States:** `IDLE → ARMING → ACTIVE → COOLDOWN → IDLE`

| State | Entry condition | Behavior | Exit condition |
|---|---|---|---|
| IDLE | Default / after cooldown | No effect rendered | Gesture detected → ARMING |
| ARMING | Gesture detected this frame | Increment hold counter; no effect yet | Counter ≥ `GESTURE_HOLD_FRAMES` → ACTIVE; gesture drops → back to IDLE |
| ACTIVE | Hold threshold met | Effect renders (overlay + glitch pulse) | Gesture undetected for > `MISS_TOLERANCE_FRAMES` consecutive frames → COOLDOWN |
| COOLDOWN | Effect just ended | No effect renders; short forced pause | `COOLDOWN_MS` elapsed → IDLE |

This buys two things: (1) `GESTURE_HOLD_FRAMES` prevents a one-frame misclassification from firing the effect; (2) `MISS_TOLERANCE_FRAMES` prevents brief tracking dropouts (motion blur, hand partially leaving frame) from cutting the effect off mid-gesture, and `COOLDOWN_MS` stops immediate re-triggering/strobing right at the detection boundary.

**Hand pairing note:** `HandLandmarker` returns per-hand results with handedness (Left/Right) each frame; v1 does not need custom multi-frame hand-identity tracking — just read both hands' landmark sets each frame. If more than 2 hands are ever detected (e.g., a second person walks into frame), keep the 2 with the largest bounding-box area (assume closest to camera = the user) and ignore the rest.

---

## 11. Visual Effects — Detailed Spec

### 11.1 Web Overlay

**Trigger modes:**
- **Single-hand mode (MVP):** effect fires when at least one hand is in state ACTIVE. Overlay anchors to that hand's palm center (average of landmarks 0, 5, 9, 13, 17), fixed default size from config.
- **Two-hand mode (full behavior, matches reference clip):** when *both* hands are simultaneously ACTIVE, the overlay region is defined by the two hands instead of a fixed size:
  - **Anchor (center):** midpoint between each hand's palm center
  - **Rotation angle:** `atan2(dy, dx)` between the two hands' palm centers
  - **Width:** distance between the two palm centers × `WEB_WIDTH_SCALE`, clamped to `[MIN_WIDTH, MAX_WIDTH]`
  - **Height:** `WEB_HEIGHT_RATIO × width`, or a fixed config value

**Pattern generation:**
1. Render the cross-hatch pattern (two sets of parallel lines at diagonal angles, spaced by `WEB_LINE_SPACING`) onto a small transparent RGBA canvas at 0° rotation, sized to the computed width/height.
2. Rotate that canvas with `cv2.warpAffine` using a rotation matrix around its own center, by the computed angle.
3. Alpha-composite the rotated canvas onto the main frame at the anchor position, with edge-clipping against frame bounds and opacity `WEB_OVERLAY_OPACITY`, tinted per `WEB_OVERLAY_COLOR` (reference clip suggests a pale pink/white).

### 11.2 Chromatic Aberration / Glitch Effect

**Region:**
- **v1 (MVP):** apply within the same rotated bounding rectangle as the web overlay — simplest to implement, still visually close to the reference clip.
- **v1.1 (refinement):** clip the effect to the actual face contour using `FaceLandmarker`'s face-oval landmark subset as a mask, so the glitch hugs the face shape rather than a rectangle.

**Algorithm (per frame, while active):**
```python
b, g, r = cv2.split(region)
shift_px = int(current_intensity * GLITCH_MAX_SHIFT_PX)
r_shifted = np.roll(r, shift_px, axis=1)
b_shifted = np.roll(b, -shift_px, axis=1)
glitched = cv2.merge([b_shifted, g, r_shifted])
region[:] = cv2.addWeighted(region, 1 - GLITCH_BLEND, glitched, GLITCH_BLEND, 0)
```

**Intensity envelope (attack–decay pulse, not a static overlay):**
- On trigger (state → ACTIVE): `current_intensity` ramps 0 → 1 over `GLITCH_ATTACK_FRAMES`
- While ACTIVE: `current_intensity` decays each frame — `current_intensity *= GLITCH_DECAY_RATE` — creating a "power-up pulse" rather than a constant shimmer; optionally re-spike on each new frame the gesture is freshly re-confirmed
- Optional polish: add small per-frame random jitter (± a few px) to `shift_px` for an "unstable signal" feel, and/or faint horizontal scanline darkening (stretch — see §18)

### 11.3 Compositing Order

```
1. Draw base camera frame
2. If ACTIVE: apply glitch effect to target region (in place)
3. If ACTIVE: alpha-composite web overlay on top
4. If DEBUG: draw landmark dots/skeleton + FPS + state text
5. Display frame
```

---

## 12. Non-Functional Requirements

| Category | Requirement |
|---|---|
| Performance | ≥ 20 FPS target on a mid-range laptop CPU at 1280×720; minimum acceptable 15 FPS. If unmet: drop capture resolution, use the "lite" model variants where available, or run detection every other frame with interpolated landmarks in between |
| Reliability | No unhandled exceptions on: no camera found, no hand in frame, no face in frame (v1.1), hand leaving mid-gesture |
| Portability | Runs on Windows/macOS/Linux — pure OpenCV + MediaPipe, no OS-specific calls |
| Maintainability | Config-driven thresholds (§14); no cross-module magic numbers |
| Privacy | 100% local processing — no frame data leaves the device, nothing is uploaded or persisted unless recording is explicitly added later (§18) |

---

## 13. Edge Cases & Error Handling

| Case | Handling |
|---|---|
| No camera detected | Print clear error, exit gracefully (no stack trace) |
| Model `.task` file missing | Fail fast at startup with a message telling the user which file is missing and its expected download URL/path |
| Zero hands in frame | State machine simply stays/returns to IDLE; no crash |
| Only one hand present during two-hand mode | Fall back to single-hand mode automatically rather than blocking the effect entirely |
| More than 2 hands detected (extra people) | Keep the 2 largest/most confident, ignore rest |
| Very low light | Out of scope to "fix" algorithmically in v1 — document as a known limitation in the README |
| Gesture held right at the confidence boundary | Handled by `GESTURE_HOLD_FRAMES` + `MISS_TOLERANCE_FRAMES` debounce (§10.5) |
| Device too slow for target FPS | Documented fallback options in NFR row above; consider exposing a `--low-power` CLI flag that lowers resolution and disables debug drawing |

---

## 14. Configuration Reference

All values below live in `config.py`; treat these as starting points to tune during Phase 5.

| Parameter | Purpose | Starting value |
|---|---|---|
| `CAMERA_INDEX` | Which camera device to open | `0` |
| `FRAME_WIDTH` / `FRAME_HEIGHT` | Capture resolution | `1280` / `720` |
| `MIN_HAND_DETECTION_CONFIDENCE` | HandLandmarker threshold | `0.6` |
| `MIN_HAND_PRESENCE_CONFIDENCE` | HandLandmarker threshold | `0.6` |
| `EXTENSION_RATIO_THRESHOLD` | Finger extended/curled cutoff (§10.3) | `1.3` |
| `GESTURE_HOLD_FRAMES` | Frames required to confirm trigger | `5` (~165 ms at 30 FPS) |
| `MISS_TOLERANCE_FRAMES` | Dropout frames tolerated before releasing | `6` |
| `COOLDOWN_MS` | Forced pause before re-arming | `500` |
| `WEB_WIDTH_SCALE` | Overlay width relative to inter-hand distance | `1.4` |
| `WEB_HEIGHT_RATIO` | Overlay height relative to width | `0.6` |
| `WEB_LINE_SPACING` | Cross-hatch line spacing (px) | `12` |
| `WEB_OVERLAY_OPACITY` | Overlay alpha | `0.35` |
| `WEB_OVERLAY_COLOR` | Overlay tint (BGR) | pale pink/white, tune to reference |
| `GLITCH_MAX_SHIFT_PX` | Max per-channel pixel shift | `14` |
| `GLITCH_ATTACK_FRAMES` | Frames to ramp intensity to full | `3` |
| `GLITCH_DECAY_RATE` | Per-frame decay multiplier | `0.85` |
| `GLITCH_BLEND` | Blend weight of glitched vs. original | `0.8` |
| `SHOW_DEBUG_HUD` | Toggle landmark/FPS/state overlay | `False` |

---

## 15. File / Folder Structure

```
webshooter-fx/
├── main.py
├── config.py
├── camera.py
├── detectors/
│   ├── __init__.py
│   ├── hand_detector.py
│   └── face_detector.py          # v1.1
├── gestures/
│   ├── __init__.py
│   ├── gesture_classifier.py
│   └── state_machine.py
├── effects/
│   ├── __init__.py
│   ├── web_overlay.py
│   └── glitch.py
├── utils/
│   ├── __init__.py
│   ├── geometry.py
│   └── smoothing.py
├── models/
│   ├── hand_landmarker.task
│   └── face_landmarker_v2_with_blendshapes.task   # v1.1
├── tests/
│   ├── test_gesture_classifier.py
│   └── test_geometry.py
├── requirements.txt
└── README.md
```

---

## 16. Testing Plan

**Unit tests (no camera needed):**
- `gesture_classifier`: feed synthetic landmark fixtures for known poses (web-shooter, open palm, fist, thumbs-up, victory) and assert correct boolean output for each
- `geometry`: verify angle/distance/midpoint helpers against hand-computed expected values
- `state_machine`: simulate frame sequences (e.g., gesture detected for 3 frames then dropped) and assert correct state transitions

**Manual QA checklist (live webcam):**
- [ ] Effect triggers within ~200ms of forming the gesture
- [ ] Effect does not trigger on open palm / fist / pointing / thumbs-up / peace sign
- [ ] Overlay rotates correctly as hands tilt
- [ ] Overlay scales correctly as hands move closer/farther apart
- [ ] Effect fades out smoothly on release, no hard cut or flicker
- [ ] App survives hands leaving the frame entirely
- [ ] App survives camera occlusion (hand fully covering lens) and recovers
- [ ] Debug HUD toggle works and shows accurate FPS/state

**Performance benchmark:**
- Log rolling-average FPS to console/HUD; record baseline FPS with effect off vs. on to quantify effect-rendering overhead

---

## 17. Build Phases & Acceptance Criteria

| Phase | Scope | Done when |
|---|---|---|
| 0 — Setup | Repo scaffold, venv, `requirements.txt`, download model files | `python -c "import cv2, mediapipe, numpy"` runs clean; model files present in `models/` |
| 1 — Tracking foundation | Camera capture loop + `HandLandmarker` wired up, draw raw landmarks | Live feed shows accurate landmark dots on both hands at ≥ 20 FPS |
| 2 — Gesture logic | Finger-extension + web-shooter classifier, state machine, console-log transitions | Console logs correct IDLE→ARMING→ACTIVE→COOLDOWN transitions matching real gesture attempts |
| 3 — Web overlay | Overlay geometry (anchor/angle/scale) + cross-hatch rendering, single-hand mode first, then two-hand mode | Overlay visibly tracks hand position/rotation/scale in real time |
| 4 — Glitch effect | Channel-shift algorithm + attack/decay envelope, composited into the same region | Glitch pulses in sync with gesture trigger, confined to target region |
| 5 — Integration & polish | Combine all effects, add debug HUD, tune `config.py` values against the reference clip, write README | End-to-end demo matches the reference clip's look closely enough for a portfolio clip; README lets a stranger run it in minutes |
| 6 — Stretch (optional) | See §18 | N/A — post-MVP |

---

## 18. Future Enhancements (Stretch Goals)

- 🔊 Sound effect triggered on gesture activation
- ✨ Replace the static cross-hatch overlay with an animated particle "web-shoot" burst
- 🎥 Clip recording/export (auto-save a short `.mp4` around each trigger) for easy social/portfolio sharing
- 🌐 Port to browser via `@mediapipe/tasks-vision` (JS/WASM) for a shareable link instead of a local script
- 🎭 Additional gesture → effect presets (a small library of "superhero filter" triggers)
- 🧠 Swap the hand-written classifier for MediaPipe's `GestureRecognizer` custom-gesture fine-tuning if precision needs improve beyond rule-based matching

---

## 19. Appendix

**Glossary**
- **Debounce:** requiring a signal to be stable for N frames/ms before acting on it, to filter out noise/flicker
- **Chromatic aberration:** a visual effect (here, deliberately applied) where color channels are offset, mimicking a lens/glitch artifact
- **Alpha compositing:** blending a semi-transparent layer onto a base image using a per-pixel opacity value
- **Landmark:** a tracked (x, y, z) keypoint on a detected hand or face

**Reference documentation**
- MediaPipe Tasks — Hand Landmarker (Python): `https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python`
- MediaPipe Tasks — Face Landmarker (Python): `https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python`
- MediaPipe Tasks — Gesture Recognizer: `https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer`
