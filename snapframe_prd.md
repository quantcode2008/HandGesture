# Product Requirements Document
## Gesture-Controlled Camera Filter App (working title: **SnapFrame**)

> **Supersedes** `webshooter_fx_prd.md`. The web-shooter gesture, web-mesh overlay, and glitch effect are dropped entirely. Hand-tracking foundation (HandLandmarker, Tasks API) is retained; everything downstream of it is new.

---

## 0. Document Control

| Field | Value |
|---|---|
| Version | 1.0 |
| Status | Ready for build |
| Author | Misba |
| Intended builder | Claude Code / Google Antigravity |
| Target repo type | Personal / portfolio project |
| Last updated | 2026-08-09 |

---

## 1. Overview

SnapFrame is a real-time webcam app with 10 distinct visual filters (in the spirit of a phone camera's filter carousel), controlled entirely by hand gestures — no mouse, no keyboard:

- **Snap right hand** → advance to the next filter
- **Snap left hand** → go back to the previous filter
- **Make a fist** → capture the current frame with the active filter applied, saved to disk

Filters cycle like a slideshow, one active at a time, with the current filter's name shown on screen.

---

## 2. Problem Statement & Motivation

- Showcase a more advanced, temporally-aware gesture-recognition pipeline (finger snaps are a *motion* gesture, not a static pose — meaningfully harder than the earlier static-pose detection) alongside real image-processing/filter engineering.
- Produce a genuinely fun, demo-able portfolio piece: a touchless camera app is immediately understandable and impressive in a 15-second clip.
- Build a clean, extensible filter architecture (one filter = one function) that's easy to grow beyond 10.

---

## 3. Goals & Success Criteria

| Goal | Success Criteria |
|---|---|
| Reliable snap detection | Correctly registers a deliberate snap on either hand in the large majority of attempts, in informal testing, without needing to snap unnaturally hard/close to camera |
| Correct hand → direction mapping | Right-hand snap reliably advances, left-hand snap reliably reverses — verified and calibrated against how the mirrored feed actually looks to the user (see §10.4) |
| No accidental captures | A held fist captures exactly once per fist, not a burst of duplicate saves |
| Distinct, high-quality filters | All 10 filters are visually distinguishable from each other at a glance; none look like a broken/glitched version of another |
| Real-time performance | ≥ 20 FPS on a mid-range laptop CPU at 1280×720 |
| Clean handoff artifact | `pip install -r requirements.txt && python main.py` works within minutes for a stranger cloning the repo |

---

## 4. Non-Goals (v1 Scope Boundaries)

- ❌ Mobile app — desktop webcam only
- ❌ Multi-person simultaneous use (v1 assumes one user's hands in frame)
- ❌ Filter intensity sliders / manual adjustment (fixed-strength filters in v1)
- ❌ Video recording (stretch goal, §19)
- ❌ Cloud upload / sharing of captures
- ❌ Training a custom ML gesture model — snap and fist detection are both landmark-geometry/motion-rule based, not custom-trained networks

---

## 5. Target Platform & User

- **Platform:** Windows / macOS / Linux desktop, standard webcam, Python 3.x
- **User:** The developer themself for demo/portfolio capture; secondarily, anyone cloning the repo

---

## 6. User Stories

1. *As a user*, I open the app and see my live camera feed with the first filter already applied and its name shown on screen.
2. *As a user*, I snap my right hand and the next filter smoothly takes over within a couple of frames.
3. *As a user*, I snap my left hand and the previous filter comes back — including wrapping from filter 1 back to filter 10.
4. *As a user*, I make a fist and get a brief visual confirmation (flash/checkmark) that a photo was captured, and I can find that photo in a `captures/` folder afterward with the correct filter baked in.
5. *As a user*, if I accidentally hold a fist for a few seconds, I get exactly one capture, not several.
6. *As a developer*, I can add an 11th filter by dropping in one new file and registering it — no changes to gesture or camera code required.

---

## 7. System Architecture

### 7.1 High-Level Data Flow

```
Webcam Frame (BGR, cv2.VideoCapture)
        │
        ▼
Frame Preprocessing (resize, mirror-flip for selfie view)
        │
        ▼
HandLandmarker.detect_async()  ──────►  up to 2 hands × 21 landmarks + handedness
        │
        ├──► Fist Detector  ──────►  per-hand boolean (this frame)
        │
        └──► Snap Detector  ──────►  per-hand event (fires once, on release)
        │
        ▼
Event Manager (debounce + cooldown + handedness calibration)
        │
        ├──(right snap event)──► filter_index += 1  (mod 10)
        ├──(left snap event)───► filter_index -= 1  (mod 10)
        └──(fist confirmed, edge-triggered)──► Capture Manager.save(current_frame)
        │
        ▼
Filter Engine (apply filters[filter_index] to frame)
        │
        ▼
UI Overlay (filter name, optional debug HUD, optional capture flash)
        │
        ▼
cv2.imshow
```

### 7.2 Module Breakdown

| Module | Responsibility |
|---|---|
| `camera.py` | Opens webcam, yields frames, handles resolution/FPS config, mirror flip |
| `detectors/hand_detector.py` | Wraps `HandLandmarker` (Tasks API), returns landmarks + handedness per hand |
| `gestures/fist_detector.py` | Per-frame boolean: is this hand a closed fist |
| `gestures/snap_detector.py` | Per-hand temporal state machine: tracks thumb–middle distance over time, fires a discrete snap event on a fast contact→release cycle |
| `gestures/event_manager.py` | Debounces/cooldowns both gesture types into single, non-repeating trigger events; owns the handedness-calibration flag (§10.4) |
| `filters/registry.py` | Ordered list of the 10 filters (name + function reference) |
| `filters/*.py` | One file per filter — pure function: frame in, filtered frame out |
| `capture/capture_manager.py` | Saves the current filtered frame to disk with a timestamped filename, manages cooldown, triggers on-screen feedback |
| `utils/geometry.py` | Distance/velocity helpers used by the snap detector |
| `config.py` | All tunable constants |
| `main.py` | Wires everything together; the render loop |

---

## 8. Tech Stack & Dependencies

| Component | Choice | Notes |
|---|---|---|
| Language | Python 3.x | Verify current MediaPipe PyPI package's supported Python range at install time |
| CV / rendering | `opencv-python` | Capture, filter math, drawing, blending |
| Hand tracking | `mediapipe` — **Tasks API** (`mediapipe.tasks.python.vision`), not the deprecated `mp.solutions.*` | Only `HandLandmarker` is required — no face tracking needed for this version |
| Math | `numpy` | Vectorized filter math (LUTs, matrix color transforms) |

**requirements.txt**
```
opencv-python
mediapipe
numpy
```

**Model file required:**

| Model | Download URL |
|---|---|
| Hand Landmarker | `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task` |

Store under `models/hand_landmarker.task`, loaded via `BaseOptions(model_asset_path=...)`.

**Optional (Approach A for fist only, see §10.2):** MediaPipe's `GestureRecognizer` task ships a canned `Closed_Fist` category — a trained-model alternative to the hand-written fist classifier below. There is **no canned "snap" gesture** in MediaPipe's built-in set, so the snap detector must be custom regardless.

---

## 9. Functional Requirements

| ID | Requirement | Acceptance Criteria |
|---|---|---|
| FR-1 | Capture live webcam video, mirrored for natural selfie view | Live feed displays correctly oriented (not laterally reversed from the user's perspective) |
| FR-2 | Detect up to 2 hands per frame with 21 landmarks + handedness each | Landmarks available every frame hands are in view |
| FR-3 | Detect a closed-fist pose per hand, per frame | Correctly true on a genuine fist, false on open palm / pointing / relaxed hand |
| FR-4 | Detect a finger-snap motion per hand (temporal, not per-frame) | Fires exactly one event per physical snap, not zero and not multiple |
| FR-5 | Right-hand snap advances filter index; left-hand snap reverses it, both wrapping at the ends | Verified against actual left/right as perceived by the user in the mirrored view (§10.4) |
| FR-6 | Fist triggers exactly one capture per fist gesture, even if held | Enforced via edge-triggering + cooldown, not per-frame firing |
| FR-7 | 10 distinct, high-quality filters, each a pure function over a frame | Each filter visually distinguishable; documented technique per filter (§11.1) |
| FR-8 | Current filter name displayed on screen at all times | Text updates immediately on filter change |
| FR-9 | Captured images saved to disk with the active filter baked in | File appears in `captures/` with correct visual content and a sensible filename |
| FR-10 | Visual feedback on capture (flash/checkmark/text) | Feedback visible for a short, config-defined duration, doesn't block the render loop |
| FR-11 | All thresholds/parameters centralized in config | No magic numbers inside detection/filter logic |
| FR-12 | Optional debug HUD | Toggle shows: raw landmarks, FPS, current filter index, last gesture event |
| FR-13 | Runs without crashing when no hand is visible | No exceptions on empty detection results |

---

## 10. Gesture Detection — Detailed Spec

### 10.1 Hand Landmark Reference (unchanged — MediaPipe 21-point topology)

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

### 10.2 Fist Detection

**Approach A (recommended, less code):** Use MediaPipe `GestureRecognizer`'s canned `Closed_Fist` category directly — it's a trained model specifically for this pose and should be more robust than a hand-written rule, at the cost of one extra model bundle.

**Approach B (custom, full control):** All four fingers curled + thumb tucked, reusing the orientation-invariant extension check:

```python
def is_fist(landmarks, ratio_threshold=1.3):
    index_ext  = is_finger_extended(landmarks, 8, 5, ratio_threshold)
    middle_ext = is_finger_extended(landmarks, 12, 9, ratio_threshold)
    ring_ext   = is_finger_extended(landmarks, 16, 13, ratio_threshold)
    pinky_ext  = is_finger_extended(landmarks, 20, 17, ratio_threshold)
    thumb_tucked = is_thumb_tucked(landmarks)
    return not (index_ext or middle_ext or ring_ext or pinky_ext) and thumb_tucked
```

(`is_finger_extended` / `is_thumb_tucked` — same distance-ratio helpers as before: tip-to-wrist distance vs. MCP-to-wrist distance.)

### 10.3 Snap Detection (custom, temporal — the core new piece)

A snap is a **motion**, not a pose: thumb and middle fingertip come together (contact), then rapidly separate (release/flick). Per-frame classification cannot detect this — it requires tracking the **thumb-tip ↔ middle-fingertip distance over a short window of frames**.

**Signal:** normalized distance between landmark 4 (THUMB_TIP) and landmark 12 (MIDDLE_FINGER_TIP), normalized by a hand-size reference (e.g., divide by the distance from WRIST (0) to MIDDLE_FINGER_MCP (9)) so the same physical snap registers consistently regardless of how close the hand is to the camera.

**Per-hand rolling buffer:** keep the last `SNAP_WINDOW_FRAMES` samples of `(timestamp, normalized_distance)`.

**Detection logic:**
1. **Contact phase:** at some point in the buffer, `normalized_distance` drops below `SNAP_CONTACT_THRESHOLD` (thumb and middle finger are touching/near-touching).
2. **Release phase:** within `SNAP_RELEASE_WINDOW_FRAMES` frames *after* the contact point, `normalized_distance` rises above `SNAP_RELEASE_THRESHOLD`.
3. **Speed check:** the release must happen fast — `(distance_now - distance_at_contact) / (time_now - time_at_contact) > SNAP_RELEASE_VELOCITY_THRESHOLD` — to reject slow, deliberate finger-opening motions that aren't actually snaps.
4. When all three conditions are met, fire a single `SnapEvent(hand=handedness, timestamp=...)` and clear that hand's buffer to avoid re-firing on the same physical motion.

```python
def update_snap_detector(hand_buffer, new_sample):
    hand_buffer.append(new_sample)  # (t, normalized_distance)
    trim_to_window(hand_buffer, SNAP_WINDOW_FRAMES)
    contact = find_contact_point(hand_buffer, SNAP_CONTACT_THRESHOLD)
    if contact and released_fast_enough(hand_buffer, contact,
                                         SNAP_RELEASE_THRESHOLD,
                                         SNAP_RELEASE_WINDOW_FRAMES,
                                         SNAP_RELEASE_VELOCITY_THRESHOLD):
        clear(hand_buffer)
        return SnapEvent(...)
    return None
```

**Optional v1.1 robustness signal:** cross-check with a sudden downward/outward velocity spike of the middle fingertip itself (snaps typically end with the middle finger flicking away from the palm) as a secondary confirmation — not required for v1.

### 10.4 Handedness, Mirroring & Left/Right Mapping — read before wiring FR-5

Camera feeds are conventionally mirror-flipped for a natural selfie view (`cv2.flip(frame, 1)`), and MediaPipe's handedness label is known to behave inconsistently across setups depending on whether detection runs before or after that flip — this is a common, well-documented source of "my left/right gestures are swapped" bugs, and isn't safe to assume a fixed correct answer for in advance.

**Spec:**
1. Pick one consistent order and stick to it: flip the frame for mirrored display **first**, then run `HandLandmarker` on that already-flipped frame, so what MediaPipe calls "left"/"right" is being computed on the same image the user is looking at.
2. Add `INVERT_HANDEDNESS` (bool) to `config.py`, defaulting to `False`.
3. **Calibration step (do this explicitly in Phase 2, don't skip it):** run the app, snap your actual right hand, and check whether the filter advances or reverses. If it's backwards, flip `INVERT_HANDEDNESS` to `True` rather than touching detection code.

### 10.5 Event State Machines

**Filter navigation (per snap event):** stateless and immediate — each confirmed `SnapEvent` directly increments/decrements `filter_index` (mod `FILTER_COUNT`), subject only to a short per-hand `SNAP_COOLDOWN_MS` to prevent one physical snap's tracking noise from firing twice.

**Capture (fist → single save):** needs the same edge-triggering discipline as filter navigation, to satisfy FR-6:

| State | Entry condition | Behavior | Exit condition |
|---|---|---|---|
| IDLE | Default | Not capturing | Fist detected → ARMING |
| ARMING | Fist detected this frame | Hold counter increments | Counter ≥ `FIST_HOLD_FRAMES` → CAPTURE; fist drops → back to IDLE |
| CAPTURE | Hold threshold met | Fires capture exactly once, shows feedback | Immediately → COOLDOWN |
| COOLDOWN | Right after a capture | No new captures fire, even if fist is still held | Fist released **and** `CAPTURE_COOLDOWN_MS` elapsed → IDLE |

Requiring the fist to be *released* (not just cooldown elapsed) before re-arming is what stops a single held fist from producing a burst of photos.

---

## 11. Filter Engine — Detailed Spec

### 11.1 The 10 Filters

Each filter is a pure function `filter(frame: np.ndarray) -> np.ndarray`, same shape/dtype in and out, no shared state.

| # | Name | Look | Technique |
|---|---|---|---|
| 1 | **Noir** | High-contrast black & white, filmic | Grayscale conversion → S-curve contrast LUT → subtle Gaussian grain → back to 3-channel |
| 2 | **Vintage Film** | Warm, faded, slightly desaturated | 3×3 warm color matrix (`cv2.transform`, boost R, cut B) → desaturate ~15% in HSV → radial vignette mask → grain |
| 3 | **Arctic Blue** | Cold, crisp, blue-shifted | Color matrix boosting B/G, cutting R → contrast boost |
| 4 | **Golden Hour** | Warm glow on highlights | Luminance-masked highlight tone curve (boost R/G in bright regions) → Gaussian-blurred bright-pass layer screen-blended for bloom |
| 5 | **Cyberpunk Duotone** | Two-tone magenta-shadow / cyan-highlight | Grayscale luminance → custom 2-color gradient LUT (`cv2.LUT`) → contrast boost pre-mapping |
| 6 | **Sepia Classic** | Traditional warm sepia tone | Standard sepia 3×3 transform matrix via `cv2.transform`, clipped to [0,255] |
| 7 | **Soft Portrait** | Smooth skin, gentle glow | `cv2.bilateralFilter` edge-preserving smoothing blended ~60/40 with original, slight brightness/warmth lift |
| 8 | **Cartoon Sketch** | Flattened color + bold edges | Repeated bilateral filtering for color flattening + adaptive-threshold edge mask from blurred grayscale, edges composited over flattened color |
| 9 | **Cross-Process Pop** | Punchy, color-shifted highlights/shadows | Independent S-curve tone curve per R/G/B channel (different midpoints) + ~30% saturation boost |
| 10 | **Infrared Dream** | Surreal false-color | Channel remap (e.g., swap/boost R↔B or push G) + bloom on bright regions |

### 11.2 Filter Switching / Slideshow UX

- `filters/registry.py` holds an ordered list `[(name, function), ...]` of exactly 10 entries — this list, not any gesture code, is the single source of truth for order and count.
- On a confirmed snap event, update `filter_index = (filter_index ± 1) % FILTER_COUNT`.
- Cross-fade between the outgoing and incoming filtered frame over `FILTER_TRANSITION_MS` (simple `cv2.addWeighted` blend ramped over a few frames) so switches read as a slideshow transition, not a hard cut.
- Display the active filter's name in an on-screen label (e.g., bottom-third semi-transparent bar) that updates the instant `filter_index` changes.

### 11.3 Rendering Pipeline

```
1. Read raw camera frame, mirror-flip
2. Run HandLandmarker → landmarks + handedness
3. Update fist_detector and snap_detector per hand
4. Event manager resolves any fired events → update filter_index / trigger capture
5. Apply filters[filter_index] to the frame (cross-fading if mid-transition)
6. Draw filter-name label
7. If a capture just fired: draw brief flash/checkmark feedback
8. If DEBUG: draw landmarks + FPS + filter index + last event
9. Display frame
```

---

## 12. Capture Spec

- Trigger: CAPTURE state in §10.5 (edge-triggered, once per fist)
- What's saved: the **filtered** frame (post filter, pre debug-HUD/flash overlay) — captures should look like the polished output, not a debug view
- Location: `captures/` (create if missing; add to `.gitignore`)
- Filename pattern: `capture_{filter_name}_{YYYYMMDD_HHMMSS}.png` — human-readable and collision-resistant
- Feedback: a full-frame brief white flash (2–3 frames) and/or an on-screen "Saved ✓" label for `CAPTURE_FEEDBACK_MS`, non-blocking (doesn't pause the render loop)

---

## 13. Non-Functional Requirements

| Category | Requirement |
|---|---|
| Performance | ≥ 20 FPS target at 1280×720 on a mid-range laptop CPU. Snap detection specifically needs reasonably high frame rate to resolve the fast contact→release motion — if FPS drops too low, snap timing thresholds in §14 will need retuning, or drop capture resolution first |
| Reliability | No unhandled exceptions on: no camera, no hand in frame, hand leaving mid-gesture, rapid repeated snaps |
| Portability | Pure OpenCV + MediaPipe, no OS-specific calls |
| Maintainability | Adding an 11th filter requires only a new file + one registry entry — no gesture/camera code changes |
| Privacy | 100% local processing; captures are the *only* persisted output, saved locally, nothing uploaded |

---

## 14. Edge Cases & Error Handling

| Case | Handling |
|---|---|
| Slow, deliberate finger separation (not a real snap) | Rejected by the velocity check in §10.3 — must exceed `SNAP_RELEASE_VELOCITY_THRESHOLD` |
| Two rapid snaps close together | Second snap only registers after the first hand's buffer is cleared and any `SNAP_COOLDOWN_MS` elapses |
| Both hands snap in the same frame | Process independently; both events fire (net effect: index moves by the combination, e.g. +1 then -1 cancels out) — acceptable v1 behavior, no special-casing needed |
| Fist held for a long time | Exactly one capture, per the state machine in §10.5 — no repeat captures until release + cooldown |
| Left/right feels swapped during testing | Flip `INVERT_HANDEDNESS` in config per §10.4 — do not hack around it in detection logic |
| No camera detected | Clear error message, graceful exit, no stack trace |
| Model `.task` file missing | Fail fast at startup naming the missing file and its expected download URL |
| Zero hands in frame | No gesture events fire; app keeps running, current filter stays applied |
| Very low light | Both fist and snap detection degrade; document as a known limitation, not something v1 algorithmically compensates for |
| Disk write fails on capture | Catch the exception, show an on-screen "capture failed" message instead of crashing |

---

## 15. Configuration Reference

| Parameter | Purpose | Starting value |
|---|---|---|
| `CAMERA_INDEX` | Which camera device | `0` |
| `FRAME_WIDTH` / `FRAME_HEIGHT` | Capture resolution | `1280` / `720` |
| `MIN_HAND_DETECTION_CONFIDENCE` | HandLandmarker threshold | `0.6` |
| `MIN_HAND_PRESENCE_CONFIDENCE` | HandLandmarker threshold | `0.6` |
| `EXTENSION_RATIO_THRESHOLD` | Finger extended/curled cutoff | `1.3` |
| `SNAP_WINDOW_FRAMES` | Rolling buffer length for snap detection | `12` |
| `SNAP_CONTACT_THRESHOLD` | Normalized thumb–middle distance counted as "touching" | tune empirically, start small (e.g. `0.15`) |
| `SNAP_RELEASE_THRESHOLD` | Normalized distance counted as "released" | tune empirically, start larger (e.g. `0.4`) |
| `SNAP_RELEASE_WINDOW_FRAMES` | Max frames allowed between contact and release | `6` |
| `SNAP_RELEASE_VELOCITY_THRESHOLD` | Minimum separation speed to count as a snap, not a slow open | tune empirically |
| `SNAP_COOLDOWN_MS` | Per-hand cooldown after a confirmed snap | `400` |
| `INVERT_HANDEDNESS` | Calibration flip per §10.4 | `False` |
| `FIST_HOLD_FRAMES` | Frames required to confirm a fist | `4` |
| `CAPTURE_COOLDOWN_MS` | Minimum gap between captures | `800` |
| `FILTER_COUNT` | Number of filters in rotation | `10` |
| `FILTER_TRANSITION_MS` | Cross-fade duration on filter switch | `250` |
| `CAPTURE_DIR` | Where captured images are saved | `"captures/"` |
| `CAPTURE_FEEDBACK_MS` | Duration of the on-screen "Saved" flash | `400` |
| `SHOW_DEBUG_HUD` | Toggle landmark/FPS/state overlay | `False` |

---

## 16. File / Folder Structure

```
snapframe/
├── main.py
├── config.py
├── camera.py
├── detectors/
│   ├── __init__.py
│   └── hand_detector.py
├── gestures/
│   ├── __init__.py
│   ├── fist_detector.py
│   ├── snap_detector.py
│   └── event_manager.py
├── filters/
│   ├── __init__.py
│   ├── registry.py
│   ├── noir.py
│   ├── vintage_film.py
│   ├── arctic_blue.py
│   ├── golden_hour.py
│   ├── cyberpunk_duotone.py
│   ├── sepia_classic.py
│   ├── soft_portrait.py
│   ├── cartoon_sketch.py
│   ├── cross_process_pop.py
│   └── infrared_dream.py
├── capture/
│   ├── __init__.py
│   └── capture_manager.py
├── utils/
│   ├── __init__.py
│   └── geometry.py
├── captures/                  # gitignored — output images land here
├── models/
│   └── hand_landmarker.task
├── tests/
│   ├── test_fist_detector.py
│   ├── test_snap_detector.py
│   └── test_filters.py
├── requirements.txt
└── README.md
```

---

## 17. Testing Plan

**Unit tests (no camera needed):**
- `fist_detector`: synthetic landmark fixtures for closed fist vs. open palm vs. pointing vs. relaxed hand
- `snap_detector`: synthetic `(t, distance)` sequences —
  - fast contact→release **should** fire
  - slow gradual separation **should not** fire
  - small noisy oscillation around one distance value **should not** fire
  - two fast contact→release cycles in sequence should fire twice, correctly cooled down between
- `filters`: each of the 10 filter functions returns an array of matching shape/dtype; doesn't crash on all-black or all-white input frames

**Manual QA checklist (live webcam):**
- [ ] Right-hand snap advances the filter; label updates immediately
- [ ] Left-hand snap reverses the filter; label updates immediately
- [ ] Index wraps correctly at both ends (10 → 1, 1 → 10)
- [ ] Slow finger separation does *not* trigger a filter change
- [ ] Fist held for several seconds produces exactly one saved capture
- [ ] Captured file on disk visually matches what was on screen (correct filter baked in)
- [ ] Capture feedback appears and disappears without freezing the feed
- [ ] All 10 filters are visually distinct from each other side by side
- [ ] App survives hands leaving frame entirely
- [ ] Debug HUD toggle shows accurate FPS/state/last event

**Performance benchmark:** log rolling-average FPS; confirm filter application and gesture detection together don't drop below the §13 target.

---

## 18. Build Phases & Acceptance Criteria

| Phase | Scope | Done when |
|---|---|---|
| 0 — Setup | Repo scaffold, `requirements.txt`, download `hand_landmarker.task` | Imports run clean; model file present |
| 1 — Tracking foundation | Camera capture + `HandLandmarker`, draw raw landmarks | Live feed shows accurate landmark dots at ≥ 20 FPS |
| 2 — Gesture logic | `fist_detector`, `snap_detector`, `event_manager`, handedness calibration (§10.4) | Console logs correct fist/snap events matching real attempts; right/left mapping confirmed correct for the user, not just "technically detected" |
| 3 — Filter engine | All 10 filter functions + registry + manual keyboard left/right for testing (temporary, pre-gesture-wiring) | Cycling through all 10 filters with arrow keys shows 10 clearly distinct looks |
| 4 — Gesture-to-app wiring | Replace temporary keyboard controls with real snap events; wire fist → capture manager | Snap/fist gestures control the app end-to-end, matching FR-5/FR-6 exactly |
| 5 — Integration & polish | Debug HUD, capture feedback, filter-name label, cross-fade transitions, README | Manual QA checklist in §17 passes; README lets a stranger run it in minutes |
| 6 — Stretch (optional) | See §19 | N/A — post-MVP |

---

## 19. Future Enhancements (Stretch Goals)

- 🎞️ Filmstrip thumbnail strip showing all 10 filters at once, current one highlighted
- ⭐ Double-snap-in-a-row shortcut to jump to a "favorite" filter
- 📸 Burst-capture mode (hold fist longer → multiple shots)
- 🎥 Video recording per filter, not just stills
- 🎚️ Pinch-and-drag gesture to adjust a filter's intensity live
- 💾 Remember the last-used filter between app launches

---

## 20. Appendix

**Glossary**
- **Edge-triggered:** an action fires once on the transition into a state, not repeatedly while the state persists
- **Debounce:** requiring a signal to be stable/confirmed before acting on it, to filter out noise
- **LUT (lookup table):** a precomputed mapping applied per-pixel-value for fast color/tone transforms
- **Bloom:** a soft glow effect created by blending a blurred bright-pass layer back over the original image

**Reference documentation**
- MediaPipe Tasks — Hand Landmarker (Python): `https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python`
- MediaPipe Tasks — Gesture Recognizer (canned `Closed_Fist` category): `https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer`
