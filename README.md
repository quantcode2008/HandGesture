# WebShooter FX 🕸️⚡

Real-time gesture-triggered AR "Web-Shooter" camera filter built in Python using OpenCV and MediaPipe Tasks.

![Python 3.x](https://img.shields.io/badge/Python-3.x-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-5.0+-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks%20API-orange.svg)

---

## 🌟 Overview

**WebShooter FX** tracks hand geometry in real-time through your computer's webcam. Whenever you hold up your hands in the classic "web-shooter" pose (index and pinky fingers extended, middle and ring fingers curled), the filter fires a stylized visual effect anchored to your hand movement:
- **Cross-Hatch Web Mesh Overlay**: Rotates, scales, and repositions live based on your single-hand orientation or inter-hand distance.
- **RGB Chromatic-Aberration Glitch Pulse**: A localized color-channel shift pulse that triggers dynamically on gesture confirmation.

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+
- A built-in or USB webcam

### 2. Installation & Setup

Clone the repository and install dependencies:
```bash
git clone https://github.com/your-username/webshooter-fx.git
cd webshooter-fx

# Install Python requirements
pip install -r requirements.txt
```

### 3. Run the Filter

```bash
python main.py
```

---

## 🎮 Controls

| Key | Action |
|-----|--------|
| `d` | Toggle Debug HUD (shows/hides 21 raw hand landmarks, skeleton lines, FPS, and state machine transitions) |
| `q` or `ESC` | Exit application |

---

## 🏗️ Architecture & Module Breakdown

```
webshooter-fx/
├── main.py                     # Main application loop & composite layer manager
├── config.py                   # Centralized configuration parameters
├── camera.py                   # OpenCV VideoCapture camera wrapper
├── detectors/
│   ├── hand_detector.py        # MediaPipe Tasks HandLandmarker wrapper
│   └── face_detector.py        # (v1.1 stub) FaceLandmarker wrapper
├── gestures/
│   ├── gesture_classifier.py   # Orientation-invariant distance-ratio gesture classifier
│   └── state_machine.py        # Debounced state machine (IDLE -> ARMING -> ACTIVE -> COOLDOWN)
├── effects/
│   ├── web_overlay.py          # Cross-hatch web pattern generator & alpha compositor
│   └── glitch.py               # RGB chromatic-aberration channel-shift pulse effect
├── utils/
│   ├── geometry.py             # Distance, angle, midpoint, & rotation matrix helpers
│   └── smoothing.py            # Exponential moving average & rolling buffer utilities
├── models/
│   └── hand_landmarker.task    # MediaPipe HandLandmarker model weights
└── tests/
    ├── test_geometry.py        # Unit tests for geometry helpers
    ├── test_gesture_classifier.py # Fixture-based gesture classifier unit tests
    ├── test_web_overlay.py     # Web overlay renderer unit tests
    └── test_glitch.py          # Glitch effect unit tests
```

---

## 🧪 Running Unit Tests

Run all unit tests across geometry, gesture classification, web overlay, and glitch effects:

```bash
python -m unittest discover tests -v
```

---

## ⚙️ Configuration (`config.py`)

All thresholds, colors, and timing parameters can be tuned in `config.py`:
- `GESTURE_HOLD_FRAMES` (Default: `5`): Consecutive frames required to activate the gesture.
- `MISS_TOLERANCE_FRAMES` (Default: `6`): Consecutive missed frames tolerated before entering cooldown.
- `WEB_WIDTH_SCALE` (Default: `1.4`): Web overlay width multiplier relative to hand size / inter-hand distance.
- `WEB_LINE_SPACING` (Default: `12`): Spacing between cross-hatch web lines.
- `GLITCH_MAX_SHIFT_PX` (Default: `14`): Maximum pixel offset for RGB chromatic aberration.
- `GLITCH_DECAY_RATE` (Default: `0.85`): Per-frame decay rate of the glitch pulse.
- `SHOW_DEBUG_HUD` (Default: `False`): Initial state of the debug HUD.

---

## 📜 License

MIT License. Personal, non-commercial fan-inspired portfolio project.
