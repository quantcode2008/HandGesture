# SnapFrame 📸✨

A real-time, touchless gesture-controlled camera filter application built in Python using OpenCV and MediaPipe Tasks API.

![Python 3.x](https://img.shields.io/badge/Python-3.x-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-5.0+-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks%20API-orange.svg)

---

## 🌟 Overview

**SnapFrame** transforms your computer's webcam into a slideshow camera filter booth controlled entirely by hand gestures:
- **Snap Right Hand 🤏➡️**: Advance to the next visual filter.
- **Snap Left Hand 🤏⬅️**: Go back to the previous visual filter (wraps around).
- **Make a Closed Fist ✊📸**: Capture a high-resolution photo with the active filter applied, saved automatically to your computer's Pictures folder.

---

## 🖼️ Saved Photo Location

Captured photos are automatically saved to a permanent, OS-appropriate **SnapFrame** folder in your system's Pictures directory:

- **Windows**: `C:\Users\<Username>\Pictures\SnapFrame` (`%USERPROFILE%\Pictures\SnapFrame`)
- **macOS**: `/Users/<Username>/Pictures/SnapFrame` (`~/Pictures/SnapFrame`)
- **Linux**: `/home/<Username>/Pictures/SnapFrame` (`~/Pictures/SnapFrame`)

Filename pattern: `capture_{filter_name}_{YYYYMMDD_HHMMSS}.png` (e.g., `capture_noir_20260809_183500.png`).

---

## 🎨 The 10 Visual Filters

SnapFrame features 10 distinct, high-quality image processing filters:

1. **Noir**: High-contrast filmic black & white with S-curve contrast and subtle film grain.
2. **Vintage Film**: Warm faded tones with desaturation and radial vignette.
3. **Arctic Blue**: Cold, crisp blue-shifted tones with contrast boost.
4. **Golden Hour**: Warm highlight tone curve with Gaussian bright-pass bloom glow.
5. **Cyberpunk Duotone**: Two-tone magenta shadows to electric cyan highlights gradient.
6. **Sepia Classic**: Traditional 3×3 sepia color transformation.
7. **Soft Portrait**: Bilateral skin smoothing (60/40 blend) with subtle warmth lift.
8. **Cartoon Sketch**: Bilateral color quantization overlaid with adaptive threshold ink outlines.
9. **Cross-Process Pop**: Independent channel S-curve LUTs with 30% saturation boost.
10. **Infrared Dream**: Surreal false-color foliage remap with soft highlight bloom.

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+
- A built-in or USB webcam

### 2. Installation & Setup

```bash
# Clone the repository
git clone https://github.com/quantcode2008/HandGesture.git
cd HandGesture

# Install dependencies
pip install -r requirements.txt
```

### 3. Run SnapFrame

```bash
python main.py
```

---

## 🎮 Controls

| Gesture / Key | Action |
|---------------|--------|
| **Right Hand Snap** | Advance to next filter (+1) with smooth cross-fade |
| **Left Hand Snap** | Reverse to previous filter (-1) with smooth cross-fade |
| **Make a Fist** | Capture photo frame (saved to `~/Pictures/SnapFrame/` with flash + badge feedback) |
| **`f`** | Toggle Filmstrip thumbnail carousel bar |
| **`d`** | Toggle Debug HUD (landmarks, FPS, filter index, gesture events) |
| **`q` or `ESC`** | Exit application |

---

## 🏗️ Architecture & Project Structure

```
snapframe/
├── main.py                     # Application loop, gesture dispatch & cross-fade render engine
├── config.py                   # Centralized configuration thresholds & ~/Pictures/SnapFrame path
├── camera.py                   # OpenCV VideoCapture wrapper (1280x720, mirrored selfie view)
├── detectors/
│   └── hand_detector.py        # MediaPipe Tasks HandLandmarker wrapper
├── gestures/
│   ├── fist_detector.py        # Orientation-invariant closed fist pose detector
│   ├── snap_detector.py        # Temporal thumb-middle motion detector (rolling buffer)
│   └── event_manager.py        # Handedness mapping, edge-triggered capture state machine
├── filters/
│   ├── registry.py             # Single source of truth for 10 filter functions
│   ├── noir.py                 # Filter 1: Noir
│   ├── vintage_film.py         # Filter 2: Vintage Film
│   ├── arctic_blue.py          # Filter 3: Arctic Blue
│   ├── golden_hour.py          # Filter 4: Golden Hour
│   ├── cyberpunk_duotone.py    # Filter 5: Cyberpunk Duotone
│   ├── sepia_classic.py        # Filter 6: Sepia Classic
│   ├── soft_portrait.py        # Filter 7: Soft Portrait
│   ├── cartoon_sketch.py       # Filter 8: Cartoon Sketch
│   ├── cross_process_pop.py    # Filter 9: Cross-Process Pop
│   └── infrared_dream.py       # Filter 10: Infrared Dream
├── capture/
│   └── capture_manager.py      # Filename generator, disk saver (~/Pictures/SnapFrame), feedback
├── ui/
│   └── filmstrip.py            # Live filmstrip thumbnail carousel bar (Stretch Goal §19)
├── models/
│   └── hand_landmarker.task    # MediaPipe HandLandmarker model file
└── tests/                      # Unit test suites (18 tests)
```

---

## 🧪 Unit Testing

Run all unit test suites:

```bash
python -m unittest discover tests -v
```

---

## 📜 License

MIT License. Personal portfolio / demonstration project.
