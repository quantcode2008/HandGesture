"""
config.py - Centralized configuration for SnapFrame.

All tunable constants live here (SnapFrame PRD Section 15 & 19 Stretch Goals).
Values are tuned based on real-time empirical webcam gesture testing.
"""

import os

# --- Paths ------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
HAND_LANDMARKER_MODEL = os.path.join(MODEL_DIR, "hand_landmarker.task")
FACE_LANDMARKER_MODEL = os.path.join(MODEL_DIR, "face_landmarker.task")


# Permanent OS-appropriate user Pictures directory (~/Pictures/SnapFrame)
PICTURES_DIR = os.path.join(os.path.expanduser("~"), "Pictures")
CAPTURE_DIR = os.path.join(PICTURES_DIR, "SnapFrame")

# --- Camera -----------------------------------------------------------------
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# --- Hand Detection --------------------------------------------------------
MAX_NUM_HANDS = 2
MIN_HAND_DETECTION_CONFIDENCE = 0.6
MIN_HAND_PRESENCE_CONFIDENCE = 0.6
MIN_HAND_TRACKING_CONFIDENCE = 0.6

# --- Finger Extension & Pose -----------------------------------------------
EXTENSION_RATIO_THRESHOLD = 1.3

# --- Snap Motion Detection (Fast 2-Phase Snap <100ms) -----------------------
SNAP_PINCH_THRESHOLD = 0.22             # Thumb & middle finger touching (normalized distance)
SNAP_RELEASE_THRESHOLD = 0.38           # Finger separation threshold on release
SNAP_MIN_PINCH_FRAMES = 2               # Minimum consecutive frames in pinch to reject single-frame jitter
SNAP_RELEASE_WINDOW_FRAMES = 7          # Max frames allowed for release phase (rolling window ~6-8 frames)
SNAP_VELOCITY_THRESHOLD = 0.045         # Minimum normalized distance change per frame during release
SNAP_COOLDOWN_MS = 400                  # Per-hand cooldown to avoid double triggers
LANDMARK_EMA_ALPHA = 0.75               # Responsive EMA landmark smoothing factor (0.0 to 1.0)

# Backwards-compatibility aliases
SNAP_CONTACT_THRESHOLD = SNAP_PINCH_THRESHOLD
SNAP_WINDOW_FRAMES = SNAP_RELEASE_WINDOW_FRAMES
SNAP_RELEASE_VELOCITY_THRESHOLD = 0.80  # Legacy per-second alias


# --- Handedness Calibration -----------------------------------------------
INVERT_HANDEDNESS = False               # Set True if left/right snap directions are inverted

# --- Fist & Capture ---------------------------------------------------------
FIST_HOLD_FRAMES = 4                    # Consecutive frames to confirm a fist
CAPTURE_COOLDOWN_MS = 800               # Cooldown gap between photo captures
CAPTURE_FEEDBACK_MS = 400               # Duration of 'Saved ✓' badge overlay

# --- Filter Engine ----------------------------------------------------------
FILTER_COUNT = 10
FILTER_TRANSITION_MS = 250              # Smooth cross-fade duration on filter switch

# --- Filmstrip Thumbnail Strip (Stretch Goal §19) ---------------------------
SHOW_FILMSTRIP = True
FILMSTRIP_THUMB_WIDTH = 96
FILMSTRIP_THUMB_HEIGHT = 54
FILMSTRIP_UPDATE_INTERVAL_FRAMES = 6

# --- Debug ------------------------------------------------------------------
SHOW_DEBUG_HUD = False                  # Initial Debug HUD state (toggle with 'd')

# --- Window & Display -------------------------------------------------------
WINDOW_NAME = "SnapFrame"
START_MAXIMIZED = True                  # Automatically maximize window on launch to fill screen

# --- Facial Emotion Detection (MediaPipe FaceLandmarker Tasks API) ----------
ENABLE_EMOTION_DETECTION = True
SHOW_EMOTION_BADGE = True               # Initial overlay badge visibility (toggle with 'e')
EMOTION_FRAME_INTERVAL = 2              # Process face detection every Nth frame (keeps FPS > 20)
EMOTION_SMOOTHING_WINDOW = 10           # Frame window for majority/EMA temporal smoothing
EMOTION_MIN_CONFIDENCE = 0.28           # Minimum score to classify an active emotion vs Neutral
EMOTION_NEUTRAL_LABEL = "Neutral"

# Rule-based blendshape scoring rules and weights
EMOTION_RULES = {
    "Happy": {
        "positive": [
            ("mouthSmileLeft", 0.50),
            ("mouthSmileRight", 0.50),
            ("cheekSquintLeft", 0.15),
            ("cheekSquintRight", 0.15),
        ],
        "negative": [
            ("mouthFrownLeft", 0.40),
            ("mouthFrownRight", 0.40),
            ("browDownLeft", 0.25),
            ("browDownRight", 0.25),
        ],
    },
    "Sad": {
        "positive": [
            ("mouthFrownLeft", 0.45),
            ("mouthFrownRight", 0.45),
            ("browInnerUp", 0.35),
            ("mouthShrugLower", 0.20),
        ],
        "negative": [
            ("mouthSmileLeft", 0.60),
            ("mouthSmileRight", 0.60),
        ],
    },
    "Angry": {
        "positive": [
            ("browDownLeft", 0.45),
            ("browDownRight", 0.45),
            ("mouthPressLeft", 0.20),
            ("mouthPressRight", 0.20),
            ("noseSneerLeft", 0.15),
            ("noseSneerRight", 0.15),
        ],
        "negative": [
            ("mouthSmileLeft", 0.50),
            ("mouthSmileRight", 0.50),
            ("jawOpen", 0.30),
            ("browInnerUp", 0.25),
        ],
    },
    "Surprised": {
        "positive": [
            ("jawOpen", 0.45),
            ("eyeWideLeft", 0.25),
            ("eyeWideRight", 0.25),
            ("browInnerUp", 0.25),
            ("browOuterUpLeft", 0.15),
            ("browOuterUpRight", 0.15),
        ],
        "negative": [
            ("mouthSmileLeft", 0.20),
            ("mouthSmileRight", 0.20),
            ("browDownLeft", 0.20),
            ("browDownRight", 0.20),
        ],
    },
    "Fear": {
        "positive": [
            ("eyeWideLeft", 0.35),
            ("eyeWideRight", 0.35),
            ("browInnerUp", 0.35),
            ("mouthStretchLeft", 0.20),
            ("mouthStretchRight", 0.20),
            ("jawOpen", 0.15),
        ],
        "negative": [
            ("mouthSmileLeft", 0.40),
            ("mouthSmileRight", 0.40),
        ],
    },
    "Disgust": {
        "positive": [
            ("noseSneerLeft", 0.45),
            ("noseSneerRight", 0.45),
            ("mouthUpperUpLeft", 0.25),
            ("mouthUpperUpRight", 0.25),
            ("browDownLeft", 0.15),
            ("browDownRight", 0.15),
        ],
        "negative": [
            ("mouthSmileLeft", 0.40),
            ("mouthSmileRight", 0.40),
        ],
    },
}

# Emotion display metadata and auto-suggested filter mappings
EMOTION_METADATA = {
    "Happy": {"label": "Happy :)", "filter": "Golden Hour", "color": (0, 220, 255)},
    "Sad": {"label": "Sad :(", "filter": "Noir", "color": (220, 150, 100)},
    "Angry": {"label": "Angry >:(", "filter": "Cyberpunk Duotone", "color": (50, 50, 240)},
    "Surprised": {"label": "Surprised :O", "filter": "Cartoon Sketch", "color": (0, 255, 120)},
    "Fear": {"label": "Fear :S", "filter": "Infrared Dream", "color": (200, 100, 255)},
    "Disgust": {"label": "Disgust :P", "filter": "Vintage Film", "color": (50, 180, 50)},
    "Neutral": {"label": "Neutral :|", "filter": "Soft Portrait", "color": (200, 200, 200)},
}


