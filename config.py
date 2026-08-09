"""
config.py - Centralized configuration for SnapFrame.

All tunable constants live here (SnapFrame PRD Section 15 & 19 Stretch Goals).
Values are tuned based on real-time empirical webcam gesture testing.
"""

import os

# --- Paths ------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
HAND_LANDMARKER_MODEL = os.path.join(MODEL_DIR, "hand_landmarker.task")

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

# --- Snap Motion Detection (Empirically Tuned) ------------------------------
SNAP_WINDOW_FRAMES = 12
SNAP_CONTACT_THRESHOLD = 0.18           # Thumb & middle finger touching
SNAP_RELEASE_THRESHOLD = 0.38           # Finger separation threshold
SNAP_RELEASE_WINDOW_FRAMES = 6          # Max frames allowed between contact & release
SNAP_RELEASE_VELOCITY_THRESHOLD = 0.80  # Minimum separation velocity (norm_dist/sec)
SNAP_COOLDOWN_MS = 400                  # Per-hand cooldown to avoid double triggers

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
