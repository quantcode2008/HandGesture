"""
config.py - Centralized configuration for WebShooter FX.

All tunable constants live here (PRD Section 14).
Values are sensible starting points; tune during Phase 5 & stretch goals.
"""

import os

# --- Paths ------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
HAND_LANDMARKER_MODEL = os.path.join(MODEL_DIR, "hand_landmarker.task")
SOUND_EFFECT_FILE = os.path.join(MODEL_DIR, "thwip.wav")

# --- Camera -----------------------------------------------------------------
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# --- Hand Detection --------------------------------------------------------
MAX_NUM_HANDS = 2
MIN_HAND_DETECTION_CONFIDENCE = 0.6
MIN_HAND_PRESENCE_CONFIDENCE = 0.6
MIN_HAND_TRACKING_CONFIDENCE = 0.6

# --- Gesture Classification ------------------------------------------------
EXTENSION_RATIO_THRESHOLD = 1.3

# --- State Machine Timing --------------------------------------------------
GESTURE_HOLD_FRAMES = 5       # ~165 ms at 30 FPS
MISS_TOLERANCE_FRAMES = 6
COOLDOWN_MS = 500

# --- Web Overlay ------------------------------------------------------------
WEB_WIDTH_SCALE = 1.4
WEB_HEIGHT_RATIO = 0.6
WEB_LINE_SPACING = 12
WEB_OVERLAY_OPACITY = 0.35
WEB_OVERLAY_COLOR = (200, 180, 255)   # BGR - pale pink/white

# --- Glitch Effect ----------------------------------------------------------
GLITCH_MAX_SHIFT_PX = 14
GLITCH_ATTACK_FRAMES = 3
GLITCH_DECAY_RATE = 0.85
GLITCH_BLEND = 0.8

# --- Sound Effect -----------------------------------------------------------
ENABLE_SOUND_EFFECTS = True

# --- Debug ------------------------------------------------------------------
SHOW_DEBUG_HUD = False
