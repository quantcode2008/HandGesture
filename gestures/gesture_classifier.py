"""
gesture_classifier.py - Rule-based web-shooter gesture classifier.

Uses the orientation-invariant distance-ratio method from PRD Section 10.3:
a finger is "extended" if its tip is meaningfully farther from the wrist
than its MCP (knuckle) joint is.  This avoids the naive y-coordinate
comparison that breaks when the hand is rotated or upside-down.

Web-shooter pose (Section 10.4):
  - Index:  EXTENDED
  - Middle: CURLED
  - Ring:   CURLED
  - Pinky:  EXTENDED
  - Thumb:  soft/optional check (not hard-required - thumb tracking is noisy)
"""

from utils.geometry import euclidean
import config

# --- Landmark indices (PRD Section 10.1) ------------------------------------------
WRIST = 0

THUMB_TIP = 4
THUMB_MCP = 2
INDEX_MCP = 5
PINKY_MCP = 17

INDEX_TIP = 8
INDEX_MCP_IDX = 5

MIDDLE_TIP = 12
MIDDLE_MCP = 9

RING_TIP = 16
RING_MCP = 13

PINKY_TIP = 20
PINKY_MCP_IDX = 17


def is_finger_extended(
    landmarks,
    tip_idx: int,
    mcp_idx: int,
    wrist_idx: int = WRIST,
    ratio_threshold: float = config.EXTENSION_RATIO_THRESHOLD,
) -> bool:
    """Orientation-invariant finger extension check (PRD Section 10.3).

    A finger is "extended" when its tip is farther from the wrist
    than its MCP joint is, by at least *ratio_threshold*.

    Parameters
    ----------
    landmarks : list
        The 21 MediaPipe hand landmarks (NormalizedLandmark objects).
    tip_idx, mcp_idx, wrist_idx : int
        Landmark indices for the fingertip, MCP joint, and wrist.
    ratio_threshold : float
        tip-to-wrist distance must exceed mcp-to-wrist x this value.
    """
    tip = landmarks[tip_idx]
    mcp = landmarks[mcp_idx]
    wrist = landmarks[wrist_idx]

    d_tip_wrist = euclidean(tip, wrist)
    d_mcp_wrist = euclidean(mcp, wrist)

    # Guard against division by zero if mcp is exactly at wrist
    if d_mcp_wrist < 1e-9:
        return False

    return d_tip_wrist > d_mcp_wrist * ratio_threshold


def is_thumb_tucked(landmarks, tolerance: float = 0.8) -> bool:
    """Check whether the thumb tip is tucked toward the palm (PRD Section 10.3).

    Tucked thumb sits closer to the index-side knuckle than a fully
    extended thumb would.  This is a *soft* signal - not hard-required.
    """
    thumb_tip = landmarks[THUMB_TIP]
    index_mcp = landmarks[INDEX_MCP]
    pinky_mcp = landmarks[PINKY_MCP]

    return euclidean(thumb_tip, index_mcp) < euclidean(pinky_mcp, index_mcp) * tolerance


def is_web_shooter_pose(landmarks) -> bool:
    """Return True if the hand landmarks match the web-shooter gesture (PRD Section 10.4).

    Required:
      - Index finger EXTENDED
      - Pinky EXTENDED
      - Middle finger CURLED (not extended)
      - Ring finger CURLED (not extended)

    Thumb is soft-checked only - not hard-required per PRD Section 10.3 note:
    "Treat the thumb check as a soft/optional signal in v1 ... don't
    hard-require it, or the gesture will feel finicky to perform."
    """
    index_ext = is_finger_extended(landmarks, INDEX_TIP, INDEX_MCP_IDX)
    middle_ext = is_finger_extended(landmarks, MIDDLE_TIP, MIDDLE_MCP)
    ring_ext = is_finger_extended(landmarks, RING_TIP, RING_MCP)
    pinky_ext = is_finger_extended(landmarks, PINKY_TIP, PINKY_MCP_IDX)

    return index_ext and pinky_ext and not middle_ext and not ring_ext
