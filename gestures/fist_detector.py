"""
fist_detector.py - Closed fist pose detector.

Detects whether a set of 21 MediaPipe hand landmarks forms a closed fist per PRD Section 10.2:
- All four fingers (index, middle, ring, pinky) are curled (not extended).
- Thumb is tucked near the palm.
Uses orientation-invariant distance-ratio checks.
"""

from typing import List, Any
import config
from utils.geometry import euclidean

# Landmark indices
WRIST = 0

THUMB_TIP = 4
THUMB_MCP = 2
INDEX_MCP = 5
PINKY_MCP = 17

INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20

INDEX_MCP_IDX = 5
MIDDLE_MCP_IDX = 9
RING_MCP_IDX = 13
PINKY_MCP_IDX = 17


def is_finger_extended(
    landmarks: List[Any],
    tip_idx: int,
    mcp_idx: int,
    wrist_idx: int = WRIST,
    ratio_threshold: float = config.EXTENSION_RATIO_THRESHOLD,
) -> bool:
    """Check if finger tip is farther from wrist than its MCP joint by ratio_threshold."""
    tip = landmarks[tip_idx]
    mcp = landmarks[mcp_idx]
    wrist = landmarks[wrist_idx]

    d_tip_wrist = euclidean(tip, wrist)
    d_mcp_wrist = euclidean(mcp, wrist)

    if d_mcp_wrist < 1e-9:
        return False

    return d_tip_wrist > d_mcp_wrist * ratio_threshold


def is_thumb_tucked(landmarks: List[Any], tolerance: float = 0.95) -> bool:
    """Check if thumb tip is tucked toward palm."""
    thumb_tip = landmarks[THUMB_TIP]
    index_mcp = landmarks[INDEX_MCP]
    pinky_mcp = landmarks[PINKY_MCP]

    # A tucked thumb tip sits closer to the index knuckle than pinky-to-index width * tolerance
    return euclidean(thumb_tip, index_mcp) < euclidean(pinky_mcp, index_mcp) * tolerance


def is_fist(landmarks: List[Any], ratio_threshold: float = config.EXTENSION_RATIO_THRESHOLD) -> bool:
    """
    Return True if the 21 hand landmarks form a closed fist (PRD Section 10.2).
    - Index, middle, ring, pinky must all be curled (not extended).
    - Thumb must be tucked.
    """
    index_ext = is_finger_extended(landmarks, INDEX_TIP, INDEX_MCP_IDX, ratio_threshold=ratio_threshold)
    middle_ext = is_finger_extended(landmarks, MIDDLE_TIP, MIDDLE_MCP_IDX, ratio_threshold=ratio_threshold)
    ring_ext = is_finger_extended(landmarks, RING_TIP, RING_MCP_IDX, ratio_threshold=ratio_threshold)
    pinky_ext = is_finger_extended(landmarks, PINKY_TIP, PINKY_MCP_IDX, ratio_threshold=ratio_threshold)
    thumb_tucked = is_thumb_tucked(landmarks)

    return not (index_ext or middle_ext or ring_ext or pinky_ext) and thumb_tucked
