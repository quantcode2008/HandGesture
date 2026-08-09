"""
web_overlay.py - Cross-hatch web pattern overlay renderer.

Implements single-hand and two-hand web overlay modes per PRD Section 11.1.
- Single-hand mode: Anchors to active palm center, rotates with hand orientation,
  sized dynamically by hand scale or config.
- Two-hand mode: Anchors to midpoint between active palm centers, rotates by inter-hand angle,
  scaled by inter-hand distance * WEB_WIDTH_SCALE.
"""

import math
from typing import List, Optional, Tuple
import cv2
import numpy as np

import config
from utils.geometry import angle_between, compute_palm_center, euclidean_2d, midpoint


def create_web_pattern_canvas(width: int, height: int) -> np.ndarray:
    """
    Generate a transparent RGBA cross-hatch web pattern canvas.

    Parameters
    ----------
    width, height : int
        Dimensions of the pattern region.

    Returns
    -------
    np.ndarray
        RGBA image of shape (height, width, 4).
    """
    width = max(20, int(width))
    height = max(20, int(height))
    canvas = np.zeros((height, width, 4), dtype=np.uint8)

    b, g, r = config.WEB_OVERLAY_COLOR
    color = (int(b), int(g), int(r), 255)
    spacing = max(4, int(config.WEB_LINE_SPACING))

    diag = int(math.hypot(width, height))
    
    # 45 degree cross-hatch lines
    for offset in range(-diag, diag + spacing, spacing):
        # Line set 1: top-left to bottom-right slope
        p1 = (offset, 0)
        p2 = (offset + height, height)
        cv2.line(canvas, p1, p2, color, thickness=2, lineType=cv2.LINE_AA)

        # Line set 2: top-right to bottom-left slope
        p3 = (offset, height)
        p4 = (offset + height, 0)
        cv2.line(canvas, p3, p4, color, thickness=2, lineType=cv2.LINE_AA)

    # Outer web boundary ellipse to give it a clean Spider-Man web mesh shape
    center = (width // 2, height // 2)
    axes = (width // 2 - 2, height // 2 - 2)
    if axes[0] > 0 and axes[1] > 0:
        cv2.ellipse(canvas, center, axes, 0, 0, 360, color, thickness=2, lineType=cv2.LINE_AA)

    return canvas


def render_web_overlay(
    frame: np.ndarray,
    hand_landmarks_list: List[List],
    active_mask: List[bool],
) -> np.ndarray:
    """
    Composite web pattern overlay onto frame based on active hand geometry.

    Parameters
    ----------
    frame : np.ndarray
        BGR camera frame (h, w, 3).
    hand_landmarks_list : list of landmark lists
        Detected hand landmarks per hand.
    active_mask : list of bool
        Parallel list indicating if each hand is in ACTIVE state.

    Returns
    -------
    np.ndarray
        Frame with alpha-composited web overlay.
    """
    active_indices = [i for i, active in enumerate(active_mask) if active and i < len(hand_landmarks_list)]
    if not active_indices:
        return frame

    fh, fw, _ = frame.shape

    # Extract palm centers in pixel coordinates
    palm_centers = []
    for idx in active_indices:
        lm = hand_landmarks_list[idx]
        nx, ny = compute_palm_center(lm)
        palm_centers.append((nx * fw, ny * fh))

    if len(active_indices) >= 2:
        # ─── Two-Hand Mode ──────────────────────────────────────────────────
        p1, p2 = palm_centers[0], palm_centers[1]
        anchor_x = (p1[0] + p2[0]) / 2.0
        anchor_y = (p1[1] + p2[1]) / 2.0

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        inter_hand_dist = math.hypot(dx, dy)
        angle_rad = math.atan2(dy, dx)

        width = max(100, int(inter_hand_dist * config.WEB_WIDTH_SCALE))
        height = max(60, int(width * config.WEB_HEIGHT_RATIO))

    else:
        # ─── Single-Hand Mode ───────────────────────────────────────────────
        idx = active_indices[0]
        lm = hand_landmarks_list[idx]
        anchor_x, anchor_y = palm_centers[0]

        # Orientation angle from wrist (0) to middle MCP (9)
        w_x, w_y = lm[0].x * fw, lm[0].y * fh
        m_x, m_y = lm[9].x * fw, lm[9].y * fh
        angle_rad = math.atan2(m_y - w_y, m_x - w_x) - (math.pi / 2.0)

        # Scale width based on wrist-to-middle distance
        hand_size = math.hypot(m_x - w_x, m_y - w_y)
        width = max(120, int(hand_size * 2.8 * config.WEB_WIDTH_SCALE))
        height = max(70, int(width * config.WEB_HEIGHT_RATIO))

    # Clamp width/height to frame bounds
    width = min(width, fw * 2)
    height = min(height, fh * 2)

    # Generate unrotated pattern
    pat_rgba = create_web_pattern_canvas(width, height)

    # Prepare padded canvas for rotation without clipping corners
    diag = int(math.ceil(math.hypot(width, height)))
    padded = np.zeros((diag, diag, 4), dtype=np.uint8)

    pad_x = (diag - width) // 2
    pad_y = (diag - height) // 2
    padded[pad_y:pad_y + height, pad_x:pad_x + width] = pat_rgba

    # Rotate around padded canvas center
    center_pt = (diag / 2.0, diag / 2.0)
    rot_mat = cv2.getRotationMatrix2D(center_pt, math.degrees(angle_rad), 1.0)
    rotated_rgba = cv2.warpAffine(padded, rot_mat, (diag, diag), flags=cv2.INTER_LINEAR)

    # Determine bounding box on target frame
    top_left_x = int(round(anchor_x - diag / 2.0))
    top_left_y = int(round(anchor_y - diag / 2.0))

    # Frame clipping bounds
    src_x1 = max(0, -top_left_x)
    src_y1 = max(0, -top_left_y)
    src_x2 = min(diag, fw - top_left_x)
    src_y2 = min(diag, fh - top_left_y)

    dst_x1 = max(0, top_left_x)
    dst_y1 = max(0, top_left_y)
    dst_x2 = min(fw, top_left_x + diag)
    dst_y2 = min(fh, top_left_y + diag)

    if src_x2 <= src_x1 or src_y2 <= src_y1 or dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
        return frame

    # Extract overlapping regions
    overlay_crop = rotated_rgba[src_y1:src_y2, src_x1:src_x2]
    frame_crop = frame[dst_y1:dst_y2, dst_x1:dst_x2]

    # Alpha blend
    alpha = (overlay_crop[:, :, 3] / 255.0) * config.WEB_OVERLAY_OPACITY
    alpha = alpha[:, :, np.newaxis]  # shape (h, w, 1)

    overlay_bgr = overlay_crop[:, :, :3]
    blended = (frame_crop * (1.0 - alpha) + overlay_bgr * alpha).astype(np.uint8)

    frame[dst_y1:dst_y2, dst_x1:dst_x2] = blended
    return frame
