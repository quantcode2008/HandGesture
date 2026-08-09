"""
capture_manager.py - Frame capture and feedback manager (PRD Section 12).

Saves filtered frames to disk in the user's permanent Pictures directory (~/Pictures/SnapFrame):
  ~/Pictures/SnapFrame/capture_{filter_name}_{YYYYMMDD_HHMMSS}.png

Renders non-blocking on-screen capture feedback (white flash & 'Saved ✓' badge).
"""

import datetime
import os
import re
from typing import Optional
import cv2
import numpy as np

import config


def _slugify_filter_name(name: str) -> str:
    """Convert filter name like '1. Vintage Film' to 'vintage_film'."""
    # Strip leading digits and dots
    cleaned = re.sub(r"^\d+[\.\s]*", "", name).strip().lower()
    # Replace spaces and non-alphanumeric chars with underscore
    slug = re.sub(r"[^\w]+", "_", cleaned).strip("_")
    return slug or "filtered"


class CaptureManager:
    """Manages saving filtered frames to disk and rendering non-blocking visual feedback."""

    def __init__(self, capture_dir: str = config.CAPTURE_DIR):
        self.capture_dir = capture_dir
        os.makedirs(self.capture_dir, exist_ok=True)

        self.last_saved_path: Optional[str] = None
        self.flash_until: float = 0.0
        self.banner_until: float = 0.0
        self.error_until: float = 0.0
        self.error_msg: str = ""

    def save_capture(self, filtered_frame: np.ndarray, filter_name: str, now: float) -> Optional[str]:
        """
        Save the filtered frame (post-filter, pre-HUD) to disk.

        Parameters
        ----------
        filtered_frame : np.ndarray
            The filtered frame in BGR format.
        filter_name : str
            Name of the active filter (e.g. '1. Noir').
        now : float
            Current timestamp in seconds.

        Returns
        -------
        Optional[str]
            Saved file path if successful, None if write failed.
        """
        slug = _slugify_filter_name(filter_name)
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{slug}_{timestamp_str}.png"
        filepath = os.path.join(self.capture_dir, filename)

        try:
            # Ensure folder exists
            os.makedirs(self.capture_dir, exist_ok=True)
            success = cv2.imwrite(filepath, filtered_frame)
            if not success:
                raise IOError(f"cv2.imwrite returned False for {filepath}")

            self.last_saved_path = filepath
            feedback_sec = config.CAPTURE_FEEDBACK_MS / 1000.0
            self.flash_until = now + 0.08        # ~2.5 frames flash
            self.banner_until = now + feedback_sec
            print(f"[CAPTURE] Saved: {filepath}")
            return filepath

        except Exception as e:
            print(f"[ERROR] Capture save failed: {e}")
            self.error_msg = "Capture Failed!"
            self.error_until = now + 2.0
            return None

    def render_feedback(self, frame: np.ndarray, now: float) -> np.ndarray:
        """
        Apply visual capture feedback (flash overlay and 'Saved ✓' badge) onto frame.

        Parameters
        ----------
        frame : np.ndarray
            The output frame to render feedback on top of.
        now : float
            Current timestamp in seconds.

        Returns
        -------
        np.ndarray
            Frame with feedback overlays applied.
        """
        h, w, _ = frame.shape

        # 1. White Flash Overlay (lasts ~80ms)
        if now < self.flash_until:
            white = np.full((h, w, 3), 255, dtype=np.uint8)
            frame = cv2.addWeighted(frame, 0.35, white, 0.65, 0)

        # 2. 'Saved ✓' Badge Overlay (lasts CAPTURE_FEEDBACK_MS)
        if now < self.banner_until:
            badge_text = "Saved ✓"
            # Draw semi-transparent pill in top-right corner
            badge_w, badge_h = 160, 45
            x1 = w - badge_w - 20
            y1 = 20
            x2 = w - 20
            y2 = y1 + badge_h

            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 180, 0), -1)
            cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

            cv2.putText(
                frame, badge_text,
                (x1 + 25, y1 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
            )

        # 3. Error Banner if write failed
        if now < self.error_until:
            cv2.putText(
                frame, self.error_msg,
                (w // 2 - 100, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA,
            )

        return frame
