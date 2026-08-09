"""
filmstrip.py - Filmstrip thumbnail carousel bar (Stretch Goal §19).

Displays a live/cached thumbnail strip of all 10 visual filters simultaneously
across the top of the screen with the active filter highlighted in gold/cyan.
"""

from typing import List, Tuple
import cv2
import numpy as np

import config
from filters.registry import FILTER_REGISTRY


class FilmstripManager:
    """Manages fast thumbnail caching and renders the filmstrip carousel bar."""

    def __init__(self):
        self.cached_thumbnails: List[np.ndarray] = []
        self.frame_counter: int = 0
        self.last_active_index: int = -1

    def _generate_thumbnails(self, frame: np.ndarray):
        """Downscale frame and run through all 10 filters to produce small thumbnails."""
        tw, th = config.FILMSTRIP_THUMB_WIDTH, config.FILMSTRIP_THUMB_HEIGHT

        # Downscale source frame for fast processing (160x90)
        small_src = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_NEAREST)

        thumbnails = []
        for name, filter_func in FILTER_REGISTRY:
            try:
                filtered_small = filter_func(small_src)
                thumb = cv2.resize(filtered_small, (tw, th), interpolation=cv2.INTER_LINEAR)
            except Exception:
                thumb = np.zeros((th, tw, 3), dtype=np.uint8)
            thumbnails.append(thumb)

        self.cached_thumbnails = thumbnails

    def render(self, frame: np.ndarray, active_index: int) -> np.ndarray:
        """
        Composite filmstrip thumbnail carousel bar onto top of frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image to render onto.
        active_index : int
            Index of currently active filter (0..9).

        Returns
        -------
        np.ndarray
            Frame with filmstrip overlay applied.
        """
        if not config.SHOW_FILMSTRIP:
            return frame

        self.frame_counter += 1

        # Regenerate cached thumbnails periodically or immediately on filter change
        if (
            not self.cached_thumbnails
            or self.frame_counter % config.FILMSTRIP_UPDATE_INTERVAL_FRAMES == 0
            or active_index != self.last_active_index
        ):
            self._generate_thumbnails(frame)
            self.last_active_index = active_index

        fh, fw, _ = frame.shape
        tw, th = config.FILMSTRIP_THUMB_WIDTH, config.FILMSTRIP_THUMB_HEIGHT

        margin_y = 12
        padding_x = 8
        total_width = len(FILTER_REGISTRY) * tw + (len(FILTER_REGISTRY) - 1) * padding_x
        start_x = max(10, (fw - total_width) // 2)

        bar_h = th + margin_y * 2 + 10

        # Semi-transparent dark background bar at top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (fw, bar_h), (12, 12, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        for i, thumb in enumerate(self.cached_thumbnails):
            tx1 = start_x + i * (tw + padding_x)
            ty1 = margin_y
            tx2 = tx1 + tw
            ty2 = ty1 + th

            if tx2 > fw:
                break

            # Composite thumbnail image
            frame[ty1:ty2, tx1:tx2] = thumb

            if i == active_index:
                # Active Filter: Electric Cyan / Gold glowing double border
                cv2.rectangle(frame, (tx1 - 2, ty1 - 2), (tx2 + 2, ty2 + 2), (0, 230, 255), 3, cv2.LINE_AA)
                cv2.rectangle(frame, (tx1 - 4, ty1 - 4), (tx2 + 4, ty2 + 4), (255, 200, 0), 1, cv2.LINE_AA)

                # Active indicator dot / badge
                cv2.circle(frame, (tx1 + tw // 2, ty2 + 6), 4, (0, 230, 255), -1, cv2.LINE_AA)
            else:
                # Inactive Filter: Subtle white border
                cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (180, 180, 180), 1, cv2.LINE_AA)

            # Filter index number badge on thumbnail
            cv2.putText(
                frame, str(i + 1),
                (tx1 + 4, ty1 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA,
            )
            cv2.putText(
                frame, str(i + 1),
                (tx1 + 4, ty1 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA,
            )

        return frame
