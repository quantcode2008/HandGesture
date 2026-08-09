"""
glitch.py - Chromatic aberration (RGB channel-shift) effect with attack-decay envelope.

Implements the localized channel-shift effect described in PRD Section 11.2.
Maintains state for the attack-decay pulse intensity:
- Ramps intensity from 0 -> 1 over GLITCH_ATTACK_FRAMES when triggered.
- Decays per frame (intensity *= GLITCH_DECAY_RATE) while ACTIVE.
"""

from typing import List, Optional, Tuple
import cv2
import numpy as np

import config
from utils.geometry import compute_palm_center


class GlitchEffect:
    """Manages intensity envelope and applies chromatic aberration to target regions."""

    def __init__(self):
        self.intensity = 0.0
        self.attack_step = 1.0 / max(1, config.GLITCH_ATTACK_FRAMES)
        self.was_active = False

    def trigger(self):
        """Re-arm or spike the pulse when entering ACTIVE state."""
        self.intensity = 1.0

    def update_intensity(self, is_active: bool):
        """Update pulse intensity envelope for the current frame."""
        if is_active:
            if not self.was_active:
                # Fresh trigger: ramp up
                self.intensity = 1.0
            else:
                # Decay intensity while ACTIVE
                self.intensity *= config.GLITCH_DECAY_RATE
                if self.intensity < 0.05:
                    self.intensity = 0.05  # minimum baseline glow while held
        else:
            self.intensity = 0.0

        self.was_active = is_active

    def apply(
        self,
        frame: np.ndarray,
        hand_landmarks_list: List[List],
        active_mask: List[bool],
    ) -> np.ndarray:
        """
        Apply chromatic aberration channel-shift to the target region in-place.

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
            Frame with localized RGB channel-shift applied.
        """
        is_active = any(active_mask)
        self.update_intensity(is_active)

        if self.intensity <= 0.01:
            return frame

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

        # Compute bounding rectangle for the glitch effect (matches web overlay region)
        if len(active_indices) >= 2:
            p1, p2 = palm_centers[0], palm_centers[1]
            cx = (p1[0] + p2[0]) / 2.0
            cy = (p1[1] + p2[1]) / 2.0
            dist = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
            rw = max(120, int(dist * config.WEB_WIDTH_SCALE))
            rh = max(80, int(rw * config.WEB_HEIGHT_RATIO))
        else:
            idx = active_indices[0]
            lm = hand_landmarks_list[idx]
            cx, cy = palm_centers[0]
            w_x, w_y = lm[0].x * fw, lm[0].y * fh
            m_x, m_y = lm[9].x * fw, lm[9].y * fh
            hand_size = np.hypot(m_x - w_x, m_y - w_y)
            rw = max(140, int(hand_size * 2.8 * config.WEB_WIDTH_SCALE))
            rh = max(90, int(rw * config.WEB_HEIGHT_RATIO))

        # Region coordinates
        x1 = max(0, int(cx - rw / 2.0))
        y1 = max(0, int(cy - rh / 2.0))
        x2 = min(fw, int(cx + rw / 2.0))
        y2 = min(fh, int(cy + rh / 2.0))

        if x2 <= x1 or y2 <= y1:
            return frame

        region = frame[y1:y2, x1:x2]
        shift_px = int(self.intensity * config.GLITCH_MAX_SHIFT_PX)

        if shift_px <= 0:
            return frame

        # Split B, G, R channels
        b, g, r = cv2.split(region)

        # Shift Red channel right, Blue channel left (PRD Section 11.2)
        r_shifted = np.roll(r, shift_px, axis=1)
        b_shifted = np.roll(b, -shift_px, axis=1)

        glitched = cv2.merge([b_shifted, g, r_shifted])

        # Blend glitched region with original region
        blend_weight = config.GLITCH_BLEND * self.intensity
        blended = cv2.addWeighted(region, 1.0 - blend_weight, glitched, blend_weight, 0)
        frame[y1:y2, x1:x2] = blended

        return frame
