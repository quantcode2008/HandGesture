"""
camera.py - Webcam capture wrapper.

Opens the webcam via cv2.VideoCapture and yields BGR frames.
Resolution and device index are pulled from config.py (PRD Section 14).
"""

import cv2
import config


class Camera:
    """Thin wrapper around cv2.VideoCapture with config-driven defaults."""

    def __init__(
        self,
        index: int = config.CAMERA_INDEX,
        width: int = config.FRAME_WIDTH,
        height: int = config.FRAME_HEIGHT,
    ):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera at index {index}. "
                "Check that a webcam is connected and not in use by another app."
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Read back the actual resolution the driver accepted
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read(self):
        """Return (success: bool, frame: np.ndarray | None)."""
        return self.cap.read()

    def release(self):
        """Release the underlying VideoCapture."""
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False
