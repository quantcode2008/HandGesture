"""
camera.py - Threaded webcam capture wrapper.

Captures frames asynchronously in a dedicated worker thread so camera I/O
never blocks the main loop. Always delivers the latest captured frame and
automatically drops stale buffer frames for ultra-low latency (<100ms).
"""

import threading
import time
from typing import Optional, Tuple
import cv2
import numpy as np

import config


class Camera:
    """Threaded wrapper around cv2.VideoCapture with config-driven defaults."""

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

        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._running = True
        self._new_frame_event = threading.Event()

        # Start background capture thread
        self._thread = threading.Thread(target=self._capture_worker, name="CameraWorker", daemon=True)
        self._thread.start()

        # Wait up to 1 second for the very first frame to arrive
        self._new_frame_event.wait(timeout=1.0)

    def _capture_worker(self):
        """Continuously grab frames from webcam into the single-frame buffer."""
        while self._running:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue
            with self._lock:
                self._latest_frame = frame
            self._new_frame_event.set()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Return (success: bool, frame: np.ndarray | None).

        Non-blocking: immediately returns the newest frame captured by the thread,
        dropping any intermediate stale frames.
        """
        with self._lock:
            if self._latest_frame is not None:
                return True, self._latest_frame.copy()
            return False, None

    def release(self):
        """Signal worker thread to stop and release the underlying VideoCapture."""
        self._running = False
        if hasattr(self, "_thread") and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False
