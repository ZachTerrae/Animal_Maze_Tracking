"""Reading videos with OpenCV, the same library DeepLabCut uses.

Because DeepLabCut also reads frames with OpenCV, frame number N here is the
same picture that produced row N of the tracking table.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


def get_cv2():
    """Import OpenCV without letting it disturb the Qt user interface."""
    before = os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH")
    import cv2

    after = os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH")
    if after != before and after and "cv2" in after:
        # Some OpenCV builds point Qt at their own plug-ins, which breaks PySide6.
        if before is None:
            os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
        else:
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = before
    return cv2


@dataclass
class VideoInfo:
    width: int = 0
    height: int = 0
    fps: float = 0.0
    n_frames: int = 0
    ok: bool = False

    @property
    def size(self) -> tuple[int, int] | None:
        return (self.width, self.height) if self.width > 0 and self.height > 0 else None


def probe_video(path: str) -> VideoInfo:
    """Frame size, frame rate and frame count of a video file."""
    info = VideoInfo()
    if not path or not os.path.isfile(path):
        return info
    cv2 = get_cv2()
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return info
        info.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        info.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        info.fps = fps if np.isfinite(fps) and 0 < fps < 100000 else 0.0
        count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        info.n_frames = int(count) if np.isfinite(count) and 0 < count < 1e9 else 0
        info.ok = info.width > 0 and info.height > 0
    finally:
        cap.release()
    return info


class FrameReader:
    """Random access to the frames of one video."""

    def __init__(self, path: str):
        self.path = path
        cv2 = get_cv2()
        self._cv2 = cv2
        self._cap = cv2.VideoCapture(path)
        self._next = 0
        self.ok = bool(self._cap.isOpened())

    def read(self, index: int) -> np.ndarray | None:
        """Frame ``index`` as an RGB array, or None if it cannot be read."""
        if not self.ok or index < 0:
            return None
        cv2 = self._cv2
        if index != self._next:
            # Stepping forward a little is exact and often faster than seeking.
            if self._next < index <= self._next + 12:
                while self._next < index:
                    if not self._cap.grab():
                        self._next = -1
                        return None
                    self._next += 1
            else:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        success, frame = self._cap.read()
        if not success or frame is None:
            self._next = -1
            return None
        self._next = index + 1
        if frame.ndim == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        try:
            self._cap.release()
        except Exception:
            pass
        self.ok = False
