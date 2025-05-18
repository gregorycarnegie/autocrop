from pathlib import Path
from typing import Protocol

import cv2.typing as cvt

from core.face_tools import FaceToolPair
from core.job import Job


class ImageOpener(Protocol):
    """
    Strategy protocol for opening and preparing images.
    """
    def __call__(
        self,
        file: Path,
        face_detection_tools: FaceToolPair,
        job: Job
    ) -> cvt.MatLike | None:
        ...

class SimpleImageOpener(Protocol):
    """
    Strategy protocol for opening and preparing images.
    """
    def __call__(
        self,
        file: Path,
    ) -> cvt.MatLike | None:
        ...
