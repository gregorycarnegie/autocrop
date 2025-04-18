from pathlib import Path
from typing import Protocol, Optional
import cv2

from .face_tools import FaceToolPair
from .job import Job


class ImageOpener(Protocol):
    """
    Strategy protocol for opening and preparing images.
    """
    def __call__(
        self,
        file: Path,
        face_detection_tools: FaceToolPair,
        job: Job
    ) -> Optional[cv2.Mat]:
        ...
