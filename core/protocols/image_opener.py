from pathlib import Path
from typing import Protocol, Optional

import cv2

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
    ) -> Optional[cv2.Mat]:
        ...
