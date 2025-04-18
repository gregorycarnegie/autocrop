from pathlib import Path
from typing import Protocol, Optional, Union

import cv2
import numpy as np

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


class ImageWriter(Protocol):
    """
    Strategy protocol for opening and preparing images.
    """
    def __call__(
        self,
        image: Union[cv2.Mat, np.ndarray],
        file: Path,
        user_gam: float,
        is_tiff: bool = False
    ) -> None:
        ...

