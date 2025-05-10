from pathlib import Path
from typing import Protocol

import cv2
import numpy as np


class ImageWriter(Protocol):
    """
    Strategy protocol for opening and preparing images.
    """
    def __call__(
        self,
        image: cv2.Mat | np.ndarray,
        file: Path,
        user_gam: float,
        is_tiff: bool = False
    ) -> None:
        ...
