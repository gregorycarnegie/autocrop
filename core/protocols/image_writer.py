from pathlib import Path
from typing import Protocol

import cv2.typing as cvt


class ImageWriter(Protocol):
    """
    Strategy protocol for opening and preparing images.
    """
    def __call__(
        self,
        image: cvt.MatLike,
        file_path: Path,
        _user_gam: float,
        _is_tiff: bool = False
    ) -> None:
        ...
