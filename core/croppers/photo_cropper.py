from pathlib import Path
from typing import Optional

from core import utils as ut
from core.job import Job
from core.operation_types import FaceToolPair
from .cropper import Cropper


class PhotoCropper(Cropper):
    def __init__(self, face_detection_tools: list[FaceToolPair]):
        super().__init__()
        self.face_detection_tools = face_detection_tools[0]

    def crop(self, image: Path,
             job: Job,
             new: Optional[str] = None) -> None:
        """
        Crops the photo image based on the provided job parameters.

        Args:
            image (Path): The path to the image file.
            job (Job): The job containing the parameters for cropping.
            new (Optional[str]): The optional new file name.

        Returns:
            None
        """
        if image.is_file():
            ut.crop(image, job, self.face_detection_tools, new)
