from pathlib import Path
from typing import Optional

from core import processing as prc
from core.face_tools import FaceToolPair
from core.job import Job
from .base import Cropper


class PhotoCropper(Cropper):
    def __init__(self, face_detection_tools: FaceToolPair):
        super().__init__()
        self.face_detection_tools = face_detection_tools

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
        if not image.is_file():
            return None

        if job.safe_destination:
            # Check if the destination directory is writable.
            if not job.destination_accessible:
                exception, message = self.create_error('access')
                return self._display_error(exception, message)
            
            # Check if there is enough space on the disk to process the files.
            if job.free_space == 0 or job.free_space < job.approx_byte_size:
                exception, message = self.create_error('capacity')
                return self._display_error(exception, message)
        
            if self.MEM_FACTOR < 1:
                exception, message = self.create_error('memory')
                return self._display_error(exception, message)
            
        prc.crop_from_path(image, job, self.face_detection_tools, new)
        return None
