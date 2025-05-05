import threading
from pathlib import Path
from typing import Optional, Any

import numpy as np
import numpy.typing as npt

from core import processing as prc
from core.face_tools import FaceToolPair
from core.job import Job
from file_types import file_manager, FileCategory
from .batch import BatchCropper


class MappingCropper(BatchCropper):
    def __init__(self, face_detection_tools: list[FaceToolPair]):
        super().__init__(face_detection_tools)

    def worker(self, *, file_amount: int,
            job: Job,
            face_detection_tools: FaceToolPair,
            old: npt.NDArray[np.str_],
            new: npt.NDArray[np.str_],
            cancel_event: threading.Event) -> None:
        """
        Performs cropping for a mapping job using batch_process_with_mapping.
        """
        # Convert mapping arrays to list of image paths and their targets
        image_paths: list[Path] = []
        output_paths: list[Path] = []
        
        for old_name, new_name in zip(old, new):
            old_path: Path = job.safe_folder_path / old_name
            if old_path.is_file():
                new_path: Path = job.safe_destination / (new_name + old_path.suffix if job.radio_choice() == 'No' 
                                            else new_name + job.radio_choice())
                image_paths.append(old_path)
                output_paths.append(new_path)
        
        if image_paths and not cancel_event.is_set():
            prc.batch_process_with_mapping(
                image_paths,
                output_paths,
                job,
                face_detection_tools,
                cancel_event,
                False
            )
        
        # Update completion status
        self._check_completion(file_amount)
        
        if self.progress_count == file_amount or self.end_task:
            self.show_message_box = False

    def prepare_crop_operation(self, job: Job) -> tuple[Optional[int], Optional[tuple[Any, Any]]]:
        """
        Prepare the mapping crop_from_path operation by getting file lists and splitting into chunks.
        """
        if not (file_tuple := job.file_list_to_numpy()):
            return None, None

        extensions = (
                file_manager.get_extensions(FileCategory.PHOTO) |
                file_manager.get_extensions(FileCategory.RAW) |
                file_manager.get_extensions(FileCategory.TIFF)
        )
        # Get the extensions of the file names and create a mask for supported extensions
        mask, amount = prc.mask_extensions(file_tuple[0], extensions)

        # Split the file lists and the mapping data into chunks
        old_file_list, new_file_list = prc.split_by_cpus(mask, self.THREAD_NUMBER, file_tuple[0], file_tuple[1])

        return amount, (old_file_list, new_file_list)

    def set_futures_for_crop(self, job: Job, file_count: int, chunked_data: tuple[Any, Any]) -> None:
        """
        Set up futures specifically for mapping cropping.
        """
        old_list, new_list = chunked_data
        self.set_futures(self.worker, file_count, job, old_list, new_list)
