import collections.abc as c
from pathlib import Path

import numpy as np
import numpy.typing as npt

from core import utils as ut
from core.job import Job
from core.operation_types import FaceToolPair
from .batch_cropper import BatchCropper


class MappingCropper(BatchCropper):
    def __init__(self, face_detection_tools: c.Iterator[FaceToolPair]):
        super().__init__(face_detection_tools)

    def worker(self, file_amount: int,
               job: Job,
               face_detection_tools: FaceToolPair, *,
               old: npt.NDArray[np.str_],
               new: npt.NDArray[np.str_]):
        """
        Performs cropping for a mapping job by iterating over the old file list, cropping each image, and updating the progress.

        Args:
            self: The Cropper instance.
            file_amount (int): The total number of files to process.
            job (Job): The job containing the parameters for cropping.
            face_detection_tools(Tuple[Any, Any]): The worker for face-related tasks.
            old (npt.NDArray[np.str_]): The array of old file paths.
            new (npt.NDArray[np.str_]): The array of new file paths.

        Returns:
            None
        """
        for old, new in zip(old, new):
            if self.end_task:
                break

            old_path: Path = job.folder_path / old
            new_path: Path = job.destination / (new + old_path.suffix) if job.radio_choice() == 'No' else job.destination / (new + job.radio_choice())

            if old_path.is_file():
                ut.crop(old_path, job, face_detection_tools, new_path)
            self._update_progress(file_amount)

        if self.progress_count == file_amount or self.end_task:
            self.show_message_box = False

    def crop(self, job: Job) -> None:
        """
        Performs cropping for a mapping job by splitting the file lists and mapping data into chunks and running mapping workers in separate threads.
    
        Args:
            self: The Cropper instance.
            job (Job): The job containing the file lists and mapping data.
    
        Returns:
            None
        """

        if not (file_tuple := job.file_list_to_numpy()):
            return
        if not job.destination:
            return

        # Check if the destination directory is writable.
        if not job.destination_accessible:
            return self.access_error()

        total_size = job.byte_size * len(file_tuple[0])

        # Check if there is enough space on disk to process the files.
        if job.available_space == 0 or job.available_space < total_size:
            return self.capacity_error()
        
        if self.MEM_FACTOR < 1:
            return self.memory_error()
        
        # Get the extensions of the file names and
        # Create a mask that indicates which files have supported extensions.
        mask, amount = ut.mask_extensions(file_tuple[0])
        # Split the file lists and the mapping data into chunks.
        old_file_list, new_file_list = ut.split_by_cpus(mask, self.THREAD_NUMBER, file_tuple[0], file_tuple[1])

        self.emit_progress(amount)

        self.set_futures(self.worker, amount, job, old_file_list, new_file_list)

        # Attach a done callback to handle worker completion
        self.complete_futures()
