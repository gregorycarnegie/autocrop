import collections.abc as c
from itertools import batched
from pathlib import Path

from core import utils as ut
from core.job import Job
from core.operation_types import FaceToolPair
from .batch_cropper import BatchCropper


class FolderCropper(BatchCropper):
    def __init__(self, face_detection_tools: c.Iterator[FaceToolPair]):
        super().__init__(face_detection_tools)

    def worker(self, file_amount: int,
               file_list: tuple[Path],
               job: Job,
               face_detection_tools: FaceToolPair) -> None:
        """
        Performs cropping for a folder job by iterating over the file list, cropping each image, and updating the progress.

        Args:
            self: The Cropper instance.
            file_amount (int): The total number of files to process.
            file_list (npt.NDArray[Any]): The array of file paths.
            job (Job): The job containing the parameters for cropping.
            face_detection_tools(Tuple[Any, Any]): The worker for face-related tasks.

        Returns:
            None
        """

        for image in file_list:
            if self.end_task:
                break
            ut.crop(image, job, face_detection_tools)
            self._update_progress(file_amount)

        if self.progress_count == file_amount or self.end_task:
            self.show_message_box = False

    def crop(self, job: Job) -> None:
        """
        Crops all files in a directory by splitting the file list into chunks and running folder workers in separate threads.
    
        Args:
            self: The Cropper instance.
            job (Job): The job containing the file list.
    
        Returns:
            None
        """
        if not (file_tuple := job.path_iter()):
            return

        if not job.destination:
            return

        # Check if the destination directory is writable.
        if not job.destination_accessible:
            return self.access_error()

        file_list, amount = file_tuple

        if not amount:
            return self.amount_error()

        total_size = job.byte_size * amount

        # Check if there is enough space on disk to process the files.
        if job.available_space == 0 or job.available_space < total_size:
            return self.capacity_error()
        
        if self.MEM_FACTOR < 1:
            return self.memory_error()
        
        # Split the file list into chunks.
        split_array = batched(file_list, amount // self.THREAD_NUMBER + 1)

        self.emit_progress(amount)

        self.set_futures(self.worker, amount, job, split_array)

        # Attach a done callback to handle worker completion
        self.complete_futures()
