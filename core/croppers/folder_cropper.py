from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import numpy.typing as npt

from core import utils as ut
from core.job import Job
from core.operation_types import FaceToolPair
from .cropper import Cropper


class FolderCropper(Cropper):
    def __init__(self, face_detection_tools: list[FaceToolPair]):
        super().__init__()
        self.face_detection_tools = face_detection_tools

    def worker(self, file_amount: int,
               file_list: npt.NDArray[Any],
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

        if self.bar_value == file_amount or self.end_task:
            self.message_box = False

    def crop(self, job: Job) -> None:
        """
        Crops all files in a directory by splitting the file list into chunks and running folder workers in separate threads.
    
        Args:
            self: The Cropper instance.
            job (Job): The job containing the file list.
    
        Returns:
            None
        """

        if not (file_tuple := job.file_list()):
            return

        if job.destination:
            # Check if the destination directory is writable.
            if not job.destination_accessible:
                return self.access_error()
            
            file_list, amount = file_tuple
            total_size = job.byte_size * amount

            # Check if there is enough space on disk to process the files.
            if job.available_space == 0 or job.available_space < total_size:
                return self.capacity_error()
            
        # Split the file list into chunks.
        split_array = np.array_split(file_list, self.THREAD_NUMBER)

        self.bar_value = 0
        self.progress.emit((self.bar_value, amount))
        self.started.emit()

        self.executor = ThreadPoolExecutor(max_workers=self.THREAD_NUMBER)
        self.futures = [
            self.executor.submit(self.worker, amount, chunk, job, tool_pair)
            for chunk, tool_pair in zip(split_array, self.face_detection_tools)
        ]

        # Attach a done callback to handle worker completion
        for future in self.futures:
            future.add_done_callback(self.worker_done_callback)
