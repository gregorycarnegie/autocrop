from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import numpy.typing as npt

from core import utils as ut
from core.job import Job
from core.operation_types import FaceToolPair
from .cropper import Cropper


class MappingCropper(Cropper):
    def __init__(self, face_detection_tools: list[FaceToolPair]):
        super().__init__()
        self.face_detection_tools = face_detection_tools

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
        for old, new in zip(old.tolist(), new.tolist()):
            if self.end_task:
                break

            old_path: Path = job.folder_path / old
            new_path: Path = job.destination / (new + old_path.suffix) if job.radio_choice() == 'No' else job.destination / (new + job.radio_choice())

            if old_path.is_file():
                ut.crop(old_path, job, face_detection_tools, new_path)
            self._update_progress(file_amount)

        if self.bar_value == file_amount or self.end_task:
            self.message_box = False

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
        # file_list1, file_list2 = file_tuple
        if job.destination:
            # Check if the destination directory is writable.
            if not job.destination_accessible:
                return self.access_error()

            total_size = job.byte_size * len(file_tuple[0])

            # Check if there is enough space on disk to process the files.
            if job.available_space == 0 or job.available_space < total_size:
                return self.capacity_error()

        # Get the extensions of the file names and
        # Create a mask that indicates which files have supported extensions.
        mask, amount = ut.mask_extensions(file_tuple[0])
        # Split the file lists and the mapping data into chunks.
        old_file_list, new_file_list = ut.split_by_cpus(mask, self.THREAD_NUMBER, file_tuple[0], file_tuple[1])

        self.bar_value = 0
        self.progress.emit(self.bar_value, amount)
        self.started.emit()

        self.executor = ThreadPoolExecutor(max_workers=self.THREAD_NUMBER)
        self.futures = [
            self.executor.submit(self.worker, amount, job, tool_pair, old=old_chunk, new=new_chunk)
            for old_chunk, new_chunk, tool_pair in zip(old_file_list, new_file_list, self.face_detection_tools)
        ]

        # Attach a done callback to handle worker completion
        for future in self.futures:
            future.add_done_callback(self.worker_done_callback)
