import collections.abc as c
import threading
from itertools import batched
from pathlib import Path
from typing import Optional

from core import processing as prc
from core.face_tools import FaceToolPair
from core.job import Job
from .batch import BatchCropper


class FolderCropper(BatchCropper):
    def __init__(self, face_detection_tools: list[FaceToolPair]):
        super().__init__(face_detection_tools)

    def worker(self, *, file_amount: int,
            job: Job,
            face_detection_tools: FaceToolPair,
            file_list: tuple[Path, ...],
            cancel_event: threading.Event) -> None:
        """
        Performs cropping for a folder job by iterating over the file list.
        Uses batch_process_with_pipeline for efficiency.
        """
        # Emit started signal at the beginning of processing
        if not self.finished_signal_emitted:
            self.started.emit()
        
        # Convert tuple to list if needed
        image_paths = list(file_list)
        
        # This function creates the pipeline once and applies it to all images
        prc.batch_process_with_pipeline(
            image_paths, job, face_detection_tools, cancel_event, False, self.progressBars, self.progress_count
        )
        
        if self.progress_count == file_amount or self.end_task:
            self.show_message_box = False

    def prepare_crop_operation(self, job: Job) -> tuple[Optional[int], Optional[c.Iterable]]:
        """
        Prepare the folder crop_from_path operation by getting file list and splitting into chunks.
        """
        if not (file_list := job.path_iter()):
            return None, None

        amount = len(file_list)
        if not amount:
            exception, message = self.create_error('amount')
            return self._display_error(exception, message), None

        # Split the file list into chunks
        split_array = batched(file_list, amount // self.THREAD_NUMBER + 1)
        return amount, split_array

    def set_futures_for_crop(self, job: Job, file_count: int, chunked_data: c.Iterable) -> None:
        """
        Set up futures specifically for folder cropping.
        """
        self.set_futures(self.worker, file_count, job, chunked_data)
