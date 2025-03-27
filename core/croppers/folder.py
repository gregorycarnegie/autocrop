import collections.abc as c
from itertools import batched
from pathlib import Path
from typing import Optional

from core import processing as prc
from core.job import Job
from core.operation_types import FaceToolPair
from .batch import BatchCropper


class FolderCropper(BatchCropper):
    def __init__(self, face_detection_tools: c.Iterator[FaceToolPair]):
        super().__init__(face_detection_tools)

    def worker(self, *, file_amount: int,
               job: Job,
               face_detection_tools: FaceToolPair,
               file_list: tuple[Path]) -> None:
        """
        Performs cropping for a folder job by iterating over the file list.
        """
        for image in file_list:
            if self.end_task:
                break
            prc.crop(image, job, face_detection_tools)
            self._update_progress(file_amount)

        if self.progress_count == file_amount or self.end_task:
            self.show_message_box = False

    def prepare_crop_operation(self, job: Job) -> tuple[Optional[int], Optional[c.Iterable]]:
        """
        Prepare the folder crop operation by getting file list and splitting into chunks.
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
