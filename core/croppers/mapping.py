import collections.abc as c
from pathlib import Path
from typing import Optional, Any

import numpy as np
import numpy.typing as npt

from core import processing as prc
from core.face_tools import FaceToolPair
from core.job import Job
from file_types import registry
from .batch import BatchCropper


class MappingCropper(BatchCropper):
    def __init__(self, face_detection_tools: c.Iterator[FaceToolPair]):
        super().__init__(face_detection_tools)

    def worker(self, *, file_amount: int,
               job: Job,
               face_detection_tools: FaceToolPair,
               old: npt.NDArray[np.str_],
               new: npt.NDArray[np.str_]):
        """
        Performs cropping for a mapping job.
        """
        for old, new in zip(old, new):
            if self.end_task:
                break

            old_path: Path = job.folder_path / old
            new_path: Path = job.destination / (new + old_path.suffix) if job.radio_choice() == 'No' else job.destination / (new + job.radio_choice())

            if old_path.is_file():
                prc.crop(old_path, job, face_detection_tools, new_path)
            self._update_progress(file_amount)

        if self.progress_count == file_amount or self.end_task:
            self.show_message_box = False

    def prepare_crop_operation(self, job: Job) -> tuple[Optional[int], Optional[tuple[Any, Any]]]:
        """
        Prepare the mapping crop operation by getting file lists and splitting into chunks.
        """
        if not (file_tuple := job.file_list_to_numpy()):
            return None, None

        exts = registry.get_extensions('photo') | registry.get_extensions('raw')
        # Get the extensions of the file names and create a mask for supported extensions
        mask, amount = prc.mask_extensions(file_tuple[0], exts)

        # Split the file lists and the mapping data into chunks
        old_file_list, new_file_list = prc.split_by_cpus(mask, self.THREAD_NUMBER, file_tuple[0], file_tuple[1])

        return amount, (old_file_list, new_file_list)

    def set_futures_for_crop(self, job: Job, file_count: int, chunked_data: tuple[Any, Any]) -> None:
        """
        Set up futures specifically for mapping cropping.
        """
        old_list, new_list = chunked_data
        self.set_futures(self.worker, file_count, job, old_list, new_list)
