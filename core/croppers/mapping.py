import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import batched
from pathlib import Path
from typing import Any

import autocrop_rs.file_types as r_types  # type: ignore
import numpy as np
import numpy.typing as npt

from core import processing as prc
from core.face_tools import FaceToolPair
from core.job import Job
from file_types import FileCategory, file_manager

from .batch import BatchCropper


class MappingCropper(BatchCropper):
    def __init__(self, face_detection_tools: list[FaceToolPair]):
        super().__init__(face_detection_tools)

    def worker(
            self, *, file_amount: int,
            job: Job,
            face_detection_tools: FaceToolPair,
            old: npt.NDArray[np.str_],
            new: npt.NDArray[np.str_],
            cancel_event: threading.Event
    ) -> None:
        """
        Performs cropping for a mapping job using two-phase approach.
        """
        if cancel_event.is_set():
            return

        # Convert mapping arrays to lists of image paths and their targets
        image_paths, output_paths = self.get_mapping(job, old, new)

        if not image_paths or cancel_event.is_set():
            return

        # Phase 1: Generate crop instructions
        instructions = prc.generate_crop_instructions(
            image_paths, job, face_detection_tools, output_paths, self.rejected_files
        )

        # Phase 2: Execute the cropping instructions in parallel
        self.execute_parallel_job(instructions, cancel_event)

        self.update_completion_status(file_amount)

    @staticmethod
    def get_mapping(job: Job, old: npt.NDArray[np.str_], new: npt.NDArray[np.str_]) -> tuple[list[Path], list[Path]]:
        """
        Extracts the file names from the old and new arrays and returns them as separate arrays.
        """
        image_paths: list[Path] = []
        output_paths: list[Path] = []

        for old_name, new_name in zip(old, new):
            old_path: Path = job.safe_folder_path / old_name
            if old_path.is_file():
                new_path: Path = job.safe_destination / (new_name + old_path.suffix if job.radio_choice() == 'No'
                                        else new_name + job.radio_choice())
                image_paths.append(old_path)
                output_paths.append(new_path)

        return image_paths, output_paths

    def prepare_crop_operation(
            self, job: Job
    ) -> tuple[int | None, tuple[npt.NDArray[np.str_], npt.NDArray[np.str_]] | None]:
        """
        Prepare the mapping crop_from_path operation by getting file lists and splitting into chunks.
        """
        if not (file_tuple := job.file_list_to_numpy()):
            return None, None

        old, new = file_tuple

        extensions = (
                file_manager.get_extensions(FileCategory.PHOTO) |
                file_manager.get_extensions(FileCategory.RAW) |
                file_manager.get_extensions(FileCategory.TIFF)
        )
        # Get the extensions of the file names and create a mask for supported extensions
        mask, amount = prc.mask_extensions(old, extensions)

        return (None, None) if amount == 0 else (amount, (old[mask], new[mask]))

    def set_futures_for_crop(self, job: Job, file_count: int, file_lists: tuple[Any, Any]) -> None:
        """
        Set up futures specifically for mapping cropping.
        """
        old_list, new_list = file_lists
        self.set_futures(self.worker, file_count, job, old_list, new_list)

    def set_futures(
            self, worker: Callable[..., None],
            amount: int,
            job: Job,
            list_1: npt.NDArray[np.str_],
            list_2: npt.NDArray[np.str_]
    ) -> None:
        """
        Configure worker futures for parallel execution with enhanced security.
        """

        # Check if we should use multithreading
        if not self._should_use_multithreading(amount):
            # Single-threaded processing

            # For mapping operations, process directly
            self.worker(
                file_amount=amount,
                job=job,
                face_detection_tools=self.face_detection_tools[0],
                old=list_1,
                new=list_2,
                cancel_event=self.cancel_event
            )

            # Immediately signal completion
            self.emit_done()
            return

        # Recreate executor if it was shut down
        if self.executor is None or self.executor._shutdown:
            self.executor = ThreadPoolExecutor(max_workers=self.THREAD_NUMBER)

        # Validate inputs before processing
        if not self._validate_inputs(worker, amount, job, list_1, list_2):
            # Log error and return early if validation fails
            return

        # Convert input paths to strings for Rust validation
        path_strings = [job.safe_folder_path / path for path in list_1]

        # Determine file categories for each path (0=Photo, 1=Raw, 2=Tiff, 3=Video, 4=Table, 5=Unknown)
        categories = []
        for path in path_strings:
            if file_manager.is_valid_type(path, FileCategory.PHOTO):
                categories.append(0)  # Photo category
            elif file_manager.is_valid_type(path, FileCategory.RAW):
                categories.append(1)  # Raw category
            elif file_manager.is_valid_type(path, FileCategory.TIFF):
                categories.append(2)  # Tiff category
            elif file_manager.is_valid_type(path, FileCategory.VIDEO):
                categories.append(3)  # Video category
            elif file_manager.is_valid_type(path, FileCategory.TABLE):
                categories.append(4)  # Table category
            else:
                categories.append(5)  # Unknown category

        path_strings = [path.as_posix() for path in path_strings]

        # Validate all files in parallel using Rust
        validation_results = r_types.validate_files(path_strings, categories)

        # Filter valid files
        valid_indices = np.where(validation_results)[0]
        valid_list_1 = [list_1[i] for i in valid_indices]

        valid_list_2 = [list_2[i] for i in valid_indices] if len(valid_indices) > 0 else []

        # Use partial with validated inputs only
        worker_with_params = partial(self._secure_worker_wrapper,
                                     original_worker=worker,
                                     file_amount=len(valid_indices),
                                     job=job,
                                     cancel_event=self.cancel_event)

        self.futures = []

        # Prevent submitting to a shutdown executor
        if self.executor is None or self.executor._shutdown:
            self.end_task = True
            return

        # Create futures with additional security checks
        chunk_size = max(len(valid_list_1) // self.THREAD_NUMBER, 1) if valid_list_1 else 1

        try:
            old_batch = batched(valid_list_1, chunk_size)
            new_batch = batched(valid_list_2, chunk_size)
            self.futures.extend(
                self.executor.submit(
                    worker_with_params,
                    face_detection_tools=tool_pair,
                    old=tuple(old_chunk),
                    new=tuple(new_chunk),
                )
                for old_chunk, new_chunk, tool_pair in zip(
                    old_batch, new_batch, self.face_detection_tools
                )
            )
        except RuntimeError as e:
            if "cannot schedule new futures after shutdown" not in str(e):
                raise
            self.end_task = True
