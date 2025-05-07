import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import batched
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from autocrop_rs import validate_files

from core import processing as prc
from core.face_tools import FaceToolPair
from core.job import Job
from file_types import FileCategory, file_manager
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
        # Convert tuple to list if needed
        image_paths = list(file_list)
        
        # Process the images
        prc.batch_process_with_pipeline(
            image_paths, job, face_detection_tools, cancel_event, False
        )
        
        # Update completion status
        self._check_completion(file_amount)
        
        if self.progress_count == file_amount or self.end_task:
            self.show_message_box = False

    def prepare_crop_operation(self, job: Job) -> tuple[Optional[int], Optional[list[Path]]]:
        """
        Prepare the folder crop_from_path operation by getting the file list and splitting into chunks.
        """
        if not (file_list := job.iter_images()):
            return None, None

        amount = len(file_list)
        if not amount:
            exception, message = self.create_error('amount')
            return self._display_error(exception, message), None

        # Split the file list into chunks
        return amount, file_list

    def set_futures_for_crop(self, job: Job, file_count: int, file_list: list[Path]) -> None:
        """
        Set up futures specifically for folder cropping with adaptive threading.
        """
        self.set_futures(self.worker, file_count, job, file_list)


    def set_futures(self, worker: Callable[..., None],
                    amount: int,
                    job: Job,
                    list_1: list[Path]):
        """
        Configure worker futures for parallel execution with enhanced security.
        """

        # Check if we should use multithreading
        if not self._should_use_multithreading(amount):
            # Single-threaded processing
            self._process_single_threaded(list_1, job)
            # Immediately signal completion
            self.emit_done()
            return

        # Recreate executor if it was shut down
        if self.executor is None or self.executor._shutdown:
            self.executor = ThreadPoolExecutor(max_workers=self.THREAD_NUMBER)

        # Validate inputs before processing
        if not self._validate_inputs(worker, amount, job, list_1, None):
            # Log error and return early if validation fails
            return

        # Convert input paths to strings for Rust validation
        path_strings = [path.as_posix() for path in list_1]

        # Determine file categories for each path (0=Photo, 1=Raw, 2=Tiff, 3=Video, 4=Table, 5=Unknown)
        categories = []
        for path in list_1:
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

        # Validate all files in parallel using Rust
        validation_results = validate_files(path_strings, categories)

        # Filter valid files
        valid_indices = np.where(validation_results)[0]
        valid_list_1 = [list_1[i] for i in valid_indices]

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
            batch = batched(valid_list_1, chunk_size)
            self.futures.extend(
                self.executor.submit(
                    worker_with_params,
                    face_detection_tools=tool_pair,
                    file_list=tuple(chunk),
                )
                for chunk, tool_pair in zip(batch, self.face_detection_tools)
            )
        except RuntimeError as e:
            if "cannot schedule new futures after shutdown" not in str(e):
                raise
            self.end_task = True
