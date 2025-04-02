import collections.abc as c
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Callable, Optional, Union, TypeVar

import numpy as np
import numpy.typing as npt

from core.face_tools import FaceToolPair
from core.job import Job
from .base import Cropper

# Type definitions for better type hints
T = TypeVar('T')
FileList = Union[list[Path], npt.NDArray[np.str_]]


class BatchCropper(Cropper):
    """
    A class that manages image-cropping tasks using a thread pool.
    """

    def __init__(self, face_detection_tools: c.Iterator[FaceToolPair]):
        super().__init__()
        self.executor: Optional[ThreadPoolExecutor] = None
        self.futures: list[Future] = []
        self.executor = ThreadPoolExecutor(max_workers=self.THREAD_NUMBER)
        self.face_detection_tools = list(face_detection_tools)

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__}("
            f"threads={self.THREAD_NUMBER}, "
            f"progress_count={self.progress_count}, "
            f"end_task={self.end_task}, "
            f"show_message_box={self.show_message_box})>"
        )

    def terminate(self) -> None:
        """
        Terminates all pending tasks and shuts down the executor.
        """
        if not self.end_task:
            self.end_task = True
            self.emit_done()

        if self.executor:
            for future in self.futures:
                if not future.done():
                    future.cancel()
            self.executor.shutdown(wait=False)
            self.executor = None

    def emit_progress(self, amount: int):
        """Initialize progress tracking and emit started signal"""
        self.progress_count = 0
        self.progress.emit(self.progress_count, amount)
        self.started.emit()

    def set_futures(self, worker: Callable[..., None],
                    amount: int,
                    job: Job,
                    list_1: c.Iterable[T],
                    list_2: Optional[c.Iterable] = None):
        """
        Configure worker futures for parallel execution.
        """
        # Recreate executor if it was shut down
        if self.executor is None or self.executor._shutdown:
            self.executor = ThreadPoolExecutor(max_workers=self.THREAD_NUMBER)

        worker_with_params = partial(worker, file_amount=amount, job=job)
        if list_2:
            self.futures = [
                worker_with_params(face_detection_tools=tool_pair, old=old_chunk, new=new_chunk)
                for old_chunk, new_chunk, tool_pair in zip(list_1, list_2, self.face_detection_tools)
            ]
        else:
            self.futures = [
                self.executor.submit(worker_with_params, face_detection_tools=tool_pair, file_list=chunk)
                for chunk, tool_pair in zip(list_1, self.face_detection_tools)
                ]

    def complete_futures(self):
        """Attach a done callback to all futures"""
        for future in self.futures:
            future.add_done_callback(self.worker_done_callback)

    def all_tasks_done(self) -> None:
        """
        Checks if all futures have completed. If they have, shuts down the executor and emits done.
        """
        if not self.end_task and all(future.done() for future in self.futures):
            if self.executor:
                self.executor.shutdown(wait=False)  # Non-blocking shutdown
            self.emit_done()
            self.end_task = True

    def worker_done_callback(self, future: Future) -> None:
        """
        Callback function to handle completion of a worker thread with detailed error reporting.
        """
        try:
            future.result()  # This raises any exceptions that occurred during execution
        except FileNotFoundError:
            self.create_error('file',"File not found. Please check that all input files exist.")
        except PermissionError:
            self.create_error('access')
        except OSError as e:
            if "space" in str(e).lower():
                self.create_error('capacity')
            else:
                self._display_error(e, "File system error. Check input and output paths.")
        except ValueError as e:
            self._display_error(e, "Invalid data format. Please check input files.")
        except (TypeError, AttributeError) as e:
            self._display_error(e, "Data type error. This may indicate a corrupted image file.")
        except MemoryError:
            self.create_error('memory')
        except Exception as e:
            self._display_error(e, f"An unexpected error occurred: {type(e).__name__}")
        finally:
            # Check if all futures are done, then emit finished signal
            if all(f.done() for f in self.futures):
                self.all_tasks_done()

    def validate_job(self, job: Job, file_count: Optional[int] = None) -> bool:
        """
        Validate job parameters and available resources.

        Returns:
            bool: True if job is valid, False if validation failed and error was reported
        """
        if not job.destination:
            return False

        # Check if the destination directory is writable
        if not job.destination_accessible:
            self.create_error('access')
            return False

        # If file count provided, check disk capacity
        if file_count is not None:
            total_size = job.byte_size * file_count

            # Check if there is enough space on disk to process the files
            if job.available_space == 0 or job.available_space < total_size:
                self.create_error('capacity')
                return False

        if self.MEM_FACTOR < 1:
            self.create_error('memory')
            return False

        return True

    def prepare_crop_operation(self, job: Job) -> tuple[Optional[int], Optional[c.Iterable]]:
        """
        Abstract method to be implemented by child classes.
        Should prepare the crop operation by validating inputs and creating
        the list of items to process.

        Returns:
            tuple: (file_count, chunked_data) or (None, None) if preparation failed
        """
        raise NotImplementedError("Child classes must implement prepare_crop_operation")

    def crop(self, job: Job) -> None:
        """
        Common implementation of the crop method. Uses template method pattern.
        """
        # Let child class prepare the operation
        file_count, chunked_data = self.prepare_crop_operation(job)

        if file_count is None or chunked_data is None:
            return

        # Validate common job parameters
        if not self.validate_job(job, file_count):
            return

        # Start the processing
        self.emit_progress(file_count)

        # Let child class set up the futures based on its specific worker
        self.set_futures_for_crop(job, file_count, chunked_data)

        # Complete futures is common to all
        self.complete_futures()

    def set_futures_for_crop(self, job: Job, file_count: int, chunked_data: c.Iterable) -> None:
        """
        Abstract method to be implemented by child classes.
        Should set up the futures for the crop operation.
        """
        raise NotImplementedError("Child classes must implement set_futures_for_crop")
