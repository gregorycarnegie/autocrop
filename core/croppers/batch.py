import atexit
import collections.abc as c
import threading
from concurrent.futures import Future, ThreadPoolExecutor, CancelledError
from functools import partial
from pathlib import Path
from typing import Callable, Optional, Union, TypeVar, Any

import numpy as np
import numpy.typing as npt
from PyQt6.QtCore import QMetaObject, Qt, Q_ARG
from PyQt6.QtWidgets import QApplication, QProgressBar

from core.face_tools import FaceToolPair
from core.job import Job
from .base import Cropper

T = TypeVar('T')
FileList = Union[list[Path], npt.NDArray[np.str_]]


class BatchCropper(Cropper):
    """
    A class that manages image-cropping tasks using a thread pool.
    """

    def __init__(self, face_detection_tools: list[FaceToolPair]):
        super().__init__()
        self.futures: list[Future] = []
        self.executor = ThreadPoolExecutor(max_workers=self.THREAD_NUMBER)
        self.face_detection_tools = face_detection_tools
        self.progressBars: list[QProgressBar] = []
        
        # Add a cancellation event for cooperative termination
        self.cancel_event = threading.Event()

        # Register an exit handler to ensure proper clean-up
        atexit.register(self._cleanup_executor)

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__}("
            f"threads={self.THREAD_NUMBER}, "
            f"progress_count={self.progress_count}, "
            f"end_task={self.end_task}, "
            f"show_message_box={self.show_message_box})>"
        )

    def cleanup(self) -> None:
        """
        Properly clean up resources before application exit.
        This method should be called during application shutdown.
        """
        if self.executor and not self.executor._shutdown:
            # Cancel any pending futures
            for future in self.futures:
                if future and not future.done():
                    future.cancel()
            
            # Shutdown the executor with a timeout
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None
            self.futures = []

    def _cleanup_executor(self):
        """Ensure the executor is properly shut down at application exit."""
        if self.executor and not self.executor._shutdown:
            self.executor.shutdown(wait=True)

    def terminate(self) -> None:
        """
        Terminates all pending tasks and shuts down the executor.
        """
        # Set the cancellation event to signal workers to stop
        self.cancel_event.set()

        if not self.end_task:
            self.end_task = True
            
            # Force reset progress to prevent lingering jobs
            self.progress_count = 0
            self.progress.emit(0, 1)  # Send zero progress
            
            # Emit finished signal to reset UI
            self.emit_done()

        if self.executor:
            for future in self.futures:
                if future and not future.done():
                    future.cancel()
        # Clear the list of futures
        self.futures = []

    def reset_task(self) -> None:
        """
        Resets the task-specific variables to their default values.
        """
        # Reset cancellation event
        self.cancel_event.clear()
        
        # Reset other task variables
        self.progress_count, self.end_task, self.show_message_box = self.TASK_VALUES
        self.finished_signal_emitted = False

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

        worker_with_params = partial(worker, file_amount=amount, job=job, cancel_event=self.cancel_event)
        if list_2:
            self.futures = [
                self.executor.submit(worker_with_params, face_detection_tools=tool_pair, old=old_chunk, new=new_chunk)
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
            if future is not None:
                future.add_done_callback(self.worker_done_callback)

    def all_tasks_done(self) -> None:
        """
        Checks if all futures have completed. If they have, shut down the executor and emits done.
        """
        if not self.end_task and all(future.done() for future in self.futures):
            self.emit_done()
            self.end_task = True

    def worker(self, *args: Any) -> None:
        """Worker function to be overridden in subclasses"""
        raise NotImplementedError("Worker function must be implemented in subclasses.")

    def worker_done_callback(self, future: Future) -> None:
        """
        Callback function to handle completion of a worker thread with detailed error reporting.
        Uses a functional approach for cleaner exception handling.
        """
        # Define error handlers based on exception types
        error_handlers = {
            FileNotFoundError: lambda _: self.create_error('file', "File not found. Please check that all input files exist."),
            PermissionError: lambda _: self.create_error('access'),
            MemoryError: lambda _: self.create_error('memory'),
            OSError: lambda _: self.create_error('capacity') if "space" in str(_).lower() else
                            self._display_error(_, "File system error. Check input and output paths."),
            ValueError: lambda _: self._display_error(_, "Invalid data format. Please check input files."),
            (TypeError, AttributeError): lambda _: self._display_error(_, "Data type error. This may indicate a corrupted image file."),
            CancelledError: lambda _: None,  # Silently handle cancelled futures
        }
        
        try:
            future.result()  # This raises any exceptions that occurred during execution
        except Exception as e:
            # Special handling for cancellation - don't show any error
            if isinstance(e, CancelledError):
                return
                
            # Find matching exception handler or use default
            for exc_type, handler in error_handlers.items():
                if isinstance(e, exc_type if isinstance(exc_type, tuple) else exc_type):
                    handler(e)
                    break
            else:
                # Default handler for unspecified exceptions
                self._display_error(e, f"An unexpected error occurred: {type(e).__name__}")
        finally:
            # Check if all futures are done, then emit finished signal
            if self.end_task or all(f.done() for f in self.futures):
                self.all_tasks_done()

    def validate_job(self, job: Job, file_count: Optional[int] = None) -> bool:
        """
        Validate job parameters and available resources.

        Returns:
            bool: True if a job is valid, False if validation failed and error was reported
        """
        if not job.destination:
            return False

        # Check if the destination directory is writable
        if not job.destination_accessible:
            self.create_error('access')
            return False

        # If a file count is provided, check disk capacity
        if file_count is not None:
            total_size = job.approx_byte_size * file_count

            # Check if there is enough space on the disk to process the files
            if job.free_space == 0 or job.free_space < total_size:
                self.create_error('capacity')
                return False

        if self.MEM_FACTOR < 1:
            self.create_error('memory')
            return False

        return True

    def prepare_crop_operation(self, job: Job) -> tuple[Optional[int], Optional[c.Iterable]]:
        """
        Abstract method to be implemented by child classes.
        Should prepare the crop_from_path operation by validating inputs and creating
        the list of items to process.

        Returns:
            tuple: (file_count, chunked_data) or (None, None) if preparation failed
        """
        raise NotImplementedError("Child classes must implement prepare_crop_operation")

    def crop(self, job: Job) -> None:
        """
        Common implementation of the crop_from_path method. Uses a template method pattern.
        """
        # Let child class prepare the operation
        file_count, chunked_data = self.prepare_crop_operation(job)

        if file_count is None or chunked_data is None:
            return

        # Validate common job parameters
        if not self.validate_job(job, file_count):
            return

        # Start the processing - emit signal before setting up futures
        self.reset_task()  # Make sure we're starting fresh
        self.progress_count = 0
        self.progress.emit(self.progress_count, file_count)
        self.started.emit()  # Emit started signal immediately

        # Let child class set up the futures based on its specific worker
        self.set_futures_for_crop(job, file_count, chunked_data)

        # complete_futures is common to all
        self.complete_futures()
        
        # Make sure cancel buttons are enabled right away
        QApplication.processEvents()

    def set_futures_for_crop(self, job: Job, file_count: int, chunked_data: c.Iterable) -> None:
        """
        Abstract method to be implemented by child classes.
        Should set up the futures for the crop_from_path operation.
        """
        raise NotImplementedError("Child classes must implement set_futures_for_crop")

    def emit_done(self) -> None:
        """
        Emits the `finished` signal if it has not already been emitted.
        Uses a cross-thread safe approach.
        """
        if not self.finished_signal_emitted:
            # Set flag to prevent multiple emissions
            self.finished_signal_emitted = True
            self.finished.emit()
            # Use QMetaObject.invokeMethod for cross-thread signal emission
            QMetaObject.invokeMethod(
                self, 
                "finished", 
                Qt.ConnectionType.QueuedConnection
            )
            # Also force a progress update to 100% to ensure the UI is updated
            self.progress_count = 0  # Reset progress count
            QMetaObject.invokeMethod(
                self,
                "progress",
                Qt.ConnectionType.QueuedConnection,
                Q_ARG(int, 0),
                Q_ARG(int, 1)
            )
            # print("Finished signal emitted")
