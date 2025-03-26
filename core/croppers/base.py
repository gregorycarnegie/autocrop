from threading import Lock
from typing import ClassVar, Optional

import psutil
from PyQt6.QtCore import pyqtSignal, QObject

from ui import utils as ut

TOTAL_MEMORY, MEM_THRESHOLD = psutil.virtual_memory().total, 2147483648
MEM_FACTOR = TOTAL_MEMORY // MEM_THRESHOLD

class Cropper(QObject):
    """
    A class that manages image-cropping tasks using a thread pool.
    """

    THREAD_NUMBER: ClassVar[int] = min(psutil.cpu_count(), MEM_FACTOR, 8)
    MEM_FACTOR = MEM_FACTOR
    TASK_VALUES: ClassVar[tuple[int, bool, bool]] = 0, False, True

    started = pyqtSignal()
    finished = pyqtSignal()
    error = pyqtSignal()
    progress = pyqtSignal(int, int)

    def __init__(self, parent: Optional[QObject] = None):
        super(Cropper, self).__init__(parent)

        # # Task state
        self.progress_count, self.end_task, self.show_message_box = self.TASK_VALUES
        self.finished_signal_emitted = False

        # # Synchronization
        self.lock = Lock()

    def __repr__(self) -> str:
        return f"<{type(self).__name__}>"

    def reset_task(self) -> None:
        """
        Resets the task-specific variables to their default values.
        """

        self.progress_count, self.end_task, self.show_message_box = self.TASK_VALUES
        self.finished_signal_emitted = False

    def emit_done(self) -> None:
        """
        Emits the `finished` signal if it has not already been emitted.
        """
        if not self.finished_signal_emitted:
            self.finished.emit()
            self.finished_signal_emitted = True

    def _update_progress(self, file_amount: int) -> None:
        """
        Increments the progress count in a thread-safe manner
        and emits the progress signal accordingly.
        """
        with self.lock:
            self.progress_count += 1

            if self.progress_count == file_amount:
                # Either we've reached or exceeded the target file amount
                self.progress.emit(file_amount, file_amount)
                self.emit_done()
            elif self.progress_count < file_amount:
                self.progress.emit(self.progress_count, file_amount)

    def _display_error(self, exception: BaseException, suggestion: str) -> None:
        """
        Displays an error message using the window_functions module
        and emits the error signal.
        """
        
        ut.show_error_box(f"{exception}\n{suggestion}")
        self.error.emit()
        self.end_task = True
        return 
    
    def access_error(self) -> None:
        """
        Raises a permission error if the destination directory is not writable.
        """
        return self._display_error(
            PermissionError("Permission denied. Please check file permissions."),
            "Please check the destination directory and try again."
        )

    def amount_error(self) -> None:
        """
        Raises a file-not-found error if the source directory has no compatible files.
        """
        return self._display_error(
            FileNotFoundError("Source directory has no compatible files."),
            "Please check the source directory and try again."
        )
    
    def capacity_error(self) -> None:
        """
        Raises an OSError if the disk does not have enough space.
        """
        return self._display_error(
            OSError("Not enough space on disk."),
            "Please free up some space and try again."
        )
    
    def file_error(self, message: str = "Please check the file path and try again.") -> None:
        """
        Raises a file-not-found error if a file does not exist.
        """
        return self._display_error(
            FileNotFoundError("File does not exist."),
            message
        )
    
    def file_type_error(self) -> None:
        """
        Raises a TypeError if the file type is unsupported.
        """
        return self._display_error(
            TypeError("File type is not supported."),
            "Please check the file type and try again."
        )
    
    def directory_error(self) -> None:
        """
        Raises a file-not-found error if the directory does not exist.
        """
        return self._display_error(
            FileNotFoundError("Directory does not exist."),
            "Please check the directory path and try again."
        )
    
    def memory_error(self) -> None:
        """
        Raises a MemoryError if the system does not have enough RAM to complete the task.
        """
        return self._display_error(
            MemoryError("Not enough memory to complete task."),
            "Please free up some memory and try again."
        )
    
    def thread_error(self) -> None:
        """
        Raises a RuntimeError if the thread limit has been reached.
        """
        return self._display_error(
            RuntimeError("Thread limit reached."),
            "Please try again later."
        )
