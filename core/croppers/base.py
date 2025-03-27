from threading import Lock
from typing import ClassVar, Literal, Optional

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

    @staticmethod
    def create_error(error_type: Literal['access', 'amount', 'capacity', 'file', 'file_type', 'directory', 'memory', 'thread'], custom_message: Optional[str] = None) -> tuple[Exception, str]:
        errors = {
            'access': (
                PermissionError("Permission denied."),
                custom_message or "Please check file permissions."
            ),
            'amount': (
                FileNotFoundError("Source directory has no compatible files."),
                custom_message or "Please check the source directory and try again."
            ),
            # Add other error types
            'capacity': (
                OSError("Not enough space on disk."),
                custom_message or "Please free up some space and try again."
            ),
            'file': (
                FileNotFoundError("File does not exist."),
                custom_message or "Please check the file path and try again."
            ),
            'file_type': (
                TypeError("File type is not supported."),
                custom_message or "Please check the file type and try again."
            ),
            'directory': (
                FileNotFoundError("Directory does not exist."),
                custom_message or "Please check the directory path and try again."
            ),
            'memory': (
                MemoryError("Not enough memory to complete task."),
                custom_message or "Please free up some memory and try again."
            ),
            'thread': (
                RuntimeError("Thread limit reached."),
                custom_message or "Please try again later."
            )
        }
        return errors.get(error_type, (Exception("Unknown error"), "An error occurred"))

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
