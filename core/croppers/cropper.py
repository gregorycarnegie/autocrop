from concurrent.futures import Future, ThreadPoolExecutor
from multiprocessing import cpu_count
from typing import ClassVar, Optional

from PyQt6.QtCore import pyqtSignal, QObject

from core import window_functions as wf


class Cropper(QObject):
    THREAD_NUMBER: ClassVar[int] = min(cpu_count(), 8)
    TASK_VALUES: ClassVar[tuple[int, bool, bool]] = 0, False, True

    started, finished, progress = pyqtSignal(), pyqtSignal(), pyqtSignal(object)
    error = pyqtSignal()

    def __init__(self, parent: Optional[QObject] = None):
        super(Cropper, self).__init__(parent)
        self.executor: Optional[ThreadPoolExecutor] = None
        self.futures: list[Future] = []
        self.bar_value, self.end_task, self.message_box = self.TASK_VALUES

    def reset_task(self):
        """
        Resets the task values based on the provided function type.

        Returns:
            None
        """

        self.bar_value, self.end_task, self.message_box = self.TASK_VALUES

    def _update_progress(self, file_amount: int) -> None:
        self.bar_value += 1
        if self.bar_value == file_amount:
            self.progress.emit((file_amount, file_amount))
            self.finished.emit()
        elif self.bar_value < file_amount:
            self.progress.emit((self.bar_value, file_amount))

    def terminate(self) -> None:
        """
        Terminates the specified series of tasks.

        Returns:
            None
        """

        self.end_task = True
        self.finished.emit()

        if self.executor:
            for future in self.futures:
                future.cancel()
            self.futures.clear()
            self.executor.shutdown(wait=True)
            self.executor = None

    def display_error(self, *args: str):
        wf.show_error_box(*args)
        self.error.emit()
        self.end_task = True
        return
    
    def access_error(self):
        return self.display_error(
            str(PermissionError("Destination directory is not writable.")),
            "Please check the destination directory and try again.",
        )
    
    def capacity_error(self):
        return self.display_error(
            str(OSError("Not enough space on disk.")),
            "Please free up some space and try again.",
        )