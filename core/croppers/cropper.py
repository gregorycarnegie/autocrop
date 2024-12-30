from concurrent.futures import Future, ThreadPoolExecutor
from multiprocessing import cpu_count
from threading import Lock
from typing import ClassVar, Optional

from PyQt6.QtCore import pyqtSignal, QObject

from core import window_functions as wf


class Cropper(QObject):
    THREAD_NUMBER: ClassVar[int] = min(cpu_count(), 8)
    TASK_VALUES: ClassVar[tuple[int, bool, bool]] = 0, False, True

    started = pyqtSignal()
    finished = pyqtSignal()
    error = pyqtSignal()
    progress = pyqtSignal(int, int)

    def __init__(self, parent: Optional[QObject] = None):
        super(Cropper, self).__init__(parent)
        self.executor: Optional[ThreadPoolExecutor] = None
        self.futures: list[Future] = []
        self.bar_value, self.end_task, self.message_box = self.TASK_VALUES
        self.finished_signal_emitted = False
        self.lock = Lock()

    def __repr__(self):
        return """{}
        Class: {}
        Threads: {}
        Bar Value: {}
        End Task: {}
        Message Box: {}
        """.format(
            super(Cropper, self).__repr__(),
            type(self).__name__,
            self.THREAD_NUMBER,
            self.bar_value,
            self.end_task,
            self.message_box
        )

    def reset_task(self) -> None:
        """
        Resets the task values based on the provided function type.

        Returns:
            None
        """

        self.bar_value, self.end_task, self.message_box = self.TASK_VALUES
        self.finished_signal_emitted = False

    def emit_done(self) -> None:
        if not self.finished_signal_emitted:
            self.finished.emit()
            self.finished_signal_emitted = True

    def _update_progress(self, file_amount: int) -> None:
        with self.lock:
            self.bar_value += 1
            if self.bar_value == file_amount:
                self.progress.emit(file_amount, file_amount)
                self.emit_done()
            elif self.bar_value < file_amount:
                self.progress.emit(self.bar_value, file_amount)

    def terminate(self) -> None:
        """
        Terminates the specified series of tasks.

        Returns:
            None
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

    def _display_error(self, error: BaseException, action: str) -> None:
        wf.show_error_box(str(error, action))
        self.error.emit()
        self.end_task = True
        return 
    
    def access_error(self) -> None:
        return self._display_error(
            PermissionError("Destination directory is not writable."),
            "Please check the destination directory and try again."
        )

    def amount_error(self) -> None:
        return self._display_error(
            FileNotFoundError("Source directory has no compatible files."),
            "Please check the source directory and try again."
        )
    
    def capacity_error(self) -> None:
        return self._display_error(
            OSError("Not enough space on disk."),
            "Please free up some space and try again."
        )
    
    def file_error(self) -> None:
        return self._display_error(
            FileNotFoundError("File does not exist."),
            "Please check the file path and try again."
        )
    
    def file_type_error(self) -> None:
        return self._display_error(
            TypeError("File type is not supported."),
            "Please check the file type and try again."
        )
    
    def directory_error(self) -> None:
        return self._display_error(
            FileNotFoundError("Directory does not exist."),
            "Please check the directory path and try again."
        )
    
    def all_tasks_done(self) -> None:
        """
        Check if all futures have completed. This method should be called from the main thread.
        """
        if not self.end_task and all(future.done() for future in self.futures):
            if self.executor:
                self.executor.shutdown(wait=False)  # Non-blocking shutdown
            self.emit_done()
            self.end_task = True

    def worker_done_callback(self, future: Future) -> None:
        """
        Callback function to handle completion of a worker thread.
        """
        try:
            future.result()  # This raises any exceptions that occurred during execution
        except Exception as e:
            self.error.emit(f"Error in worker execution: {str(e)}")
        finally:
            # Check if all futures are done, then emit finished signal
            if all(f.done() for f in self.futures):
                self.all_tasks_done()
