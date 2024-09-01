from multiprocessing import cpu_count
from typing import ClassVar, Optional

from PyQt6.QtCore import pyqtSignal, QObject


class Cropper(QObject):
    THREAD_NUMBER: ClassVar[int] = min(cpu_count(), 8)
    TASK_VALUES: ClassVar[tuple[int, bool, bool]] = 0, False, True

    started, finished, progress = pyqtSignal(), pyqtSignal(), pyqtSignal(object)

    def __init__(self, parent: Optional[QObject] = None):
        super(Cropper, self).__init__(parent)
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
