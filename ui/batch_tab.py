import collections.abc as c
from multiprocessing import Process
from typing import Any, ClassVar, Optional

from PyQt6 import QtCore, QtWidgets

from core import Job
from ui import ui_utils as wf
from .crop_widget import UiCropWidget
from .enums import GuiIcon


class UiCropBatchWidget(UiCropWidget):
    PROGRESSBAR_STEPS: ClassVar[int] = 1_000

    def __init__(self, object_name: str, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self.setObjectName(object_name)

        self.page_1 = QtWidgets.QWidget()
        self.page_1.setObjectName(u"page_1")

        self.page_2 = QtWidgets.QWidget()
        self.page_2.setObjectName(u"page_2")

        self.verticalLayout_200 = wf.setup_vbox(u"verticalLayout_200", self.page_1)
        self.verticalLayout_300 = wf.setup_vbox(u"verticalLayout_300", self.page_2)

        self.horizontalLayout_4 = wf.setup_hbox(u'horizontalLayout_4')
        
        self.cancelButton = QtWidgets.QPushButton()
        self.cancelButton.setObjectName(u"cancelButton")
        self.cancelButton.setMinimumSize(QtCore.QSize(0, 40))
        self.cancelButton.setMaximumSize(QtCore.QSize(16_777_215, 40))
        icon = wf.create_button_icon(GuiIcon.CANCEL)
        self.cancelButton.setIcon(icon)
        self.cancelButton.setIconSize(QtCore.QSize(18, 18))

        self.progressBar = self.create_progress_bar(u"progressBar")

    def connect_crop_worker(self) -> None:
        """Only sublasses of the CropBatchWidget class should implement this method"""
        pass

    @classmethod
    def create_progress_bar(cls, name: str, parent: Optional[QtWidgets.QWidget] = None) -> QtWidgets.QProgressBar:
        progress_bar = QtWidgets.QProgressBar() if parent is None else QtWidgets.QProgressBar(parent)
        progress_bar.setObjectName(name)
        progress_bar.setMinimumSize(QtCore.QSize(0, 12))
        progress_bar.setMaximumSize(QtCore.QSize(16_777_215, 12))
        progress_bar.setRange(0, cls.PROGRESSBAR_STEPS)
        progress_bar.setValue(0)
        progress_bar.setTextVisible(False)
        return progress_bar
    
    def update_progress(self, x: int, y:int) -> None:
        """Only sublasses of the CropBatchWidget class should implement this method"""
        self.progressBar.setValue(int(self.PROGRESSBAR_STEPS * x / y))
        QtWidgets.QApplication.processEvents()

    @staticmethod
    def run_batch_process(job: Job, *,
                          function: c.Callable[..., Any],
                          reset_worker_func: c.Callable[..., Any]) -> None:
        """Only sublasses of the CropBatchWidget class should implement this method"""
        reset_worker_func()
        process = Process(target=function, daemon=True, args=(job,))
        process.run()

    @staticmethod
    def cancel_button_operation(cancel_button: QtWidgets.QPushButton, *crop_buttons: QtWidgets.QPushButton) -> None:
        """Only sublasses of the CropBatchWidget class should implement this method"""
        cancel_button.setDisabled(True)
        for crop_button in crop_buttons:
            crop_button.setEnabled(True)
