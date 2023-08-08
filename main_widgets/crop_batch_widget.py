from multiprocessing import Process
from typing import Optional, Callable, Any, Tuple

from PyQt6 import QtCore, QtWidgets

from core import Cropper, CustomDialWidget, ExtWidget, Job
from line_edits import NumberLineEdit
from .custom_crop_widget import CustomCropWidget
from .enums import ButtonType

PROGRESSBAR_STEPS = 1_000


class CropBatchWidget(CustomCropWidget):
    def __init__(self, crop_worker: Cropper,
                 width_line_edit: NumberLineEdit,
                 height_line_edit: NumberLineEdit,
                 ext_widget: ExtWidget,
                 sensitivity_dial_area: CustomDialWidget,
                 face_dial_area: CustomDialWidget,
                 gamma_dial_area: CustomDialWidget,
                 top_dial_area: CustomDialWidget,
                 bottom_dial_area: CustomDialWidget,
                 left_dial_area: CustomDialWidget,
                 right_dial_area: CustomDialWidget,
                 parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(crop_worker, width_line_edit, height_line_edit, ext_widget, sensitivity_dial_area,
                         face_dial_area, gamma_dial_area, top_dial_area, bottom_dial_area, left_dial_area,
                         right_dial_area, parent)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName('horizontalLayout_4')
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName('horizontalLayout_5')
        self.cancelButton = self.setup_process_button('cancelButton', 'cancel', ButtonType.PROCESS_BUTTON)
        self.progressBar = QtWidgets.QProgressBar(parent=self.frame)
        self.progressBar.setMinimumSize(QtCore.QSize(0, 12))
        self.progressBar.setMaximumSize(QtCore.QSize(16777215, 12))
        self.progressBar.setRange(0, PROGRESSBAR_STEPS)
        self.progressBar.setValue(0)
        self.progressBar.setTextVisible(False)

    def connect_crop_worker(self) -> None:
        """Only sublasses of the CropBatchWidget class should implement this method"""
        pass

    def update_progress(self, data: Tuple[int, int]) -> None:
        """Only sublasses of the CropBatchWidget class should implement this method"""
        x, y = data
        self.progressBar.setValue(int(PROGRESSBAR_STEPS * x / y))
        QtWidgets.QApplication.processEvents()

    
    @staticmethod
    def run_batch_process(function: Callable[..., Any],
                          reset_worker_func: Callable[..., Any],
                          job: Job) -> None:
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
