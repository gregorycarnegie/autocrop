from multiprocessing import Process
from typing import Optional, Callable, Any

import pandas as pd
from PyQt6 import QtWidgets

from core import CustomDialWidget, ExtWidget, Job
from file_types import Photo
from line_edits import PathLineEdit, NumberLineEdit, LineEditState
from .cropper import Cropper
from .enums import FunctionTabSelectionState

CHECKBOX_STYLESHEET = """QCheckBox:unchecked{color: red}
        QCheckBox:checked{color: white}
        QCheckBox::indicator{
                width: 20px;
                height: 20px;
        }
        QCheckBox::indicator:checked{
                image: url(resources/icons/checkbox_checked.svg);
        }
        QCheckBox::indicator:unchecked{
                image: url(resources/icons/checkbox_unchecked.svg);
        }
        QCheckBox::indicator:checked:hover{
                image: url(resources/icons/checkbox_checked_hover.svg);
        }
        QCheckBox::indicator:unchecked:hover{
                image: url(resources/icons/checkbox_unchecked_hover.svg);
        }"""


class CustomCropWidget(QtWidgets.QWidget):
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
        super().__init__(parent)
        self.folderLineEdit: Optional[PathLineEdit] = None
        
        self.progressBar = QtWidgets.QProgressBar()
        self.crop_worker = crop_worker
        self.widthLineEdit = width_line_edit
        self.heightLineEdit = height_line_edit
        self.extWidget = ext_widget
        self.sensitivity_dialArea = sensitivity_dial_area
        self.face_dialArea = face_dial_area
        self.gamma_dialArea = gamma_dial_area
        self.top_dialArea = top_dial_area
        self.bottom_dialArea = bottom_dial_area
        self.left_dialArea = left_dial_area
        self.right_dialArea = right_dial_area
        self.CHECKBOX_STYLESHEET = CHECKBOX_STYLESHEET
        self.selection_state = FunctionTabSelectionState.NOT_SELECTED

    def update_progress(self, value: int) -> None:
        self.progressBar.setValue(value)
        QtWidgets.QApplication.processEvents()

    @staticmethod
    def run_batch_process(function: Callable[..., Any],
                          reset_worker_func: Callable[..., Any],
                          job: Job) -> None:
        reset_worker_func()
        process = Process(target=function, daemon=True, args=(job,))
        process.run()

    def create_job(self, exposure: QtWidgets.QCheckBox,
                   multi: QtWidgets.QCheckBox,
                   tilt: QtWidgets.QCheckBox,
                   photo_path: Optional[PathLineEdit] = None,
                   destination: Optional[PathLineEdit] = None,
                   folder_path: Optional[PathLineEdit] = None,
                   table: Optional[pd.DataFrame] = None,
                   column1: Optional[QtWidgets.QComboBox] = None,
                   column2: Optional[QtWidgets.QComboBox] = None,
                   video_path: Optional[PathLineEdit] = None,
                   start_position: Optional[float] = None,
                   stop_position: Optional[float] = None) -> Job:
        return Job(self.widthLineEdit,
                   self.heightLineEdit,
                   exposure,
                   multi,
                   tilt,
                   self.sensitivity_dialArea.dial,
                   self.face_dialArea.dial,
                   self.gamma_dialArea.dial,
                   self.top_dialArea.dial,
                   self.bottom_dialArea.dial,
                   self.left_dialArea.dial,
                   self.right_dialArea.dial,
                   (self.extWidget.radioButton_1, self.extWidget.radioButton_2,
                    self.extWidget.radioButton_3, self.extWidget.radioButton_4,
                    self.extWidget.radioButton_5, self.extWidget.radioButton_6),
                    photo_path=photo_path,
                    destination=destination,
                    folder_path=folder_path,
                    table=table,
                    column1=column1,
                    column2=column2,
                    video_path=video_path,
                    start_position=start_position,
                    stop_position=stop_position)

    @staticmethod
    def cancel_button_operation(cancel_button: QtWidgets.QPushButton, *crop_buttons: QtWidgets.QPushButton):
        cancel_button.setDisabled(True)
        for crop_button in crop_buttons:
            crop_button.setEnabled(True)

    def open_folder(self, line_edit: PathLineEdit) -> None:
        f_name = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Directory', Photo().default_directory)
        line_edit.setText(f_name)
        if line_edit is self.folderLineEdit:
                self.load_data()

    def load_data(self) -> None:
        """Only sublasses of this class should implement this method"""
        pass
