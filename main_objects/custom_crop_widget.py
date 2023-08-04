from multiprocessing import Process
from typing import Optional, Callable, Any

import pandas as pd
from PyQt6 import QtWidgets, QtCore, QtGui

from core import CustomDialWidget, ExtWidget, Job, window_functions
from file_types import Photo
from line_edits import PathLineEdit, NumberLineEdit
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
        self.frame = QtWidgets.QFrame(parent=self)
        self.frame.setStyleSheet('background: #1f2c33')
        self.frame.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.frame.setObjectName('frame')

        self.folderLineEdit: Optional[PathLineEdit] = None
        self.CHECKBOX_STYLESHEET = CHECKBOX_STYLESHEET
        self.exposureCheckBox = QtWidgets.QCheckBox()
        self.mfaceCheckBox = QtWidgets.QCheckBox()
        self.tiltCheckBox = QtWidgets.QCheckBox()

        self.mfaceCheckBox.setObjectName('mfaceCheckBox')
        self.tiltCheckBox.setObjectName('tiltCheckBox')
        self.exposureCheckBox.setObjectName('exposureCheckBox')

        self.exposureCheckBox.setStyleSheet(self.CHECKBOX_STYLESHEET)
        self.mfaceCheckBox.setStyleSheet(self.CHECKBOX_STYLESHEET)
        self.tiltCheckBox.setStyleSheet(self.CHECKBOX_STYLESHEET)

        self.cropButton = QtWidgets.QPushButton()
        self.cropButton.setMinimumSize(QtCore.QSize(0, 24))
        self.cropButton.setMaximumSize(QtCore.QSize(16777215, 24))
        self.cropButton.setText('')
        crop_icon = QtGui.QIcon()
        crop_icon.addPixmap(QtGui.QPixmap('resources\\icons\\crop.svg'), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.cropButton.setIcon(crop_icon)
        self.cropButton.setObjectName('cropButton')

        self.destinationLineEdit = PathLineEdit()
        self.destinationLineEdit.setMinimumSize(QtCore.QSize(0, 24))
        self.destinationLineEdit.setMaximumSize(QtCore.QSize(16777215, 24))
        self.destinationLineEdit.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhUrlCharactersOnly)
        self.destinationLineEdit.setObjectName('destinationLineEdit')

        self.destinationButton = QtWidgets.QPushButton(parent=self)
        self.destinationButton.setMinimumSize(QtCore.QSize(124, 24))
        self.destinationButton.setMaximumSize(QtCore.QSize(16777215, 24))
        destination_icon = QtGui.QIcon()
        destination_icon.addPixmap(QtGui.QPixmap('resources\\icons\\folder.svg'), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.destinationButton.setIcon(destination_icon)
        self.destinationButton.setObjectName('destinationButton')

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
        
        self.selection_state = FunctionTabSelectionState.NOT_SELECTED

    def reload_widgets(self) -> None:
        """Only sublasses of this class should implement this method"""
        pass 
    
    def disable_buttons(self) -> None:
        """Only sublasses of this class should implement this method"""
        pass 

    def connect_checkboxs(self, input_widget: QtWidgets.QCheckBox):
        input_widget.stateChanged.connect(lambda: self.reload_widgets())
        match input_widget:
            case self.mfaceCheckBox:
                input_widget.clicked.connect(
                    lambda: window_functions.uncheck_boxes(self.exposureCheckBox, self.tiltCheckBox))
            case self.exposureCheckBox | self.tiltCheckBox:
                input_widget.clicked.connect(lambda: window_functions.uncheck_boxes(self.mfaceCheckBox))
            case _: pass

    def connect_input_widgets(self, *input_widgets: QtWidgets.QWidget) -> None:
        """Only sublasses of this class should implement this method"""
        for input_widget in input_widgets:
            match input_widget:
                case NumberLineEdit() | PathLineEdit():
                    input_widget.textChanged.connect(lambda: self.reload_widgets())
                    input_widget.textChanged.connect(lambda: self.disable_buttons())
                case QtWidgets.QDial():
                    input_widget.valueChanged.connect(lambda: self.reload_widgets())
                case QtWidgets.QCheckBox():
                    self.connect_checkboxs(input_widget)
                case _: pass

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
