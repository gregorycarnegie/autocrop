from typing import Optional

import pandas as pd
from PyQt6 import QtWidgets, QtCore, QtGui

from core import CustomDialWidget, Cropper, ExtWidget, FunctionTabSelectionState, Job, window_functions
from file_types import Photo
from line_edits import PathLineEdit, NumberLineEdit, PathType
from .enums import ButtonType

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
        self.horizontalLayout_1 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_1.setObjectName('horizontalLayout_1')
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName('horizontalLayout_2')
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName('horizontalLayout_3')
        self.frame = QtWidgets.QFrame(parent=self)
        self.frame.setStyleSheet('background: #1f2c33')
        self.frame.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.frame.setObjectName('frame')
        self.folderLineEdit: Optional[PathLineEdit] = None
        self.CHECKBOX_STYLESHEET = CHECKBOX_STYLESHEET
        self.exposureCheckBox = self.setup_checkbox('exposureCheckBox')
        self.mfaceCheckBox = self.setup_checkbox('mfaceCheckBox')
        self.tiltCheckBox = self.setup_checkbox('tiltCheckBox')
        self.cropButton = self.setup_process_button('cropButton', 'crop', ButtonType.PROCESS_BUTTON)
        self.destinationLineEdit = self.setup_path_line_edit('destinationLineEdit')
        self.destinationButton = self.setup_process_button('destinationButton', 'folder', ButtonType.NAVIGATION_BUTTON)
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

    def setup_process_button(self, name: str, icon_name: str,  button_type: ButtonType) -> QtWidgets.QPushButton: 
        match button_type:
            case ButtonType.PROCESS_BUTTON:
                button = QtWidgets.QPushButton(parent=self.frame)
                button.setMinimumSize(QtCore.QSize(0, 24))
            case ButtonType.NAVIGATION_BUTTON:
                button = QtWidgets.QPushButton(parent=self)
                button.setMinimumSize(QtCore.QSize(124, 24))
        button.setMaximumSize(QtCore.QSize(16777215, 24))
        button.setText('')
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(f'resources\\icons\\{icon_name}.svg'), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        button.setIcon(icon)
        button.setObjectName(name)
        return button

    def setup_path_line_edit(self, name: str, path_type: PathType = PathType.FOLDER) -> PathLineEdit:
        """Only sublasses of this class should implement this method"""
        line_edit = PathLineEdit(path_type=path_type, parent=self)
        line_edit.setMinimumSize(QtCore.QSize(0, 24))
        line_edit.setMaximumSize(QtCore.QSize(16777215, 24))
        line_edit.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhUrlCharactersOnly)
        line_edit.setObjectName(name)
        return line_edit

    def setup_checkbox(self, name: str) -> QtWidgets.QCheckBox:
        """Only sublasses of this class should implement this method"""
        checkbox = QtWidgets.QCheckBox(self.frame)
        checkbox.setObjectName(name)
        checkbox.setStyleSheet(self.CHECKBOX_STYLESHEET)
        return checkbox

    def reload_widgets(self) -> None:
        """Only sublasses of this class should implement this method"""
        pass

    def disable_buttons(self) -> None:
        """Only sublasses of this class should implement this method"""
        pass

    def connect_checkboxs(self, input_widget: QtWidgets.QCheckBox) -> None:
        """Only sublasses of this class should implement this method"""
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
        """Only sublasses of this class should implement this method"""
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

    def open_folder(self, line_edit: PathLineEdit) -> None:
        """Only sublasses of this class should implement this method"""
        f_name = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Directory', Photo().default_directory)
        line_edit.setText(f_name)
        if line_edit is self.folderLineEdit:
            self.load_data()

    def load_data(self) -> None:
        """Only sublasses of this class should implement this method"""
        pass
