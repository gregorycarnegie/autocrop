import collections.abc as c
from pathlib import Path
from typing import ClassVar, Optional

import pandas as pd
from PyQt6 import QtCore, QtWidgets

from core import Job, ImageWidget
from core import window_functions as wf
from core.enums import FunctionType, FunctionTabSelectionState, GuiIcon
from core.operation_types import FaceToolPair
from file_types import Photo
from line_edits import NumberLineEdit, PathLineEdit, PathType
from .ui_control_widget import UiCropControlWidget


class UiCropWidget(QtWidgets.QWidget):
    SELECTED: ClassVar[FunctionTabSelectionState] = FunctionTabSelectionState.SELECTED
    NOT_SELECTED: ClassVar[FunctionTabSelectionState] = FunctionTabSelectionState.NOT_SELECTED
    size_policy1: ClassVar[QtWidgets.QSizePolicy] = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                                                          QtWidgets.QSizePolicy.Policy.Fixed)
    size_policy2: ClassVar[QtWidgets.QSizePolicy] = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                                                          QtWidgets.QSizePolicy.Policy.Expanding)
    CHECKBOX_STYLESHEET: ClassVar[str] = """QCheckBox:unchecked{color: red}
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

    def __init__(self, parent: QtWidgets.QWidget, face_tool_list: list[FaceToolPair]):
        super().__init__(parent)
        self.face_tool_list = face_tool_list
        self.size_policy1.setHorizontalStretch(0)
        self.size_policy1.setVerticalStretch(0)

        self.size_policy2.setHorizontalStretch(0)
        self.size_policy2.setVerticalStretch(0)

        self.verticalLayout_100 = wf.setup_vbox(u"verticalLayout_100", self)

        self.horizontalLayout_1 = wf.setup_hbox(u"horizontalLayout_1")
        self.horizontalLayout_2 = wf.setup_hbox(u"horizontalLayout_2")
        self.horizontalLayout_3 = wf.setup_hbox(u"horizontalLayout_3")

        self.folder_icon = wf.create_button_icon(GuiIcon.FOLDER)

        self.destination: Path = Path.home()
        # self.crop_worker = crop_worker

        self.selection_state = self.NOT_SELECTED

        self.inputButton = self.create_nav_button(u"inputButton")
        self.inputLineEdit = self.create_str_line_edit(u"inputLineEdit", PathType.FOLDER)

        self.destinationButton = self.create_nav_button(u"destinationButton")
        self.destinationLineEdit = self.create_str_line_edit(u"destinationLineEdit", PathType.FOLDER)

        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                            QtWidgets.QSizePolicy.Policy.Expanding)
        size_policy.setHorizontalStretch(1)
        size_policy.setVerticalStretch(1)

        self.imageWidget = ImageWidget()
        self.imageWidget.setObjectName(u"imageWidget")
        wf.apply_size_policy(self.imageWidget, size_policy, min_size=QtCore.QSize(0, 0),
                             max_size=QtCore.QSize(16_777_215, 16_777_215))
        self.imageWidget.setStyleSheet(u"")
        self.horizontalLayout = wf.setup_hbox(u"horizontalLayout", self.imageWidget)
        self.controlWidget = UiCropControlWidget(self.imageWidget)
        self.controlWidget.setObjectName(u"controlWidget")
        self.horizontalLayout.addWidget(self.controlWidget)

        self.toggleCheckBox = self.create_checkbox(u"toggleCheckBox")
        self.toggleCheckBox.setChecked(True)
        self.mfaceCheckBox = self.create_checkbox(u"mfaceCheckBox")
        self.tiltCheckBox = self.create_checkbox(u"tiltCheckBox")
        self.exposureCheckBox = self.create_checkbox(u"exposureCheckBox")
        # Add other common initialization code here if needed

    def retranslateUi(self):
        # This method should be overridden by the specific classes to provide their own implementations
        pass

    def create_checkbox(self, name: str) -> QtWidgets.QCheckBox:
        check_box = QtWidgets.QCheckBox()
        check_box.setObjectName(name)
        wf.apply_size_policy(check_box, self.size_policy1)
        check_box.setStyleSheet(self.CHECKBOX_STYLESHEET)
        return check_box

    def create_str_line_edit(self, name: str, path_type: PathType) -> PathLineEdit:
        line_edit = PathLineEdit(path_type=path_type)
        line_edit.setObjectName(name)
        wf.apply_size_policy(line_edit, self.size_policy1)
        line_edit.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhUrlCharactersOnly)
        return line_edit

    def create_nav_button(self, name: str) -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton()
        button.setObjectName(name)
        wf.apply_size_policy(button, self.size_policy1, min_size=QtCore.QSize(186, 30))
        return button

    def setup_path_line_edit(self, name: str, path_type: PathType = PathType.FOLDER) -> PathLineEdit:
        """Only subclasses of the CustomCropWidget class should implement this method"""
        line_edit = PathLineEdit(path_type=path_type, parent=self)
        line_edit.setMinimumSize(QtCore.QSize(0, 24))
        line_edit.setMaximumSize(QtCore.QSize(16_777_215, 24))
        line_edit.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhUrlCharactersOnly)
        line_edit.setObjectName(name)
        return line_edit

    def reload_widgets(self) -> None:
        """Only subclasses of the CustomCropWidget class should implement this method"""
        pass

    def disable_buttons(self) -> None:
        """Only subclasses of the CustomCropWidget class should implement this method"""
        pass

    def connect_checkbox(self, checkbox: QtWidgets.QCheckBox) -> None:
        """Only subclasses of the CustomCropWidget class should implement this method"""
        checkbox.stateChanged.connect(lambda: self.reload_widgets())
        match checkbox:
            case self.mfaceCheckBox:
                checkbox.clicked.connect(
                    lambda: wf.uncheck_boxes(self.exposureCheckBox, self.tiltCheckBox))
            case self.exposureCheckBox | self.tiltCheckBox:
                checkbox.clicked.connect(lambda: wf.uncheck_boxes(self.mfaceCheckBox))
            case _:
                pass

    def connect_input_widgets(self, *input_widgets: QtWidgets.QWidget) -> None:
        """Only subclasses of the CustomCropWidget class should implement this method"""
        for input_widget in input_widgets:
            match input_widget:
                case NumberLineEdit() | PathLineEdit():
                    input_widget.textChanged.connect(lambda: self.reload_widgets())
                    input_widget.textChanged.connect(lambda: self.disable_buttons())
                case QtWidgets.QDial():
                    input_widget.valueChanged.connect(lambda: self.reload_widgets())
                case QtWidgets.QCheckBox():
                    self.connect_checkbox(input_widget)
                case _:
                    pass

    @staticmethod
    def _get_file_names_without_extension(directory: Path) -> c.Set[str]:
        """Return a set of filenames (without extensions) in the given directory."""
        return {p.stem for p in directory.iterdir() if p.is_file()}

    @staticmethod
    def _check_matching_files(destination: Path, filenames: c.Set[str], extensions: c.Set[str]) -> bool:
        """Recursively check if destination contains any files with the given extensions that match the filenames."""
        return any(p.is_file()
                   and p.suffix.lower() in extensions
                   and p.stem in filenames
                   for p in destination.rglob('*'))

    @staticmethod
    def _create_unique_folder(base_path: Path, valid_extensions: c.Set[str]) -> Path:
        print(f"base_path: {base_path}")
        if not base_path.exists():
            return base_path
        if all(file.suffix not in valid_extensions
               for file in base_path.iterdir()
               if file.is_file()):
            return base_path  # Return the original path if no files with the specified extensions are found
        counter = 1
        new_path = base_path if base_path.name else base_path / 'results'
        while new_path.exists():
            new_path = base_path.with_name(f"{base_path.name}_{counter}")
            counter += 1
        new_path.mkdir(parents=True, exist_ok=True)
        return new_path

    def _handle_folder_path(self, destination: Path, folder_path: Path, extensions: c.Set[str]) -> Path:
        if destination.exists() and destination.name:
            folder_filenames = self._get_file_names_without_extension(folder_path)
            if self._check_matching_files(destination, folder_filenames, extensions):
                destination = destination.with_name(f'{destination.name}_CROPS')
            destination = self._create_unique_folder(destination, extensions)
        return destination

    def _handle_video_path(self, destination: Path,
                           video_path: Path,
                           extensions: c.Set[str]) -> Path:
        
        if destination.exists():
            folder_filenames = self._get_file_names_without_extension(video_path.parent)
            if any(video_path.name.lower() in name.lower() for name in folder_filenames):
                destination = destination.with_name(f'{destination.name}_CROPS')
            destination = self._create_unique_folder(destination, extensions)
        return destination

    def create_job(self, function_type: Optional[FunctionType] = None,
                   photo_path: Optional[Path] = None,
                   destination: Optional[Path] = None,
                   folder_path: Optional[Path] = None,
                   table: Optional[pd.DataFrame] = None,
                   column1: Optional[QtWidgets.QComboBox] = None,
                   column2: Optional[QtWidgets.QComboBox] = None,
                   video_path: Optional[Path] = None,
                   start_position: Optional[float] = None,
                   stop_position: Optional[float] = None) -> Job:
        """Only subclasses of the CustomCropWidget class should implement this method"""
        if function_type is not None:
            match function_type:
                case FunctionType.FOLDER | FunctionType.MAPPING:
                    if destination and folder_path:
                        destination = self._handle_folder_path(destination, folder_path, Photo.SAVE_TYPES)
                case FunctionType.VIDEO:
                    if destination and video_path:
                        destination = self._handle_video_path(destination, video_path, Photo.SAVE_TYPES)
                case FunctionType.PHOTO | FunctionType.FRAME:
                    pass

        self.destination = destination if destination is not None else self.destination

        return Job(self.controlWidget.widthLineEdit.value(),
                   self.controlWidget.heightLineEdit.value(),
                   self.exposureCheckBox.isChecked(),
                   self.mfaceCheckBox.isChecked(),
                   self.tiltCheckBox.isChecked(),
                   self.controlWidget.sensitivityDial.value(),
                   self.controlWidget.fpctDial.value(),
                   self.controlWidget.gammaDial.value(),
                   self.controlWidget.topDial.value(),
                   self.controlWidget.bottomDial.value(),
                   self.controlWidget.leftDial.value(),
                   self.controlWidget.rightDial.value(),
                   self.controlWidget.radio_tuple,
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
        """Only subclasses of the CustomCropWidget class should implement this method"""
        f_name = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Directory', Photo.default_directory)
        line_edit.setText(f_name)
        match self.inputLineEdit.path_type:
            case PathType.FOLDER:
                self.load_data()
            case _:
                pass

    def load_data(self) -> None:
        """Only subclasses of the CustomCropWidget class should implement this method"""
        pass
