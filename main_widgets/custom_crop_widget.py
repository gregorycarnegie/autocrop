from typing import Optional, Set, ClassVar
from pathlib import Path

import pandas as pd
from PyQt6 import QtWidgets, QtCore, QtGui

from core import CustomDialWidget, Cropper, ExtWidget, FunctionTabSelectionState, ImageWidget, Job, window_functions
from file_types import Photo
from line_edits import PathLineEdit, NumberLineEdit, PathType
from .enums import ButtonType


class CustomCropWidget(QtWidgets.QWidget):
    SELECTED: ClassVar[FunctionTabSelectionState] = FunctionTabSelectionState.SELECTED
    NOT_SELECTED: ClassVar[FunctionTabSelectionState] = FunctionTabSelectionState.NOT_SELECTED
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
        self.destination: Path = Path.home()
        self.horizontalLayout_1 = self.setup_hbox('horizontalLayout_1')
        self.horizontalLayout_2 = self.setup_hbox('horizontalLayout_2')
        self.horizontalLayout_3 = self.setup_hbox('horizontalLayout_3')
        self.frame = QtWidgets.QFrame(parent=self)
        self.frame.setStyleSheet('background: #1f2c33')
        self.frame.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.frame.setObjectName('frame')
        self.verticalLayout_1 = QtWidgets.QVBoxLayout(self.frame)
        self.verticalLayout_1.setObjectName('verticalLayout_1')
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
        self.selection_state = self.NOT_SELECTED

    @staticmethod
    def setup_image_widget(parent: QtWidgets.QWidget) -> ImageWidget:
        image_widget = ImageWidget(parent=parent)
        image_widget.setStyleSheet('')
        image_widget.setObjectName('imageWidget')
        return image_widget
    
    @staticmethod
    def setup_hbox(name: str) -> QtWidgets.QHBoxLayout:
        horizontal_layout = QtWidgets.QHBoxLayout()
        horizontal_layout.setObjectName(name)
        return horizontal_layout
    
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
        """Only sublasses of the CustomCropWidget class should implement this method"""
        line_edit = PathLineEdit(path_type=path_type, parent=self)
        line_edit.setMinimumSize(QtCore.QSize(0, 24))
        line_edit.setMaximumSize(QtCore.QSize(16777215, 24))
        line_edit.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhUrlCharactersOnly)
        line_edit.setObjectName(name)
        return line_edit

    def setup_checkbox(self, name: str) -> QtWidgets.QCheckBox:
        """Only sublasses of the CustomCropWidget class should implement this method"""
        checkbox = QtWidgets.QCheckBox(self.frame)
        checkbox.setObjectName(name)
        checkbox.setStyleSheet(self.CHECKBOX_STYLESHEET)
        return checkbox

    def reload_widgets(self) -> None:
        """Only sublasses of the CustomCropWidget class should implement this method"""
        pass

    def disable_buttons(self) -> None:
        """Only sublasses of the CustomCropWidget class should implement this method"""
        pass

    def connect_checkboxs(self, input_widget: QtWidgets.QCheckBox) -> None:
        """Only sublasses of the CustomCropWidget class should implement this method"""
        input_widget.stateChanged.connect(lambda: self.reload_widgets())
        match input_widget:
            case self.mfaceCheckBox:
                input_widget.clicked.connect(
                    lambda: window_functions.uncheck_boxes(self.exposureCheckBox, self.tiltCheckBox))
            case self.exposureCheckBox | self.tiltCheckBox:
                input_widget.clicked.connect(lambda: window_functions.uncheck_boxes(self.mfaceCheckBox))
            case _: pass

    def connect_input_widgets(self, *input_widgets: QtWidgets.QWidget) -> None:
        """Only sublasses of the CustomCropWidget class should implement this method"""
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

    @staticmethod
    def _get_file_names_without_extension(directory: Path) -> Set[str]:
        """Return a set of filenames (without extensions) in the given directory."""
        return {p.stem for p in directory.iterdir() if p.is_file()}

    @staticmethod
    def _check_matching_files(destination: Path, filenames: Set[str], extensions: Set[str]) -> bool:
        """Recursively check if destination contains any files with the given extensions that match the filenames."""
        return any(p.is_file()
                   and p.suffix.lower() in extensions
                   and p.stem in filenames
                   for p in destination.rglob('*'))

    @staticmethod
    def _create_unique_folder(base_path: Path, valid_extensions: Set[str]) -> Path:
        if not base_path.exists():
            return base_path
        if all(file.suffix not in valid_extensions
               for file in base_path.iterdir()
               if file.is_file()):
            return base_path  # Return the original path if no files with the specified extensions are found
        counter = 1
        new_path = base_path
        while new_path.exists():
            new_path = base_path.with_name(f"{base_path.name}_{counter}")
            counter += 1
        new_path.mkdir(parents=True, exist_ok=True)
        return new_path

    def _handle_folder_path(self, destination: Path, folder_path: Path, extensions: Set[str]) -> Path:
        if destination.exists():
            folder_filenames = self._get_file_names_without_extension(folder_path)
            if self._check_matching_files(destination, folder_filenames, extensions):
                destination = destination.with_name(destination.name + '_CROPS')
            destination = self._create_unique_folder(destination, extensions)
        return destination

    def _handle_video_path(self, destination: Path, video_path: Path, extensions: Set[str]) -> Path:
        if destination.exists():
            folder_filenames = self._get_file_names_without_extension(video_path.parent)
            if any(video_path.name.lower() in name.lower() for name in folder_filenames):
                destination = destination.with_name(destination.name + '_CROPS')
            destination = self._create_unique_folder(destination, extensions)
        return destination

    def create_job(self, exposure: QtWidgets.QCheckBox,
                   multi: QtWidgets.QCheckBox,
                   tilt: QtWidgets.QCheckBox,
                   photo_path: Optional[Path] = None,
                   destination: Optional[Path] = None,
                   folder_path: Optional[Path] = None,
                   table: Optional[pd.DataFrame] = None,
                   column1: Optional[QtWidgets.QComboBox] = None,
                   column2: Optional[QtWidgets.QComboBox] = None,
                   video_path: Optional[Path] = None,
                   start_position: Optional[float] = None,
                   stop_position: Optional[float] = None) -> Job:
        """Only sublasses of the CustomCropWidget class should implement this method"""
        extensions = Photo().SAVE_TYPES
        
        if destination and folder_path:
            destination = self._handle_folder_path(destination, folder_path, extensions)
        
        if destination and video_path:
            destination = self._handle_video_path(destination, video_path, extensions)

        self.destination = destination if destination is not None else self.destination

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
        """Only sublasses of the CustomCropWidget class should implement this method"""
        f_name = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Directory', Photo().default_directory)
        line_edit.setText(f_name)
        if line_edit is self.folderLineEdit:
            self.load_data()

    def load_data(self) -> None:
        """Only sublasses of the CustomCropWidget class should implement this method"""
        pass
