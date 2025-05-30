from pathlib import Path
from typing import ClassVar

import polars as pl
from PyQt6.QtCore import QCoreApplication, QSize, Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QWidget,
)

from core import Job
from core.enums import FunctionType
from file_types import FileCategory, file_manager
from line_edits import PathLineEdit, PathType
from ui import utils as ut

from .control_widget import UiCropControlWidget
from .enums import FunctionTabSelectionState, GuiIcon
from .image_widget import ImageWidget
from .tab_state import TabStateManager


class UiCropWidget(QWidget):
    """
    Enhanced base widget class for cropping functionality.
    Provides common UI setup, event handling, and state management.
    """
    SELECTED: ClassVar[FunctionTabSelectionState] = FunctionTabSelectionState.SELECTED
    NOT_SELECTED: ClassVar[FunctionTabSelectionState] = FunctionTabSelectionState.NOT_SELECTED

    # Common size policies
    size_policy_fixed: ClassVar[QSizePolicy] = QSizePolicy(
        QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    size_policy_expand_fixed: ClassVar[QSizePolicy] = QSizePolicy(
        QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    size_policy_expand_expand: ClassVar[QSizePolicy] = QSizePolicy(
        QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    # Common stylesheet
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

    def __init__(self, parent: QWidget, name: str) -> None:
        """
        Initialize the base crop_from_path widget with common components.

        Args:
            parent: The parent widget
            name: Object name for the widget
        """
        super().__init__(parent)
        self.setObjectName(name)
        self.tab_state_manager = TabStateManager(self)

        # Common state and attributes
        self.destination: Path = Path.home()
        self.selection_state = self.NOT_SELECTED
        self.folder_icon = ut.create_button_icon(GuiIcon.FOLDER)

        # Main layouts
        self.verticalLayout_100 = ut.setup_vbox(f"{name}_verticalLayout_100", self)
        self.horizontalLayout_1 = ut.setup_hbox(f"{name}_horizontalLayout_1")
        self.horizontalLayout_2 = ut.setup_hbox(f"{name}_horizontalLayout_2")
        self.horizontalLayout_3 = ut.setup_hbox(f"{name}_horizontalLayout_3")

        # Main image widget and control widget
        self.imageWidget = self.create_image_widget()
        self.controlWidget = self.create_control_widget()

        # Common checkboxes
        self.toggleCheckBox = self.create_checkbox("toggleCheckBox")
        self.toggleCheckBox.setChecked(True)
        self.mfaceCheckBox = self.create_checkbox("mfaceCheckBox")
        self.tiltCheckBox = self.create_checkbox("tiltCheckBox")
        self.exposureCheckBox = self.create_checkbox("exposureCheckBox")

        self._setup_checkbox_relationships(self.mfaceCheckBox, self.tiltCheckBox, self.exposureCheckBox)

        # Common signal connections
        self.toggleCheckBox.toggled.connect(self.controlWidget.setVisible)

    def _setup_checkbox_relationships(
            self,
            ckbx0: QCheckBox,
            ckbx1: QCheckBox,
            ckbx2: QCheckBox
    ) -> None:
        self.tab_state_manager.register_checkbox_exclusivity(ckbx0, {ckbx2, ckbx1})
        self.tab_state_manager.register_checkbox_exclusivity(ckbx2, {ckbx0})  # Exposure is exclusive with multi-face
        self.tab_state_manager.register_checkbox_exclusivity(ckbx1, {ckbx0})  # Tilt is exclusive with multi-face

    def create_image_widget(self) -> ImageWidget:
        """Create the main image display widget with consistent styling"""
        image_widget = ImageWidget()
        image_widget.setObjectName("imageWidget")
        ut.apply_size_policy(
            image_widget,
            self.size_policy_expand_expand,
            min_size=QSize(0, 0),
            max_size=QSize(16_777_215, 16_777_215)
        )
        image_widget.setStyleSheet("")
        return image_widget

    def create_control_widget(self) -> UiCropControlWidget:
        """Create the crop_from_path control widget with dials and settings"""
        horizontal_layout = ut.setup_hbox("horizontalLayout", self.imageWidget)
        control_widget = UiCropControlWidget(self.imageWidget)
        control_widget.setObjectName("controlWidget")
        horizontal_layout.addWidget(control_widget)
        return control_widget

    def create_checkbox(self, name: str) -> QCheckBox:
        """Create a styled checkbox with consistent appearance"""
        check_box = QCheckBox()
        check_box.setObjectName(name)
        ut.apply_size_policy(check_box, self.size_policy_expand_fixed)
        check_box.setStyleSheet(self.CHECKBOX_STYLESHEET)
        return check_box

    def create_str_line_edit(self, name: str, path_type: PathType) -> PathLineEdit:
        """Create a path line edit with consistent styling"""
        line_edit = PathLineEdit(path_type=path_type)
        line_edit.setObjectName(name)
        ut.apply_size_policy(line_edit, self.size_policy_expand_fixed)
        line_edit.setInputMethodHints(Qt.InputMethodHint.ImhUrlCharactersOnly)
        return line_edit

    def create_nav_button(self, name: str) -> QPushButton:
        """Create a navigation button with consistent styling"""
        button = QPushButton()
        button.setObjectName(name)
        ut.apply_size_policy(button, self.size_policy_expand_fixed, min_size=QSize(186, 30))
        return button

    def create_main_frame(self, name: str) -> QFrame:
        """Create a main content frame with consistent styling"""
        return ut.create_frame(name, self, self.size_policy_expand_expand)

    def create_main_button(self, name: str, icon: GuiIcon) -> QPushButton:
        """Create a main action button with consistent styling"""
        return ut.create_main_button(name, self.size_policy_expand_fixed, icon, self)

    def connect_signals(self) -> None:
        raise NotImplementedError("Subclasses must implement connect_signals method")

    def open_path(self, line_edit: PathLineEdit) -> None:
        """Open a file/folder dialog for selecting paths"""
        f_name = QFileDialog.getExistingDirectory(
            self,
            'Select Directory',
            file_manager.get_default_directory(FileCategory.PHOTO).as_posix()
        )
        # Validate the file exists and is accessible
        if f_name:= ut.sanitize_path(f_name):
            line_edit.setText(f_name)

    def disable_buttons(self) -> None:
        """
        Update button states based on input validation.
        This method should be overridden by subclasses to register
        their specific button dependencies.
        """
        self.tab_state_manager.update_button_states()

    def create_job(self, function_type: FunctionType | None = None,
                  photo_path: Path | None = None,
                  destination: Path | None = None,
                  folder_path: Path | None = None,
                  table: pl.DataFrame | None = None,
                  column1: QComboBox | None = None,
                  column2: QComboBox | None = None,
                  video_path: Path | None = None,
                  start_position: float | None = None,
                  stop_position: float | None = None) -> Job:
        """
        Create a job with standardized handling of overlapping paths and consistent defaults.

        Args:
            function_type: The type of function being performed
            photo_path: Optional path to a photo
            destination: Optional destination path
            folder_path: Optional folder path
            table: Optional dataframe for mapping operations
            column1: Optional column selector
            column2: Optional column selector
            video_path: Optional video path
            start_position: Optional video start position
            stop_position: Optional video stop position

        Returns:
            A configured Job object
        """
        # Handle special path cases based on a function type
        if function_type is not None:
            match function_type:
                case FunctionType.FOLDER | FunctionType.MAPPING:
                    if destination and folder_path:
                        self.destination = self._handle_folder_path(
                            destination, folder_path, file_manager.get_save_formats(FileCategory.PHOTO)
                        )
                case FunctionType.VIDEO:
                    if destination and video_path:
                        self.destination = self._handle_video_path(
                            destination, video_path, file_manager.get_save_formats(FileCategory.PHOTO)
                        )
                case FunctionType.FRAME | FunctionType.PHOTO:
                    self.destination = destination if destination is not None else self.destination

        # Create the job with all parameters
        return Job(
            self.controlWidget.widthLineEdit.value(),
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
            stop_position=stop_position
        )

    @staticmethod
    def _get_file_names_without_extension(directory: Path) -> set[str]:
        """Return a set of filenames (without extensions) in the given directory."""
        return {p.stem for p in directory.iterdir() if p.is_file()}

    @staticmethod
    def _check_matching_files(
        destination: Path,
        filenames: set[str],
        extensions: tuple[str, ...]
    ) -> bool:
        """Recursively check if the destination has any files with the given extensions that match the filenames."""
        return any(p.is_file()
                   and p.suffix.lower() in extensions
                   and p.stem in filenames
                   for p in destination.rglob('*'))

    @staticmethod
    def _create_unique_folder(base_path: Path, valid_extensions: tuple[str, ...]) -> Path:
        """Create a unique folder path that doesn't conflict with existing content"""
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

    def _handle_folder_path(
            self,
            destination: Path,
            folder_path: Path,
            extensions: tuple[str, ...]
    ) -> Path:
        """Handle folder path destination to avoid conflicts"""
        if destination.exists() and destination.name:
            folder_filenames = self._get_file_names_without_extension(folder_path)
            if self._check_matching_files(destination, folder_filenames, extensions):
                destination = destination.with_name(f'{destination.name}_CROPS')
            destination = self._create_unique_folder(destination, extensions)
        return destination

    def _handle_video_path(self, destination: Path, video_path: Path, extensions: tuple[str, ...]) -> Path:
        """Handle video path destination to avoid conflicts"""
        if destination.exists():
            folder_filenames = self._get_file_names_without_extension(video_path.parent)
            if any(video_path.name.lower() in name.lower() for name in folder_filenames):
                destination = destination.with_name(f'{destination.name}_CROPS')
            destination = self._create_unique_folder(destination, extensions)
        return destination

    def setup_checkboxes_frame(self, layout: QHBoxLayout) -> None:
        """Set up a standard checkbox layout frame"""
        layout.addWidget(self.toggleCheckBox)

        # Add spacer
        h_spacer = QSpacerItem(
            40, 20,
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum
        )
        layout.addItem(h_spacer)

        # Add checkboxes
        layout.addWidget(self.mfaceCheckBox)
        layout.addWidget(self.tiltCheckBox)
        layout.addWidget(self.exposureCheckBox)

        # Set stretch factor for spacer
        layout.setStretch(1, 20)

    def retranslateUi(self) -> None:
        """Update UI text elements (to be overridden by subclasses)"""
        self.setWindowTitle(QCoreApplication.translate("self", "Form", None))
        self.toggleCheckBox.setText(QCoreApplication.translate("self", "Toggle Settings", None))
        self.mfaceCheckBox.setText(QCoreApplication.translate("self", "Multi-Face", None))
        self.tiltCheckBox.setText(QCoreApplication.translate("self", "Autotilt", None))
        self.exposureCheckBox.setText(QCoreApplication.translate("self", "Autocorrect", None))
