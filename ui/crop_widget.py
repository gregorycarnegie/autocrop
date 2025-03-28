import collections.abc as c
from pathlib import Path
from typing import ClassVar, Optional, Callable, Dict

import polars as pl
from PyQt6 import QtCore, QtWidgets

from core import Job
from core.enums import FunctionType
from file_types import registry, FileTypeInfo
from line_edits import NumberLineEdit, PathLineEdit, PathType
from ui import utils as ut
from .control_widget import UiCropControlWidget
from .enums import GuiIcon, FunctionTabSelectionState
from .image_widget import ImageWidget
from .tab_state import TabStateManager


class UiCropWidget(QtWidgets.QWidget):
    """
    Enhanced base widget class for cropping functionality.
    Provides common UI setup, event handling, and state management.
    """
    SELECTED: ClassVar[FunctionTabSelectionState] = FunctionTabSelectionState.SELECTED
    NOT_SELECTED: ClassVar[FunctionTabSelectionState] = FunctionTabSelectionState.NOT_SELECTED
    
    # Common size policies
    size_policy_fixed: ClassVar[QtWidgets.QSizePolicy] = QtWidgets.QSizePolicy(
        QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
    size_policy_expand_fixed: ClassVar[QtWidgets.QSizePolicy] = QtWidgets.QSizePolicy(
        QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
    size_policy_expand_expand: ClassVar[QtWidgets.QSizePolicy] = QtWidgets.QSizePolicy(
        QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
    
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

    def __init__(self, parent: QtWidgets.QWidget, name: str) -> None:
        """
        Initialize the base crop widget with common components.
        
        Args:
            parent: The parent widget
            name: Object name for the widget
        """
        super().__init__(parent)
        self.setObjectName(name)
        self.tab_state_manager = TabStateManager()
        
        # Common state and attributes
        self.destination: Path = Path.home()
        self.selection_state = self.NOT_SELECTED
        self.folder_icon = ut.create_button_icon(GuiIcon.FOLDER)
        
        # Main layouts
        self.verticalLayout_100 = ut.setup_vbox(f"{name}_verticalLayout_100", self)
        self.horizontalLayout_1 = ut.setup_hbox(f"{name}_horizontalLayout_1")
        self.horizontalLayout_2 = ut.setup_hbox(f"{name}_horizontalLayout_2")
        self.horizontalLayout_3 = ut.setup_hbox(f"{name}_horizontalLayout_3")
        
        # Common input widgets
        self.inputButton = self.create_nav_button("inputButton")
        self.inputLineEdit = self.create_str_line_edit("inputLineEdit", PathType.FOLDER)
        self.destinationButton = self.create_nav_button("destinationButton")
        self.destinationLineEdit = self.create_str_line_edit("destinationLineEdit", PathType.FOLDER)
        
        # Main image widget and control widget
        self.imageWidget = self.create_image_widget()
        self.controlWidget = self.create_control_widget()
        
        # Common checkboxes
        self.toggleCheckBox = self.create_checkbox("toggleCheckBox")
        self.toggleCheckBox.setChecked(True)
        self.mfaceCheckBox = self.create_checkbox("mfaceCheckBox")
        self.tiltCheckBox = self.create_checkbox("tiltCheckBox")
        self.exposureCheckBox = self.create_checkbox("exposureCheckBox")
        
        # Common signal connections
        self.toggleCheckBox.toggled.connect(self.controlWidget.setVisible)
        
        # Initialize state callback maps
        self._input_validators: Dict[QtWidgets.QWidget, Callable[[], None]] = {}
        self._checkbox_connections: Dict[QtWidgets.QCheckBox, Callable[[], None]] = {}
        
        # Initialize the checkbox connections
        self._setup_checkbox_connections()

    def _setup_checkbox_connections(self) -> None:
        """Set up the standard checkbox connections for multi-face and exposure/tilt"""
        self._checkbox_connections.update({
            self.mfaceCheckBox: lambda: ut.uncheck_boxes(self.exposureCheckBox, self.tiltCheckBox),
            self.exposureCheckBox: lambda: ut.uncheck_boxes(self.mfaceCheckBox),
            self.tiltCheckBox: lambda: ut.uncheck_boxes(self.mfaceCheckBox)
        })

    def create_image_widget(self) -> ImageWidget:
        """Create the main image display widget with consistent styling"""
        image_widget = ImageWidget()
        image_widget.setObjectName("imageWidget")
        ut.apply_size_policy(
            image_widget, 
            self.size_policy_expand_expand,
            min_size=QtCore.QSize(0, 0),
            max_size=QtCore.QSize(16_777_215, 16_777_215)
        )
        image_widget.setStyleSheet("")
        return image_widget

    def create_control_widget(self) -> UiCropControlWidget:
        """Create the crop control widget with dials and settings"""
        horizontal_layout = ut.setup_hbox("horizontalLayout", self.imageWidget)
        control_widget = UiCropControlWidget(self.imageWidget)
        control_widget.setObjectName("controlWidget")
        horizontal_layout.addWidget(control_widget)
        return control_widget

    def create_checkbox(self, name: str) -> QtWidgets.QCheckBox:
        """Create a styled checkbox with consistent appearance"""
        check_box = QtWidgets.QCheckBox()
        check_box.setObjectName(name)
        ut.apply_size_policy(check_box, self.size_policy_expand_fixed)
        check_box.setStyleSheet(self.CHECKBOX_STYLESHEET)
        return check_box

    def create_str_line_edit(self, name: str, path_type: PathType) -> PathLineEdit:
        """Create a path line edit with consistent styling"""
        line_edit = PathLineEdit(path_type=path_type)
        line_edit.setObjectName(name)
        ut.apply_size_policy(line_edit, self.size_policy_expand_fixed)
        line_edit.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhUrlCharactersOnly)
        return line_edit

    def create_nav_button(self, name: str) -> QtWidgets.QPushButton:
        """Create a navigation button with consistent styling"""
        button = QtWidgets.QPushButton()
        button.setObjectName(name)
        ut.apply_size_policy(button, self.size_policy_expand_fixed, min_size=QtCore.QSize(186, 30))
        return button

    def create_main_frame(self, name: str) -> QtWidgets.QFrame:
        """Create a main content frame with consistent styling"""
        return ut.create_frame(name, self, self.size_policy_expand_expand)

    def create_main_button(self, name: str, icon: GuiIcon) -> QtWidgets.QPushButton:
        """Create a main action button with consistent styling"""
        return ut.create_main_button(name, self.size_policy_expand_fixed, icon, self)

    def open_path(self, line_edit: PathLineEdit) -> None:
        """Open a file/folder dialog for selecting paths"""
        f_name = QtWidgets.QFileDialog.getExistingDirectory(
            self, 
            'Select Directory', 
            registry.get_default_dir("photo").as_posix()
        )
        line_edit.setText(f_name)

    def connect_checkbox(self, checkbox: QtWidgets.QCheckBox) -> None:
        """Connect checkbox to appropriate handler based on type"""
        if checkbox in self._checkbox_connections:
            checkbox.clicked.connect(self._checkbox_connections[checkbox])

    def connect_input_widgets(self, *input_widgets: QtWidgets.QWidget) -> None:
        """Connect input widgets to state change handlers"""
        for input_widget in input_widgets:
            if isinstance(input_widget, (NumberLineEdit, PathLineEdit)):
                input_widget.textChanged.connect(self.disable_buttons)
            elif isinstance(input_widget, QtWidgets.QCheckBox):
                self.connect_checkbox(input_widget)

    def disable_buttons(self) -> None:
        """
        Disable buttons based on validation state.
        This is a stub that subclasses must override.
        """
        pass

    def create_job(self, function_type: Optional[FunctionType] = None,
                  photo_path: Optional[Path] = None,
                  destination: Optional[Path] = None,
                  folder_path: Optional[Path] = None,
                  table: Optional[pl.DataFrame] = None,
                  column1: Optional[QtWidgets.QComboBox] = None,
                  column2: Optional[QtWidgets.QComboBox] = None,
                  video_path: Optional[Path] = None,
                  start_position: Optional[float] = None,
                  stop_position: Optional[float] = None) -> Job:
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
        # Handle special path cases based on function type
        if function_type is not None:
            if function_type in (FunctionType.FOLDER, FunctionType.MAPPING):
                if destination and folder_path:
                    destination = self._handle_folder_path(destination, folder_path, FileTypeInfo.save_types)
            elif function_type == FunctionType.VIDEO:
                if destination and video_path:
                    destination = self._handle_video_path(destination, video_path, FileTypeInfo.save_types)

        # Update the destination
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
    def _get_file_names_without_extension(directory: Path) -> c.Set[str]:
        """Return a set of filenames (without extensions) in the given directory."""
        return {p.stem for p in directory.iterdir() if p.is_file()}

    @staticmethod
    def _check_matching_files(destination: Path, filenames: c.Set[str], extensions: tuple[str]) -> bool:
        """Recursively check if destination contains any files with the given extensions that match the filenames."""
        return any(p.is_file()
                   and p.suffix.lower() in extensions
                   and p.stem in filenames
                   for p in destination.rglob('*'))

    @staticmethod
    def _create_unique_folder(base_path: Path, valid_extensions: tuple[str]) -> Path:
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

    def _handle_folder_path(self, destination: Path, folder_path: Path, extensions: tuple[str]) -> Path:
        """Handle folder path destination to avoid conflicts"""
        if destination.exists() and destination.name:
            folder_filenames = self._get_file_names_without_extension(folder_path)
            if self._check_matching_files(destination, folder_filenames, extensions):
                destination = destination.with_name(f'{destination.name}_CROPS')
            destination = self._create_unique_folder(destination, extensions)
        return destination

    def _handle_video_path(self, destination: Path, video_path: Path, extensions: tuple[str]) -> Path:
        """Handle video path destination to avoid conflicts"""
        if destination.exists():
            folder_filenames = self._get_file_names_without_extension(video_path.parent)
            if any(video_path.name.lower() in name.lower() for name in folder_filenames):
                destination = destination.with_name(f'{destination.name}_CROPS')
            destination = self._create_unique_folder(destination, extensions)
        return destination
        
    def setup_checkboxes_frame(self, layout: QtWidgets.QHBoxLayout) -> None:
        """Set up a standard checkbox layout frame"""
        layout.addWidget(self.toggleCheckBox)
        
        # Add spacer
        h_spacer = QtWidgets.QSpacerItem(
            40, 20, 
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Minimum
        )
        layout.addItem(h_spacer)
        
        # Add checkboxes
        layout.addWidget(self.mfaceCheckBox)
        layout.addWidget(self.tiltCheckBox)
        layout.addWidget(self.exposureCheckBox)
        
        # Set stretch factor for spacer
        layout.setStretch(1, 20)

    def setup_destination_layout(self, layout: QtWidgets.QHBoxLayout) -> None:
        """Set up the destination path selection layout"""
        layout.addWidget(self.destinationLineEdit)
        layout.addWidget(self.destinationButton)
        layout.setStretch(0, 1)
        
        # Set the destination button icon
        self.destinationButton.setIcon(self.folder_icon)
        
        # Connect click handler
        self.destinationButton.clicked.connect(
            lambda: self.open_path(self.destinationLineEdit)
        )

    def setup_input_layout(self, layout: QtWidgets.QHBoxLayout, icon: GuiIcon = None) -> None:
        """Set up the input file/folder selection layout"""
        layout.addWidget(self.inputLineEdit)
        layout.addWidget(self.inputButton)
        layout.setStretch(0, 1)
        
        # Set button icon
        if icon:
            self.inputButton.setIcon(ut.create_button_icon(icon))
        else:
            self.inputButton.setIcon(self.folder_icon)
            
        # Connect click handler
        self.inputButton.clicked.connect(
            lambda: self.open_path(self.inputLineEdit)
        )

    def retranslateUi(self) -> None:
        """Update UI text elements (to be overridden by subclasses)"""
        self.setWindowTitle(QtCore.QCoreApplication.translate("self", "Form", None))
        self.toggleCheckBox.setText(QtCore.QCoreApplication.translate("self", "Toggle Settings", None))
        self.mfaceCheckBox.setText(QtCore.QCoreApplication.translate("self", "Multi-Face", None))
        self.tiltCheckBox.setText(QtCore.QCoreApplication.translate("self", "Autotilt", None))
        self.exposureCheckBox.setText(QtCore.QCoreApplication.translate("self", "Autocorrect", None))
        self.inputButton.setText(QtCore.QCoreApplication.translate("self", "Select Input", None))
        self.destinationButton.setText(QtCore.QCoreApplication.translate("self", "Destination Folder", None))
