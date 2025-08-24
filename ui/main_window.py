from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path

from PyQt6.QtCore import QCoreApplication, QMetaObject, QSize, QUrl
from PyQt6.QtGui import (
    QCloseEvent,
    QDragEnterEvent,
    QDragMoveEvent,
    QDropEvent,
    QIcon,
    QImage,
)
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QMainWindow,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from core import face_tools as ft
from core import processing as prc
from core.croppers import (
    DisplayCropper,
    FolderCropper,
    MappingCropper,
    PhotoCropper,
    VideoCropper,
)
from core.enums import FunctionType, Preset
from file_types import FileCategory, SignatureChecker, file_manager
from line_edits import LineEditState, NumberLineEdit, PathLineEdit, PathType
from ui import utils as ut
from ui.components import AddressBarWidget, DragDropHandler, MenuManager, TabManager, TabWidget
from ui.control_widget import UiCropControlWidget
from ui.crop_widget import UiCropWidget
from ui.enums import GuiIcon
from ui.folder_tab import UiFolderTabWidget
from ui.mapping_tab import UiMappingTabWidget
from ui.photo_tab import UiPhotoTabWidget
from ui.splash_screen import UiClickableSplashScreen
from ui.video_tab import UiVideoTabWidget


class UiMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)

        # Start with a splash screen
        splash = UiClickableSplashScreen()
        splash.show_message("Loading face detection models...")
        QApplication.processEvents()

        face_detection_tools = self.get_face_detection_tools(splash)

        # Initialize workers
        self._initialize_workers(face_detection_tools)

        # Create the central widget
        self.centralwidget = QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.main_layout = QVBoxLayout(self.centralwidget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        self.setObjectName("MainWindow")
        self.resize(1256, 652)
        icon = QIcon()
        icon.addFile(GuiIcon.LOGO, QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.setWindowIcon(icon)

        # Create a status bar first
        self.statusbar = QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)

        # Initialize modular components
        self._initialize_components()

        # Setup UI components
        self._setup_ui()

        # Connect signals
        self.connect_widgets()

        self.setCentralWidget(self.centralwidget)
        self.retranslateUi()

        # Set the initial tab
        self.tab_manager.function_tabWidget.setCurrentIndex(0)

        self.address_bar.initialize_clear_button_states()

        QMetaObject.connectSlotsByName(self)

    def _initialize_workers(self, face_detection_tools):
        """Initialize all worker instances"""
        # Single-threaded workers
        self.display_worker = DisplayCropper(face_detection_tools[0])
        self.photo_worker = PhotoCropper(face_detection_tools[0])
        self.video_worker = VideoCropper(face_detection_tools[0])

        # Multithreaded workers
        self.folder_worker = FolderCropper(face_detection_tools)
        self.mapping_worker = MappingCropper(face_detection_tools)

        # Store workers for easy access
        self.workers = {
            'display': self.display_worker,
            'photo': self.photo_worker,
            'video': self.video_worker,
            'folder': self.folder_worker,
            'mapping': self.mapping_worker
        }

    def _initialize_components(self):
        """Initialize modular components"""
        # Menu manager
        self.menu_manager = MenuManager(self)

        # Tab manager
        self.tab_manager = TabManager(self.centralwidget, self.workers)

        # Address bar widget - UNCHANGED logic preserved
        self.address_bar = AddressBarWidget(self)

        # Drag drop handler
        self.drag_drop_handler = DragDropHandler(self.tab_manager, self.statusbar)

    def _setup_ui(self):
        """Setup the main UI components"""
        # Create the main menu
        self.menu_manager.create_main_menu()

        # Add address bar to the main layout - UNCHANGED
        self.main_layout.addWidget(self.address_bar)

        # Add tab widget to the main layout
        self.main_layout.addWidget(self.tab_manager.function_tabWidget)

        # Setup video progress bars
        self.video_worker.progressBars = [
            self.tab_manager.video_tab_widget.progressBar,
            self.tab_manager.video_tab_widget.progressBar_2
        ]

        # Register state retrieval functions for each widget type
        for func_type, widget in [
            (FunctionType.PHOTO, self.tab_manager.photo_tab_widget),
            (FunctionType.FOLDER, self.tab_manager.folder_tab_widget),
            (FunctionType.MAPPING, self.tab_manager.mapping_tab_widget),
            (FunctionType.VIDEO, self.tab_manager.video_tab_widget)
        ]:
            self.display_worker.register_widget_state(
                func_type, lambda w=widget: self.get_widget_state(w), lambda w=widget: self.get_path(w)
            )

            # Connect signals
            self.connect_widget_signals(widget, partial(self.display_worker.crop, func_type))

            # Connect the image updated event to the widget's setImage method
            self.display_worker.events.image_updated.connect(
                lambda _ft, img, w=widget, ft_expected=func_type:
                    self._handle_image_update(_ft, img, w, ft_expected)
            )

    def _handle_image_update(self, function_type: FunctionType, image: QImage | None,
                             widget, expected_type: FunctionType) -> None:
        """Handle image updates from the display worker, including no-face scenarios"""
        if function_type != expected_type:
            return

        if image is not None:
            # Normal case - display the image
            widget.imageWidget.setImage(image)
        else:
            # No face detected or error case
            message = self.display_worker.get_no_face_message(function_type)
            widget.imageWidget.showNoFaceDetected(message)

    def update_address_bar_context(self):
        """Update address bar context based on current tab - UNCHANGED logic preserved"""
        current_index = self.tab_manager.function_tabWidget.currentIndex()

        # Update context using the address bar widget
        self.address_bar.update_context(current_index)

        # Get paths from current tab widgets - UNCHANGED logic
        match current_index:
            case FunctionType.PHOTO:
                primary_path = self.tab_manager.photo_tab_widget.input_path if self.tab_manager.photo_tab_widget.input_path else ""
                destination_path = self.tab_manager.photo_tab_widget.destination_path if self.tab_manager.photo_tab_widget.destination_path else ""
                secondary_path = ""

            case FunctionType.FOLDER:
                primary_path = self.tab_manager.folder_tab_widget.input_path if self.tab_manager.folder_tab_widget.input_path else ""
                destination_path = self.tab_manager.folder_tab_widget.destination_path if self.tab_manager.folder_tab_widget.destination_path else ""
                secondary_path = ""

            case FunctionType.MAPPING:
                primary_path = self.tab_manager.mapping_tab_widget.input_path if self.tab_manager.mapping_tab_widget.input_path else ""
                destination_path = self.tab_manager.mapping_tab_widget.destination_path if self.tab_manager.mapping_tab_widget.destination_path else ""
                secondary_path = self.tab_manager.mapping_tab_widget.table_path if self.tab_manager.mapping_tab_widget.table_path else ""

            case FunctionType.VIDEO:
                primary_path = self.tab_manager.video_tab_widget.input_path if self.tab_manager.video_tab_widget.input_path else ""
                destination_path = self.tab_manager.video_tab_widget.destination_path if self.tab_manager.video_tab_widget.destination_path else ""
                secondary_path = ""

            case _:
                primary_path = destination_path = secondary_path = ""

        # Update paths in address bar - UNCHANGED logic
        self.address_bar.update_paths(primary_path, secondary_path, destination_path)

    # retranslateUi
    def retranslateUi(self):
        self.setWindowTitle(QCoreApplication.translate("self", "Autocrop", None))

        # Delegate to component managers
        self.menu_manager.retranslate_ui()
        self.tab_manager.retranslate_ui()
        self.address_bar.retranslate_ui()

    def connect_widgets(self):
        """Connect widget signals to handlers"""
        # CONNECTIONS
        self.connect_combo_boxes(self.tab_manager.mapping_tab_widget)

        # Menu actions - delegate to menu manager
        self.menu_manager.get_action('about').triggered.connect(ut.load_about_form)
        self.menu_manager.get_action('golden_ratio').triggered.connect(partial(self.load_preset, Preset.GOLDEN_RATIO))
        self.menu_manager.get_action('2_3_ratio').triggered.connect(partial(self.load_preset, Preset.TWO_THIRDS))
        self.menu_manager.get_action('3_4_ratio').triggered.connect(partial(self.load_preset, Preset.THREE_QUARTERS))
        self.menu_manager.get_action('4_5_ratio').triggered.connect(partial(self.load_preset, Preset.FOUR_FIFTHS))
        self.menu_manager.get_action('square').triggered.connect(partial(self.load_preset, Preset.SQUARE))

        # Connect tab selection actions
        self.menu_manager.get_action('crop_file').triggered.connect(partial(self.tab_manager.function_tabWidget.setCurrentIndex, FunctionType.PHOTO))
        self.menu_manager.get_action('crop_folder').triggered.connect(partial(self.tab_manager.function_tabWidget.setCurrentIndex, FunctionType.FOLDER))
        self.menu_manager.get_action('use_mapping').triggered.connect(partial(self.tab_manager.function_tabWidget.setCurrentIndex, FunctionType.MAPPING))
        self.menu_manager.get_action('crop_video').triggered.connect(partial(self.tab_manager.function_tabWidget.setCurrentIndex, FunctionType.VIDEO))

        # Browser-style navigation buttons - UNCHANGED
        self.address_bar.back_button.clicked.connect(self.navigate_back)
        self.address_bar.forward_button.clicked.connect(self.navigate_forward)
        self.address_bar.refresh_button.clicked.connect(self.refresh_current_view)
        self.address_bar.info_button.clicked.connect(self.show_settings)

        # Connect unified address bar signals - UNCHANGED logic preserved
        self.connect_address_bar_signals()

        # Tab change event
        self.tab_manager.function_tabWidget.currentChanged.connect(self.check_tab_selection)
        self.tab_manager.function_tabWidget.currentChanged.connect(self.tab_manager.video_tab_widget.player.pause)

        # Error handling connections
        self.folder_worker.error.connect(self.tab_manager.folder_tab_widget.disable_buttons)
        self.photo_worker.error.connect(self.tab_manager.photo_tab_widget.disable_buttons)
        self.mapping_worker.error.connect(self.tab_manager.mapping_tab_widget.disable_buttons)
        self.video_worker.error.connect(self.tab_manager.video_tab_widget.disable_buttons)

        # Connect drag drop handler
        self.drag_drop_handler.file_dropped.connect(self._handle_file_drop)
        self.drag_drop_handler.directory_dropped.connect(self._handle_directory_drop)

    def connect_address_bar_signals(self):
        """Connect signals for the unified address bar - UNCHANGED logic preserved"""
        # Connect tab change to update address bar context
        self.tab_manager.function_tabWidget.currentChanged.connect(self.update_address_bar_context)

        # Connect the unified address bar to update the current tab's input field
        self.address_bar.unified_address_bar.textChanged.connect(self.unified_address_changed)
        self.address_bar.unified_address_bar.textChanged.connect(self.update_all_button_states)

        # Connect secondary input for mapping tab - UNCHANGED
        self.address_bar.secondary_input.textChanged.connect(self.secondary_input_changed)
        self.address_bar.secondary_input.textChanged.connect(self.update_all_button_states)

        # Connect destination input - UNCHANGED
        self.address_bar.destination_input.textChanged.connect(self.destination_input_changed)
        self.address_bar.destination_input.textChanged.connect(self.update_all_button_states)

        # Connect context-aware buttons
        self.address_bar.context_button_clicked.connect(self.handle_context_button)
        self.address_bar.secondary_button_clicked.connect(self.handle_secondary_button)
        self.address_bar.destination_button_clicked.connect(self.handle_destination_button)

    def update_all_button_states(self):
        """Update button states for all tab widgets when address bars change"""
        # Update button states for all tabs
        for tab_widget in [
            self.tab_manager.photo_tab_widget,
            self.tab_manager.folder_tab_widget,
            self.tab_manager.mapping_tab_widget,
            self.tab_manager.video_tab_widget
        ]:
            tab_widget.tab_state_manager.update_button_states()

    def unified_address_changed(self, text: str):
        """Handle changes to the unified address bar - UNCHANGED logic"""
        current_index = self.tab_manager.function_tabWidget.currentIndex()

        # Update the appropriate input field in the current tab
        if cleaned_text := ut.sanitize_path(text) if text.strip() else "":
            match current_index:
                case FunctionType.PHOTO:
                    self.tab_manager.photo_tab_widget.input_path = cleaned_text
                case FunctionType.FOLDER:
                    self.tab_manager.folder_tab_widget.input_path = cleaned_text
                case FunctionType.MAPPING:
                    self.tab_manager.mapping_tab_widget.input_path = cleaned_text
                case FunctionType.VIDEO:
                    self.tab_manager.video_tab_widget.input_path = cleaned_text
        else:
            # Clear the path if text is empty or invalid
            match current_index:
                case FunctionType.PHOTO:
                    self.tab_manager.photo_tab_widget.input_path = ""
                case FunctionType.FOLDER:
                    self.tab_manager.folder_tab_widget.input_path = ""
                case FunctionType.MAPPING:
                    self.tab_manager.mapping_tab_widget.input_path = ""
                case FunctionType.VIDEO:
                    self.tab_manager.video_tab_widget.input_path = ""

        # Trigger preview update for the current tab (only if valid)
        if text.strip() and self.address_bar.unified_address_bar.state == LineEditState.VALID_INPUT:
            self.trigger_preview_update()

    def trigger_preview_update(self):
        """Trigger preview update for the current tab when path is valid"""
        current_index = self.tab_manager.function_tabWidget.currentIndex()

        # Check if the path is valid before updating
        if self.address_bar.unified_address_bar.state == LineEditState.VALID_INPUT:
            match current_index:
                case FunctionType.PHOTO:
                    if self.tab_manager.photo_tab_widget.input_path:
                        self.display_worker.crop(FunctionType.PHOTO)
                case FunctionType.FOLDER:
                    if self.tab_manager.folder_tab_widget.input_path:
                        self.tab_manager.folder_tab_widget.load_data()
                        self.display_worker.crop(FunctionType.FOLDER)
                case FunctionType.MAPPING:
                    if self.tab_manager.mapping_tab_widget.input_path:
                        # Load folder data for mapping tab too
                        self.tab_manager.mapping_tab_widget.load_data()
                        self.display_worker.crop(FunctionType.MAPPING)
                case FunctionType.VIDEO:
                    if self.tab_manager.video_tab_widget.input_path:
                        self.display_worker.crop(FunctionType.VIDEO)

    def secondary_input_changed(self, text: str):
        """Handle changes to the secondary input - UNCHANGED logic"""
        current_index = self.tab_manager.function_tabWidget.currentIndex()

        if current_index == FunctionType.MAPPING:
            if cleaned_text := ut.sanitize_path(text) if text.strip() else "":
                self.tab_manager.mapping_tab_widget.table_path = cleaned_text
            else:
                self.tab_manager.mapping_tab_widget.table_path = ""

    def destination_input_changed(self, text: str):
        """Handle changes to the destination input - UNCHANGED logic"""
        current_index = self.tab_manager.function_tabWidget.currentIndex()

        if cleaned_text := ut.sanitize_path(text) if text.strip() else "":
            match current_index:
                case FunctionType.PHOTO:
                    self.tab_manager.photo_tab_widget.destination_path = cleaned_text
                case FunctionType.FOLDER:
                    self.tab_manager.folder_tab_widget.destination_path = cleaned_text
                case FunctionType.MAPPING:
                    self.tab_manager.mapping_tab_widget.destination_path = cleaned_text
                case FunctionType.VIDEO:
                    self.tab_manager.video_tab_widget.destination_path = cleaned_text
        else:
            # Clear destination path if text is empty or invalid
            match current_index:
                case FunctionType.PHOTO:
                    self.tab_manager.photo_tab_widget.destination_path = ""
                case FunctionType.FOLDER:
                    self.tab_manager.folder_tab_widget.destination_path = ""
                case FunctionType.MAPPING:
                    self.tab_manager.mapping_tab_widget.destination_path = ""
                case FunctionType.VIDEO:
                    self.tab_manager.video_tab_widget.destination_path = ""

    def handle_context_button(self):
        """Handle clicks on the context-aware button"""
        match self.tab_manager.function_tabWidget.currentIndex():
            case FunctionType.PHOTO:
                self.open_file_dialog(PathType.IMAGE, self.address_bar.unified_address_bar)
            case FunctionType.FOLDER:
                self.open_folder_dialog(self.address_bar.unified_address_bar)
            case FunctionType.MAPPING:
                self.open_folder_dialog(self.address_bar.unified_address_bar)
            case FunctionType.VIDEO:
                self.open_file_dialog(PathType.VIDEO, self.address_bar.unified_address_bar)

    def handle_secondary_button(self):
        """Handle clicks on the secondary button - UNCHANGED logic"""
        match self.tab_manager.function_tabWidget.currentIndex():
            case FunctionType.MAPPING:
                self.open_file_dialog(PathType.TABLE, self.address_bar.secondary_input)
            case _:
                pass

    def handle_destination_button(self):
        """Handle clicks on the destination button - UNCHANGED logic"""
        self.open_folder_dialog(self.address_bar.destination_input)

    def open_file_dialog(self, path_type: PathType, target_input: PathLineEdit) -> None:
        """Securely open a file dialog and validate the selected path"""
        try:
            # Configure file dialog options
            options = QFileDialog.Option.ReadOnly
            category = self._get_file_category_for_path_type(path_type)
            default_dir = file_manager.get_default_directory(category).as_posix()
            filter_string = file_manager.get_filter_string(category)
            title = self._get_dialog_title(path_type)

            # Open the dialog
            f_name, _ = QFileDialog.getOpenFileName(
                self, title, default_dir, filter_string, options=options
            )

            # Validate and process the selected file
            if f_name := ut.sanitize_path(f_name):
                self._process_and_update_path(f_name, category, path_type, target_input)

        except Exception as e:
            ut.show_error_box(f"An error occurred opening the file\n{e}")

    def _get_file_category_for_path_type(self, path_type: PathType) -> FileCategory:
        """Map path types to file categories"""
        category_map = {
            PathType.IMAGE: FileCategory.PHOTO,
            PathType.VIDEO: FileCategory.VIDEO,
            PathType.TABLE: FileCategory.TABLE
        }
        return category_map.get(path_type, FileCategory.PHOTO)

    def _get_dialog_title(self, path_type: PathType) -> str:
        """Get appropriate dialog title based on path type"""
        title_map = {
            PathType.IMAGE: "Open Image",
            PathType.VIDEO: "Open Video",
            PathType.TABLE: "Open Table"
        }
        return title_map.get(path_type, "Open File")

    def _process_and_update_path(self, file_path: str, category: FileCategory,
                                 path_type: PathType, target_input: PathLineEdit) -> None:
        """Process and validate the selected file path"""
        path_obj = Path(file_path).resolve()

        # Validate file exists
        if not path_obj.is_file():
            ut.show_error_box("Selected path is not a valid file")
            return

        # Validate file type
        if not file_manager.is_valid_type(path_obj, category):
            ut.show_error_box(f"Selected file is not a valid {category.name.lower()}")
            return

        # Verify file content matches extension
        if not SignatureChecker.verify_file_type(path_obj, category):
            ut.show_error_box("File content doesn't match its extension. The file may be corrupted or modified.")
            return

        # Update the appropriate input path and UI
        self._update_path_and_ui(file_path, path_type, target_input)

    def _update_path_and_ui(self, file_path: str, path_type: PathType,
                            target_input: PathLineEdit) -> None:
        """Update the path storage and UI elements"""
        current_index = self.tab_manager.function_tabWidget.currentIndex()

        if tab_widget := self.tab_manager.get_current_tab_widget():
            if path_type == PathType.IMAGE:
                tab_widget.input_path = file_path
                if current_index == FunctionType.PHOTO:
                    self.address_bar.unified_address_bar.setText(file_path)
            elif path_type == PathType.VIDEO:
                tab_widget.input_path = file_path
                self.tab_manager.video_tab_widget.player.setSource(QUrl.fromLocalFile(file_path))
                self.tab_manager.video_tab_widget.reset_video_widgets()
            elif path_type == PathType.TABLE:
                tab_widget.table_path = file_path
                data = prc.load_table(Path(file_path))
                self.tab_manager.mapping_tab_widget.process_data(data)
            elif path_type == PathType.FOLDER:
                # New: Handle folder paths for mapping tab
                tab_widget.input_path = file_path
                if current_index == FunctionType.MAPPING:
                    # Load the folder data into the mapping tab's tree view
                    self.tab_manager.mapping_tab_widget.load_data()

    def open_folder_dialog(self, target: PathLineEdit) -> None:
        """Securely open a folder dialogue and validate the selected path"""
        try:
            # Use QFileDialog with options that improve security
            options = QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontResolveSymlinks

            # Get appropriate default directory
            default_dir = file_manager.get_default_directory(FileCategory.PHOTO).as_posix()

            # Open the dialogue with the appropriate title
            f_name = QFileDialog.getExistingDirectory(
                self, 'Select Directory', default_dir, options=options
            )

            # Validate the selected path
            if f_name := ut.sanitize_path(f_name):
                # Create a Path object for additional validation
                path_obj = Path(f_name).resolve()

                # Verify directory exists and is accessible
                if not path_obj.is_dir():
                    ut.show_error_box("Selected path is not a valid directory")
                    return

                # Update the input with a safe path
                target.setText(path_obj.as_posix())

                # If this is a source directory, refresh any view that depends on it
                current_index = self.tab_manager.function_tabWidget.currentIndex()
                if target == self.address_bar.unified_address_bar:
                    if current_index == FunctionType.FOLDER:
                        self.tab_manager.folder_tab_widget.load_data()
                    elif current_index == FunctionType.MAPPING:
                        # Also load data for mapping tab
                        self.tab_manager.mapping_tab_widget.load_data()

        except Exception as e:
            # Log error internally without exposing details
            ut.show_error_box(f"An error occurred opening the directory\n{e}")

    def _handle_file_drop(self, file_path: Path, function_type: FunctionType):
        """Handle file dropped from drag drop handler"""
        self.blockSignals(True)
        self.tab_manager.set_current_tab(function_type)

        if function_type == FunctionType.PHOTO:
            self.tab_manager.photo_tab_widget.input_path = file_path.as_posix()
            self.address_bar.unified_address_bar.setText(self.tab_manager.photo_tab_widget.input_path)
            self.display_worker.crop(FunctionType.PHOTO)
        elif function_type == FunctionType.VIDEO:
            self.tab_manager.video_tab_widget.input_path = file_path.as_posix()
            self.address_bar.unified_address_bar.setText(self.tab_manager.video_tab_widget.input_path)
            self.tab_manager.video_tab_widget.player.setSource(QUrl.fromLocalFile(file_path.as_posix()))
            self.tab_manager.video_tab_widget.reset_video_widgets()
        elif function_type == FunctionType.MAPPING:
            self.tab_manager.mapping_tab_widget.table_path = file_path.as_posix()
            self.address_bar.secondary_input.setText(self.tab_manager.mapping_tab_widget.table_path)
            # Process the table data
            data = prc.load_table(file_path)
            self.tab_manager.mapping_tab_widget.process_data(data)

        self.blockSignals(False)

    def _handle_directory_drop(self, dir_path: Path, function_type: FunctionType):
        """Handle directory dropped from drag drop handler"""
        self.blockSignals(True)
        self.tab_manager.set_current_tab(function_type)

        if function_type == FunctionType.FOLDER:
            self.tab_manager.folder_tab_widget.input_path = dir_path.as_posix()
            self.address_bar.unified_address_bar.setText(self.tab_manager.folder_tab_widget.input_path)
            # Update display
            self.display_worker.current_paths[FunctionType.FOLDER] = None
            self.tab_manager.folder_tab_widget.load_data()
            self.display_worker.crop(FunctionType.FOLDER)
        elif function_type == FunctionType.MAPPING:
            self.tab_manager.mapping_tab_widget.input_path = dir_path.as_posix()
            self.address_bar.unified_address_bar.setText(self.tab_manager.mapping_tab_widget.input_path)
            # Update display
            self.display_worker.current_paths[FunctionType.MAPPING] = None
            self.display_worker.crop(FunctionType.MAPPING)

        self.blockSignals(False)

    # Browser-style navigation methods
    def navigate_back(self):
        """Navigate to the previous tab"""
        current_index = self.tab_manager.function_tabWidget.currentIndex()
        if current_index > 0:
            self.tab_manager.function_tabWidget.setCurrentIndex(current_index - 1)

    def navigate_forward(self):
        """Navigate to the next tab"""
        current_index = self.tab_manager.function_tabWidget.currentIndex()
        if current_index < self.tab_manager.function_tabWidget.count() - 1:
            self.tab_manager.function_tabWidget.setCurrentIndex(current_index + 1)

    def refresh_current_view(self):
        """Refresh the current tab's view"""
        # Handle refresh based on the current tab
        match self.tab_manager.function_tabWidget.currentIndex():
            case FunctionType.PHOTO:
                # Refresh photo preview
                if self.address_bar.unified_address_bar.text() and self.address_bar.unified_address_bar.state == LineEditState.VALID_INPUT:
                    self.display_worker.crop(FunctionType.PHOTO)
            case FunctionType.FOLDER:
                # Refresh folder view
                self.tab_manager.folder_tab_widget.load_data()
            case FunctionType.MAPPING:
                # Refresh mapping preview
                if self.address_bar.secondary_input.text() and self.address_bar.secondary_input.state == LineEditState.VALID_INPUT:
                    if file_path := Path(self.address_bar.secondary_input.text()):
                        data = prc.load_table(file_path)
                        self.tab_manager.mapping_tab_widget.process_data(data)
            case FunctionType.VIDEO:
                # Refresh video preview
                self.display_worker.crop(FunctionType.VIDEO)
            case _:
                pass

        # Update the status bar with a message
        self.statusbar.showMessage("View refreshed", 2000)

    @staticmethod
    def show_settings():
        """Show settings dialogue"""
        # For now, show the About dialogue as a placeholder
        ut.load_about_form()

    def update_address_bar(self):
        """Update address bar with current tab's path"""
        # Get the appropriate path based on the current tab
        match self.tab_manager.function_tabWidget.currentIndex():
            case FunctionType.PHOTO:
                path = self.tab_manager.photo_tab_widget.input_path
            case FunctionType.FOLDER:
                path = self.tab_manager.folder_tab_widget.input_path
            case FunctionType.MAPPING:
                path = self.tab_manager.mapping_tab_widget.table_path
            case FunctionType.VIDEO:
                path = self.tab_manager.video_tab_widget.input_path
            case _:
                return None

        # Update the address bar without triggering the text_changed event
        self.address_bar.unified_address_bar.blockSignals(True)
        self.address_bar.unified_address_bar.setText(path)
        self.address_bar.unified_address_bar.blockSignals(False)
        return None

    def update_current_tab_path(self):
        """Update the current tab's path with address bar text"""
        path = self.address_bar.unified_address_bar.text()

        # Update the appropriate input field based on the current tab
        match self.tab_manager.function_tabWidget.currentIndex():
            case FunctionType.PHOTO:
                self.tab_manager.photo_tab_widget.input_path = path
            case FunctionType.FOLDER:
                self.tab_manager.folder_tab_widget.input_path = path
            case FunctionType.MAPPING:
                self.tab_manager.mapping_tab_widget.table_path = path
            case FunctionType.VIDEO:
                self.tab_manager.video_tab_widget.input_path = path
            case _:
                pass

    def closeEvent(self, event: QCloseEvent) -> None:
        """
        Handle window close event by performing proper clean-up.
        This is called automatically when the window is closed.
        """
        # Clean up all worker thread pools
        self.cleanup_workers()

        # Call the parent class implementation
        super().closeEvent(event)

    def cleanup_workers(self) -> None:
        """
        Clean up all worker threads and resources before application exit.
        """
        # Clean up batch workers
        self.folder_worker.cleanup()
        self.mapping_worker.cleanup()

        # Stop any ongoing video playback
        self.tab_manager.video_tab_widget.player.stop()

    def get_face_detection_tools(self, splash: UiClickableSplashScreen) -> list[ft.FaceToolPair]:
        # Use ThreadPoolExecutor instead of ProcessPoolExecutor
        face_detection_tools: list[ft.FaceToolPair] = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit threads to create tool pairs
            future_tools = [executor.submit(ft.create_tool_pair) for _ in range(ft.THREAD_NUMBER)]

            total = len(future_tools)
            for completed, future in enumerate(as_completed(future_tools), start=1):
                face_detection_tools.append(future.result())
                splash.show_message(f"Loading face detection models... {completed}/{total}")
                QApplication.processEvents()
            splash.finish(self)
            return face_detection_tools

    @staticmethod
    def connect_combo_boxes(tab_widget: TabWidget) -> None:
        """
        Connects the combo boxes in the tab widget to the disable_buttons method.

        Args:
            tab_widget (TabWidget): The tab widget containing the combo boxes.
        """
        try:
            assert isinstance(tab_widget, UiMappingTabWidget)
        except AssertionError:
            return

        tab_widget.comboBox_1.currentTextChanged.connect(tab_widget.disable_buttons)
        tab_widget.comboBox_2.currentTextChanged.connect(tab_widget.disable_buttons)

    @staticmethod
    def get_widget_state(w: TabWidget):
        control = w.controlWidget
        return (
            w.input_path,
            control.widthLineEdit.text(),
            control.heightLineEdit.text(),
            w.exposureCheckBox.isChecked(),
            w.mfaceCheckBox.isChecked(),
            w.tiltCheckBox.isChecked(),
            control.sensitivityDial.value(),
            control.fpctDial.value(),
            control.gammaDial.value(),
            control.topDial.value(),
            control.bottomDial.value(),
            control.leftDial.value(),
            control.rightDial.value(),
            control.radio_tuple,
        )

    @staticmethod
    def get_path(w: TabWidget) -> str:
        return w.input_path

    @staticmethod
    def connect_widget_signals(widget: TabWidget, crop_method: Callable) -> None:
        """Connect signals with minimal overhead"""
        signals = (
            widget.controlWidget.widthLineEdit.textChanged,
            widget.controlWidget.heightLineEdit.textChanged,
            widget.exposureCheckBox.stateChanged,
            widget.mfaceCheckBox.stateChanged,
            widget.tiltCheckBox.stateChanged,
            widget.controlWidget.sensitivityDial.valueChanged,
            widget.controlWidget.fpctDial.valueChanged,
            widget.controlWidget.gammaDial.valueChanged,
            widget.controlWidget.topDial.valueChanged,
            widget.controlWidget.bottomDial.valueChanged,
            widget.controlWidget.leftDial.valueChanged,
            widget.controlWidget.rightDial.valueChanged
        )
        for signal in signals:
            signal.connect(crop_method)

    def adjust_ui(self, app: QApplication):
        if (screen := app.primaryScreen()) is None:
            return

        # Get screen dimensions
        size = screen.size()
        width, height = size.width(), size.height()

        # Calculate the appropriate window size based on screen size
        window_width = max(min(int(width * 0.85), 1600), 800)  # Cap at 1600 px for very large screens
        window_height = max(min(int(height * 0.85), 900), 600)  # Cap at 900 px for very large screens

        # Resize the window
        self.resize(window_width, window_height)

        # Center the window on the screen
        center_point = screen.geometry().center()
        frame_geometry = self.frameGeometry()
        frame_geometry.moveCenter(center_point)
        self.move(frame_geometry.topLeft())

        # Adjust font size based on screen resolution
        base_font_size = 10
        if height > 1080:
            scale_factor = min(height / 1080, 1.5)  # Cap scaling at 1.5x
            base_font_size = int(base_font_size * scale_factor)

        font = app.font()
        font.setPointSize(base_font_size)
        app.setFont(font)

        # Set appropriate tab sizes
        self.tab_manager.function_tabWidget.setTabsClosable(True)
        self.tab_manager.function_tabWidget.tabCloseRequested.connect(self.handle_tab_close)

        # Initialize navigation button states
        self.update_navigation_button_states()

    def handle_tab_close(self, index: int):
        """
        Handle tab close button clicks.
        For a browser-like experience, we don't close tabs but reset their state.
        """
        # Show a message in the status bar
        self.statusbar.showMessage(f"Tab {self.tab_manager.function_tabWidget.tabText(index)} reset", 2000)

        # Delegate to tab manager
        self.tab_manager.handle_tab_close(index)

        # Update the address bar
        self.update_address_bar()

    def update_navigation_button_states(self):
        """Update the state of navigation buttons"""
        current_index = self.tab_manager.function_tabWidget.currentIndex()
        tab_count = self.tab_manager.function_tabWidget.count()

        # Enable/disable back button
        self.address_bar.back_button.setEnabled(current_index > 0)

        # Enable/disable forward button
        self.address_bar.forward_button.setEnabled(current_index < tab_count - 1)

    def check_tab_selection(self) -> None:
        """
        Checks the current selection of the function tab widget and handles the tab states accordingly.
        """
        # Update navigation buttons
        self.update_navigation_button_states()

        # Update unified address bar context
        self.update_address_bar_context()

        # Force validation of path inputs
        self.address_bar.unified_address_bar.validate_path()
        self.address_bar.destination_input.validate_path()
        if self.address_bar.secondary_input_container.isVisible():
            self.address_bar.secondary_input.validate_path()

        # Process tab selection as before
        match self.tab_manager.function_tabWidget.currentIndex():
            case FunctionType.PHOTO:
                self.handle_function_tab_state(
                    self.tab_manager.photo_tab_widget, self.tab_manager.folder_tab_widget, self.tab_manager.mapping_tab_widget, self.tab_manager.video_tab_widget
                )
            case FunctionType.FOLDER:
                self.handle_function_tab_state(
                    self.tab_manager.folder_tab_widget, self.tab_manager.mapping_tab_widget, self.tab_manager.video_tab_widget, self.tab_manager.photo_tab_widget
                )
            case FunctionType.MAPPING:
                self.handle_function_tab_state(
                    self.tab_manager.mapping_tab_widget, self.tab_manager.video_tab_widget, self.tab_manager.photo_tab_widget, self.tab_manager.folder_tab_widget
                )
            case FunctionType.VIDEO:
                self.handle_function_tab_state(
                    self.tab_manager.video_tab_widget, self.tab_manager.photo_tab_widget, self.tab_manager.folder_tab_widget, self.tab_manager.mapping_tab_widget
                )
            case _:
                pass

    @staticmethod
    def handle_function_tab_state(selected_tab: UiCropWidget, *other_tabs: UiCropWidget):
        """
        Sets the selection state of the selected tab to SELECTED and the selection state of other tabs to NOT_SELECTED.

        Args:
            selected_tab (UiCropWidget): The selected tab.
            *other_tabs (UiCropWidget): The other tabs.
        """
        selected_tab.selection_state = selected_tab.SELECTED
        for tab in other_tabs:
            tab.selection_state = tab.NOT_SELECTED

    def load_preset(self, phi: Preset) -> None:
        """
        Loads a preset value into the width and height line edits.

        Args:
            phi (Preset): The preset value to load.
        """
        def callback(control: UiCropControlWidget) -> None:
            if any(line.state is LineEditState.INVALID_INPUT for line in
                  (control.widthLineEdit, control.heightLineEdit)):
                control.widthLineEdit.setText('1000')
                control.heightLineEdit.setText('1000')

            match phi:
                case Preset.SQUARE:
                    if control.widthLineEdit.value() > control.heightLineEdit.value():
                        control.heightLineEdit.setText(control.widthLineEdit.text())
                    elif control.widthLineEdit.value() < control.heightLineEdit.value():
                        control.widthLineEdit.setText(control.heightLineEdit.text())
                case _:
                    if control.widthLineEdit.value() >= control.heightLineEdit.value():
                        control.heightLineEdit.setText(str(int(control.widthLineEdit.value() * phi.value)))
                    else:
                        control.widthLineEdit.setText(str(int(control.heightLineEdit.value() / phi.value)))

        match self.tab_manager.function_tabWidget.currentIndex():
            case FunctionType.PHOTO:
                callback(self.tab_manager.photo_tab_widget.controlWidget)
            case FunctionType.FOLDER:
                callback(self.tab_manager.folder_tab_widget.controlWidget)
            case FunctionType.MAPPING:
                callback(self.tab_manager.mapping_tab_widget.controlWidget)
            case FunctionType.VIDEO:
                callback(self.tab_manager.video_tab_widget.controlWidget)
            case _:
                pass

    # Drag and drop event handlers - delegate to drag drop handler
    def dragEnterEvent(self, a0: QDragEnterEvent | None) -> None:
        """Handle drag enter events for browser-like drag and drop"""
        if a0:
            self.drag_drop_handler.handle_drag_enter(a0)

    def dragMoveEvent(self, a0: QDragMoveEvent | None) -> None:
        """Handle drag move events"""
        if a0:
            self.drag_drop_handler.handle_drag_move(a0)

    def dropEvent(self, a0: QDropEvent | None) -> None:
        """Handle drop events with enhanced security"""
        if a0:
            self.drag_drop_handler.handle_drop(a0)

    def disable_buttons(self, tab_widget: TabWidget) -> None:
        """
        Disables buttons based on the filled state of line edits and combo boxes.

        Args:
            self: The instance of the class.
            tab_widget: The tab widget.

        Returns:
            None
        """

        common_line_edits = (tab_widget.controlWidget.widthLineEdit, tab_widget.controlWidget.heightLineEdit)
        check_button_state = partial(self.all_filled,
                                      self.address_bar.unified_address_bar,
                                      self.address_bar.destination_input,
                                      *common_line_edits)

        match tab_widget:
            case tab_widget if isinstance(tab_widget, UiPhotoTabWidget | UiFolderTabWidget):
                ut.change_widget_state(check_button_state(),
                    tab_widget.cropButton,
                )
            case tab_widget if isinstance(tab_widget, UiMappingTabWidget):
                ut.change_widget_state(
                    check_button_state(self.address_bar.secondary_input, tab_widget.comboBox_1, tab_widget.comboBox_2),
                    self.tab_manager.mapping_tab_widget.cropButton
                )
            case tab_widget if isinstance(tab_widget, UiVideoTabWidget):
                ut.change_widget_state(
                    check_button_state(),
                    tab_widget.mediacontrolWidget_1.cropButton,
                    tab_widget.mediacontrolWidget_2.cropButton,
                    tab_widget.mediacontrolWidget_1.videocropButton,
                    tab_widget.mediacontrolWidget_2.videocropButton
                )
            case _:
                pass

    @staticmethod
    def all_filled(*line_edits: PathLineEdit | NumberLineEdit | QComboBox) -> bool:
        x = all(edit.state == LineEditState.VALID_INPUT
                for edit in line_edits if isinstance(edit, PathLineEdit | NumberLineEdit))
        y = all(edit.currentText() for edit in line_edits if isinstance(edit, QComboBox))
        return x and y
