import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path

from PyQt6.QtCore import QByteArray, QCoreApplication, QMetaObject, QRect, QSize, Qt, QUrl
from PyQt6.QtGui import (
    QAction,
    QCloseEvent,
    QDragEnterEvent,
    QDragMoveEvent,
    QDropEvent,
    QIcon,
    QImage,
    QPainter,
    QPixmap,
)
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMenuBar,
    QPushButton,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core import face_tools as ft
from core import processing as prc
from core.config import logger
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

from .control_widget import UiCropControlWidget
from .crop_widget import UiCropWidget
from .enums import GuiIcon
from .folder_tab import UiFolderTabWidget
from .mapping_tab import UiMappingTabWidget
from .photo_tab import UiPhotoTabWidget
from .splash_screen import UiClickableSplashScreen
from .video_tab import UiVideoTabWidget

type TabWidget = UiPhotoTabWidget | UiFolderTabWidget | UiMappingTabWidget | UiVideoTabWidget


class UiMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)

        # Start with a splash screen
        splash = UiClickableSplashScreen()
        splash.show_message("Loading face detection models...")
        QApplication.processEvents()

        face_detection_tools = self.get_face_detection_tools(splash)

        # Single-threaded workers
        self.display_worker = DisplayCropper(face_detection_tools[0])
        self.photo_worker = PhotoCropper(face_detection_tools[0])
        self.video_worker = VideoCropper(face_detection_tools[0])

        # Multithreaded workers
        self.folder_worker = FolderCropper(face_detection_tools)
        self.mapping_worker = MappingCropper(face_detection_tools)

        # Create the central widget
        self.centralwidget = QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.main_layout = QVBoxLayout(self.centralwidget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        self.actionAbout_Face_Cropper = QAction(self)
        self.actionUse_Mapping = QAction(self)
        self.actionCrop_File = QAction(self)
        self.actionCrop_Folder = QAction(self)
        self.actionSquare = QAction(self)
        self.actionGolden_Ratio = QAction(self)
        self.action2_3_Ratio = QAction(self)
        self.action3_4_Ratio = QAction(self)
        self.action4_5_Ratio = QAction(self)
        self.actionCrop_Video = QAction(self)

        self.menubar = QMenuBar(self)
        self.menuFile = QMenu(self.menubar)
        self.menuTools = QMenu(self.menubar)
        self.menuInfo = QMenu(self.menubar)

        self.address_bar_widget = QWidget()
        self.back_button = QPushButton()
        self.forward_button = QPushButton()
        self.refresh_button = QPushButton()
        self.unified_address_bar = PathLineEdit(path_type=PathType.IMAGE)  # Default to image
        self.context_button = QPushButton()
        self.secondary_input_container = QWidget()
        self.secondary_input = PathLineEdit(path_type=PathType.TABLE)
        self.secondary_button = QPushButton()
        self.destination_container = QWidget()
        self.destination_label = QLabel("Save to:")
        self.destination_input = PathLineEdit(path_type=PathType.FOLDER)
        self.destination_button = QPushButton()
        self.info_button = QPushButton()

        self.function_tabWidget = QTabWidget(self.centralwidget)
        self.photo_tab = QWidget()
        self.verticalLayout_2 = ut.setup_vbox("verticalLayout_2", self.photo_tab)
        self.photo_tab_widget = UiPhotoTabWidget(self.photo_worker, "photo_tab_widget", self.photo_tab)
        self.folder_tab = QWidget()
        self.verticalLayout_3 = ut.setup_vbox("verticalLayout_3", self.folder_tab)
        self.folder_tab_widget = UiFolderTabWidget(self.folder_worker, "folder_tab_widget", self.folder_tab)
        self.mapping_tab = QWidget()
        self.verticalLayout_4 = ut.setup_vbox("verticalLayout_4", self.mapping_tab)
        self.mapping_tab_widget = UiMappingTabWidget(self.mapping_worker, "mapping_tab_widget", self.mapping_tab)
        self.video_tab = QWidget()
        self.verticalLayout_5 = ut.setup_vbox("verticalLayout_5", self.video_tab)
        self.video_tab_widget = UiVideoTabWidget(self.video_worker, "video_tab_widget", self.video_tab)

        self.setObjectName("MainWindow")
        self.resize(1256, 652)
        icon = QIcon()
        icon.addFile(GuiIcon.LOGO, QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.setWindowIcon(icon)

        # Create the main menu
        self.create_main_menu()

        # Create an address bar (browser-like)
        self.create_address_bar()

        # Create tab widget (browser-like)
        self.create_tab_widgets()

        self.video_worker.progressBars = [self.video_tab_widget.progressBar, self.video_tab_widget.progressBar_2]

        # Create a status bar
        self.statusbar = QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)

        # Connect signals
        self.connect_widgets()

        self.setCentralWidget(self.centralwidget)
        self.retranslateUi()

        # Set the initial tab
        self.function_tabWidget.setCurrentIndex(0)

        self.initialize_clear_button_states()

        QMetaObject.connectSlotsByName(self)

    def create_main_menu(self):
        """Create the main menu for the application"""
        # Create actions
        self.actionAbout_Face_Cropper.setObjectName("actionAbout_Face_Cropper")
        icon0 = QIcon.fromTheme("help-browser")
        self.actionAbout_Face_Cropper.setIcon(icon0)

        self.actionUse_Mapping.setObjectName("actionUse_Mapping")
        icon1 = QIcon()
        icon1.addFile(GuiIcon.EXCEL, QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.actionUse_Mapping.setIcon(icon1)

        self.actionCrop_File.setObjectName("actionCrop_File")
        icon2 = QIcon()
        icon2.addFile(GuiIcon.PICTURE, QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.actionCrop_File.setIcon(icon2)

        self.actionCrop_Folder.setObjectName("actionCrop_Folder")
        icon3 = QIcon()
        icon3.addFile(GuiIcon.FOLDER, QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.actionCrop_Folder.setIcon(icon3)

        self.actionSquare.setObjectName("actionSquare")
        self.actionGolden_Ratio.setObjectName("actionGolden_Ratio")
        self.action2_3_Ratio.setObjectName("action2_3_Ratio")
        self.action3_4_Ratio.setObjectName("action3_4_Ratio")
        self.action4_5_Ratio.setObjectName("action4_5_Ratio")

        self.decorate_action(self.actionSquare, 1, 1, "#3498db")
        self.decorate_action(self.actionGolden_Ratio, 100, 162, "#f39c12")
        self.decorate_action(self.action2_3_Ratio, 2, 3, "#2ecc71")
        self.decorate_action(self.action3_4_Ratio, 3, 4, "#e74c3c")
        self.decorate_action(self.action4_5_Ratio, 4, 5, "#9b59b6")

        self.actionCrop_Video.setObjectName("actionCrop_Video")
        icon4 = QIcon()
        icon4.addFile(GuiIcon.CLAPPERBOARD, QSize(), QIcon.Mode.Normal,
                      QIcon.State.Off)
        self.actionCrop_Video.setIcon(icon4)

        # Create menu bar
        self.menubar.setObjectName("menubar")
        self.menubar.setGeometry(QRect(0, 0, 1256, 22))

        # Create menus
        self.menuFile.setObjectName("menuFile")
        self.menuTools.setObjectName("menuTools")
        self.menuInfo.setObjectName("menuInfo")

        # Add menus to menu bar
        self.setMenuBar(self.menubar)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuTools.menuAction())
        self.menubar.addAction(self.menuInfo.menuAction())

        # Add actions to menus
        self.menuFile.addAction(self.actionSquare)
        self.menuFile.addAction(self.actionGolden_Ratio)
        self.menuFile.addAction(self.action2_3_Ratio)
        self.menuFile.addAction(self.action3_4_Ratio)
        self.menuFile.addAction(self.action4_5_Ratio)

        self.menuTools.addAction(self.actionCrop_File)
        self.menuTools.addAction(self.actionCrop_Folder)
        self.menuTools.addAction(self.actionUse_Mapping)
        self.menuTools.addAction(self.actionCrop_Video)

        self.menuInfo.addAction(self.actionAbout_Face_Cropper)

    def create_address_bar(self):
        """Create a dynamic, context-aware address bar layout"""
        # Address bar container
        self.address_bar_widget.setObjectName("addressBarWidget")
        self.address_bar_widget.setMinimumHeight(48)
        self.address_bar_widget.setMaximumHeight(48)

        # Address bar layout
        address_bar_layout = QHBoxLayout(self.address_bar_widget)
        address_bar_layout.setContentsMargins(10, 5, 10, 5)
        address_bar_layout.setSpacing(10)

        # Navigation buttons
        self.back_button.setIcon(QIcon.fromTheme("go-previous"))
        self.back_button.setObjectName("backButton")
        self.back_button.setToolTip("Back")
        self.back_button.setFixedSize(36, 36)

        self.forward_button.setIcon(QIcon.fromTheme("go-next"))
        self.forward_button.setObjectName("forwardButton")
        self.forward_button.setToolTip("Forward")
        self.forward_button.setFixedSize(36, 36)

        self.refresh_button.setIcon(QIcon.fromTheme("view-refresh"))
        self.refresh_button.setObjectName("refreshButton")
        self.refresh_button.setToolTip("Refresh")
        self.refresh_button.setFixedSize(36, 36)

        # Unified address bar (dynamic path field)
        self.unified_address_bar.setObjectName("unifiedAddressBar")
        self.unified_address_bar.setPlaceholderText("Enter path...")

        # Context-aware open button with changing icon
        self.context_button.setObjectName("contextButton")
        self.context_button.setToolTip("Open File")
        self.context_button.setIcon(QIcon(GuiIcon.PICTURE))  # Default icon
        self.context_button.setFixedSize(36, 36)

        # Secondary input for mapping tab (initially hidden)
        self.secondary_input_container.setObjectName("secondaryInputContainer")
        self.secondary_input_container.setVisible(False)  # Hidden by default

        secondary_layout = QHBoxLayout(self.secondary_input_container)
        secondary_layout.setContentsMargins(0, 0, 0, 0)
        secondary_layout.setSpacing(5)

        self.secondary_input.setObjectName("secondaryInput")
        self.secondary_input.setPlaceholderText("Select table file...")

        self.secondary_button.setObjectName("secondaryButton")
        self.secondary_button.setIcon(QIcon(GuiIcon.EXCEL))
        self.secondary_button.setFixedSize(36, 36)
        self.secondary_button.setToolTip("Open Table File")

        secondary_layout.addWidget(self.secondary_input)
        secondary_layout.addWidget(self.secondary_button)

        # Destination section (always visible)
        self.destination_container.setObjectName("destinationContainer")

        destination_layout = QHBoxLayout(self.destination_container)
        destination_layout.setContentsMargins(0, 0, 0, 0)
        destination_layout.setSpacing(5)

        self.destination_label.setObjectName("destinationLabel")

        self.destination_input.setObjectName("destinationInput")
        self.destination_input.setPlaceholderText("Select destination folder...")

        self.destination_button.setObjectName("destinationButton")
        self.destination_button.setIcon(QIcon(GuiIcon.FOLDER))
        self.destination_button.setFixedSize(36, 36)
        self.destination_button.setToolTip("Select Destination Folder")

        destination_layout.addWidget(self.destination_label)
        destination_layout.addWidget(self.destination_input)
        destination_layout.addWidget(self.destination_button)

        # Info button (on the right)
        self.info_button.setIcon(QIcon.fromTheme("help-browser"))
        self.info_button.setObjectName("infoButton")
        self.info_button.setToolTip("Info")
        self.info_button.setFixedSize(36, 36)

        # Add widgets to layout
        address_bar_layout.addWidget(self.back_button)
        address_bar_layout.addWidget(self.forward_button)
        address_bar_layout.addWidget(self.refresh_button)
        address_bar_layout.addWidget(self.unified_address_bar)
        address_bar_layout.addWidget(self.context_button)
        address_bar_layout.addWidget(self.secondary_input_container)
        address_bar_layout.addWidget(self.destination_container)
        address_bar_layout.addWidget(self.info_button)

        # Set stretch factors
        address_bar_layout.setStretch(3, 3)  # Unified address bar gets more space
        address_bar_layout.setStretch(5, 3)  # Secondary input gets more space
        address_bar_layout.setStretch(6, 4)  # Destination gets more space

        # Add to the main layout
        self.main_layout.addWidget(self.address_bar_widget)

    def create_tab_widgets(self):
        """Create browser-like tab widget layout"""
        # Create the tab widget
        self.function_tabWidget.setObjectName("function_tabWidget")
        self.function_tabWidget.setMovable(True)

        # Add tabs
        self.create_photo_tab()
        self.create_folder_tab()
        self.create_mapping_tab()
        self.create_video_tab()

        # Add tab widget to the main layout
        self.main_layout.addWidget(self.function_tabWidget)

    def create_photo_tab(self):
        """Create photo tab without redundant input fields"""
        icon2 = QIcon()
        icon2.addFile(GuiIcon.PICTURE, QSize(), QIcon.Mode.Normal, QIcon.State.Off)

        self.photo_tab.setObjectName("photo_tab")

        # Hide the redundant input fields that will be handled by unified address bar
        self.photo_tab_widget.horizontalLayout_2.setParent(None)  # Remove input layout
        self.photo_tab_widget.horizontalLayout_3.setParent(None)  # Remove destination layout

        self.verticalLayout_2.addWidget(self.photo_tab_widget)
        self.function_tabWidget.addTab(self.photo_tab, icon2, "")

    def create_folder_tab(self):
        """Create folder tab without redundant input fields"""
        icon3 = QIcon()
        icon3.addFile(GuiIcon.FOLDER, QSize(), QIcon.Mode.Normal, QIcon.State.Off)

        self.folder_tab.setObjectName("folder_tab")

        # Modify the setup to remove redundant input fields
        # Access the verticalLayout_200 in page_1 to remove input and destination layouts
        input_layout = None
        destination_layout = None

        # Find the input and destination layouts
        for i in range(self.folder_tab_widget.verticalLayout_200.count()):
            item = self.folder_tab_widget.verticalLayout_200.itemAt(i)
            if isinstance(item, QHBoxLayout) and hasattr(item, "objectName"):
                if item.objectName() == "horizontalLayout_4":  # Input layout
                    input_layout = item
                elif item.objectName() == "horizontalLayout_3":  # Destination layout
                    destination_layout = item

        # Remove input and destination layouts if found
        if input_layout:
            self.folder_tab_widget.verticalLayout_200.removeItem(input_layout)
        if destination_layout:
            self.folder_tab_widget.verticalLayout_200.removeItem(destination_layout)

        self.verticalLayout_3.addWidget(self.folder_tab_widget)
        self.function_tabWidget.addTab(self.folder_tab, icon3, "")

    def create_mapping_tab(self):
        """Create mapping tab without redundant input fields"""
        icon1 = QIcon()
        icon1.addFile(GuiIcon.EXCEL, QSize(), QIcon.Mode.Normal, QIcon.State.Off)

        self.mapping_tab.setObjectName("mapping_tab")

        self.verticalLayout_4.addWidget(self.mapping_tab_widget)
        self.function_tabWidget.addTab(self.mapping_tab, icon1, "")

    def create_video_tab(self):
        """Create video tab without redundant input fields"""
        icon4 = QIcon()
        icon4.addFile(GuiIcon.CLAPPERBOARD, QSize(), QIcon.Mode.Normal, QIcon.State.Off)

        self.video_tab.setObjectName("video_tab")

        # Register state retrieval functions for each widget type
        for func_type, widget in [
            (FunctionType.PHOTO, self.photo_tab_widget),
            (FunctionType.FOLDER, self.folder_tab_widget),
            (FunctionType.MAPPING, self.mapping_tab_widget),
            (FunctionType.VIDEO, self.video_tab_widget)
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

        self.verticalLayout_5.addWidget(self.video_tab_widget)
        self.function_tabWidget.addTab(self.video_tab, icon4, "")

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

    def decorate_action(self,action: QAction, width: float, height: float, color: str):
        """Set the aspect ratio icons for the address bar and destination input"""
        max_dim = 64
        # Compute a uniform scale factor that fits the longer side to `max_dim`.
        scale = max_dim / max(width, height)
        w, h = int(width * scale), int(height * scale)

        # Centre the rectangle on the square canvas.
        x_offset, y_offset = (max_dim - w) // 2, (max_dim - h) // 2

        # Create SVG data with the rectangle
        svg_data = f"""
        <svg width="{max_dim}" height="{max_dim}" xmlns="http://www.w3.org/2000/svg">
            <rect x="{x_offset}" y="{y_offset}" width="{w}" height="{h}" fill="{color}"
                stroke="#333333" stroke-width="2" rx="2" ry="2"/>
        </svg>
        """

        svg_bytes = QByteArray(svg_data.encode())
        renderer = QSvgRenderer(svg_bytes)

        # Create pixmap at multiple sizes for better scaling
        icon = QIcon()
        for size in [16, 24, 32, 48, 64]:
            pixmap = QPixmap(size, size)
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)
            renderer.render(painter)
            painter.end()
            icon.addPixmap(pixmap)

        action.setIcon(icon)

    def initialize_clear_button_states(self):
        """Initialize clear button states for all path line edits"""
        # Update clear button states for all path line edits
        self.unified_address_bar.update_clear_button(self.unified_address_bar.text())
        self.destination_input.update_clear_button(self.destination_input.text())
        self.secondary_input.update_clear_button(self.secondary_input.text())

        # Make sure visibility is properly set
        if not self.unified_address_bar.text():
            self.unified_address_bar.clearButton.setVisible(False)
        if not self.destination_input.text():
            self.destination_input.clearButton.setVisible(False)
        if not self.secondary_input.text():
            self.secondary_input.clearButton.setVisible(False)

    def update_address_bar_context(self):
        """Update address bar context based on current tab"""
        current_index = self.function_tabWidget.currentIndex()

        match current_index:
            case FunctionType.PHOTO:
                # Update primary address bar
                self.unified_address_bar.set_path_type(PathType.IMAGE)
                self.unified_address_bar.setPlaceholderText("Enter image file path...")
                self.context_button.setIcon(QIcon(GuiIcon.PICTURE))
                self.context_button.setToolTip("Open Image")

                # Hide secondary input (not needed for photo tab)
                self.secondary_input_container.setVisible(False)

                # Get the current path from tab widget if it exists
                self.unified_address_bar.blockSignals(True)
                # Only set text if path is not empty to prevent default drive issues
                if self.photo_tab_widget.input_path:
                    self.unified_address_bar.setText(self.photo_tab_widget.input_path)
                else:
                    self.unified_address_bar.clear()
                self.unified_address_bar.blockSignals(False)

                # Get a destination path
                self.destination_input.blockSignals(True)
                if self.photo_tab_widget.destination_path:
                    self.destination_input.setText(self.photo_tab_widget.destination_path)
                else:
                    self.destination_input.clear()
                self.destination_input.blockSignals(False)

            case FunctionType.FOLDER:
                self.unified_address_bar.set_path_type(PathType.FOLDER)
                self.unified_address_bar.setPlaceholderText("Enter folder path...")
                self.context_button.setIcon(QIcon(GuiIcon.FOLDER))
                self.context_button.setToolTip("Select Folder")

                # Hide secondary input (not needed for folder tab)
                self.secondary_input_container.setVisible(False)

                # Get the current path from tab widget if it exists
                self.unified_address_bar.blockSignals(True)
                if self.folder_tab_widget.input_path:
                    self.unified_address_bar.setText(self.folder_tab_widget.input_path)
                else:
                    self.unified_address_bar.clear()
                self.unified_address_bar.blockSignals(False)

                # Get a destination path
                self.destination_input.blockSignals(True)
                if self.folder_tab_widget.destination_path:
                    self.destination_input.setText(self.folder_tab_widget.destination_path)
                else:
                    self.destination_input.clear()
                self.destination_input.blockSignals(False)

            case FunctionType.MAPPING:
                self.unified_address_bar.set_path_type(PathType.FOLDER)
                self.unified_address_bar.setPlaceholderText("Enter source folder path...")
                self.context_button.setIcon(QIcon(GuiIcon.FOLDER))
                self.context_button.setToolTip("Select Source Folder")

                # Show and configure secondary input for mapping tab
                self.secondary_input_container.setVisible(True)
                self.secondary_input.set_path_type(PathType.TABLE)
                self.secondary_input.setPlaceholderText("Enter table file path...")

                # Get current paths from tab widget if they exist
                self.unified_address_bar.blockSignals(True)
                if self.mapping_tab_widget.input_path:
                    self.unified_address_bar.setText(self.mapping_tab_widget.input_path)
                else:
                    self.unified_address_bar.clear()
                self.unified_address_bar.blockSignals(False)

                self.secondary_input.blockSignals(True)
                if self.mapping_tab_widget.table_path:
                    self.secondary_input.setText(self.mapping_tab_widget.table_path)
                else:
                    self.secondary_input.clear()
                self.secondary_input.blockSignals(False)

                # Get a destination path
                self.destination_input.blockSignals(True)
                if self.mapping_tab_widget.destination_path:
                    self.destination_input.setText(self.mapping_tab_widget.destination_path)
                else:
                    self.destination_input.clear()
                self.destination_input.blockSignals(False)

            case FunctionType.VIDEO:
                self.unified_address_bar.set_path_type(PathType.VIDEO)
                self.unified_address_bar.setPlaceholderText("Enter video file path...")
                self.context_button.setIcon(QIcon(GuiIcon.CLAPPERBOARD))
                self.context_button.setToolTip("Open Video")

                # Hide secondary input (not needed for video tab)
                self.secondary_input_container.setVisible(False)

                # Get the current path from tab widget if it exists
                self.unified_address_bar.blockSignals(True)
                if self.video_tab_widget.input_path:
                    self.unified_address_bar.setText(self.video_tab_widget.input_path)
                else:
                    self.unified_address_bar.clear()
                self.unified_address_bar.blockSignals(False)

                # Get a destination path
                self.destination_input.blockSignals(True)
                if self.video_tab_widget.destination_path:
                    self.destination_input.setText(self.video_tab_widget.destination_path)
                else:
                    self.destination_input.clear()
                self.destination_input.blockSignals(False)

        # Force update of clear button visibility after changing tab
        # This is important since we block signals during setText operations
        self.unified_address_bar.update_clear_button(self.unified_address_bar.text())
        self.destination_input.update_clear_button(self.destination_input.text())
        if self.secondary_input_container.isVisible():
            self.secondary_input.update_clear_button(self.secondary_input.text())

    # retranslateUi
    def retranslateUi(self):
        self.setWindowTitle(QCoreApplication.translate("self", "Autocrop", None))
        self.actionAbout_Face_Cropper.setText(QCoreApplication.translate("self", "About Autocrop", None))
        self.actionUse_Mapping.setText(QCoreApplication.translate("self", "Use Mapping", None))
        self.actionCrop_File.setText(QCoreApplication.translate("self", "Crop File", None))
        self.actionCrop_Folder.setText(QCoreApplication.translate("self", "Crop Folder", None))
        self.actionSquare.setText(QCoreApplication.translate("self", "Square", None))
        self.actionGolden_Ratio.setText(QCoreApplication.translate("self", "Golden Ratio", None))
        self.action2_3_Ratio.setText(QCoreApplication.translate("self", "2:3 Ratio", None))
        self.action3_4_Ratio.setText(QCoreApplication.translate("self", "3:4 Ratio", None))
        self.action4_5_Ratio.setText(QCoreApplication.translate("self", "4:5 Ratio", None))
        self.actionCrop_Video.setText(QCoreApplication.translate("self", "Crop Video", None))
        self.function_tabWidget.setTabText(self.function_tabWidget.indexOf(self.photo_tab),
                                          QCoreApplication.translate("self", "Photo Crop", None))
        self.function_tabWidget.setTabText(self.function_tabWidget.indexOf(self.folder_tab),
                                          QCoreApplication.translate("self", "Folder Crop", None))
        self.function_tabWidget.setTabText(self.function_tabWidget.indexOf(self.mapping_tab),
                                          QCoreApplication.translate("self", "Mapping Crop", None))
        self.function_tabWidget.setTabText(self.function_tabWidget.indexOf(self.video_tab),
                                          QCoreApplication.translate("self", "Video Crop", None))
        self.menuFile.setTitle(QCoreApplication.translate("self", "Presets", None))
        self.menuTools.setTitle(QCoreApplication.translate("self", "Tools", None))
        self.menuInfo.setTitle(QCoreApplication.translate("self", "Info", None))

        # Address bar elements
        self.unified_address_bar.setPlaceholderText(QCoreApplication.translate("self",
                                            "Enter file path or drag and drop files here", None))
        self.back_button.setToolTip(QCoreApplication.translate("self", "Back", None))
        self.forward_button.setToolTip(QCoreApplication.translate("self", "Forward", None))
        self.refresh_button.setToolTip(QCoreApplication.translate("self", "Refresh Preview", None))
        self.info_button.setToolTip(QCoreApplication.translate("self", "Settings", None))

    def connect_widgets(self):
        """Connect widget signals to handlers"""
        # CONNECTIONS
        self.connect_combo_boxes(self.mapping_tab_widget)

        # Menu actions
        self.actionAbout_Face_Cropper.triggered.connect(ut.load_about_form)
        self.actionGolden_Ratio.triggered.connect(partial(self.load_preset, Preset.GOLDEN_RATIO))
        self.action2_3_Ratio.triggered.connect(partial(self.load_preset, Preset.TWO_THIRDS))
        self.action3_4_Ratio.triggered.connect(partial(self.load_preset, Preset.THREE_QUARTERS))
        self.action4_5_Ratio.triggered.connect(partial(self.load_preset, Preset.FOUR_FIFTHS))
        self.actionSquare.triggered.connect(partial(self.load_preset, Preset.SQUARE))

        # Connect tab selection actions
        self.actionCrop_File.triggered.connect(partial(self.function_tabWidget.setCurrentIndex, FunctionType.PHOTO))
        self.actionCrop_Folder.triggered.connect(partial(self.function_tabWidget.setCurrentIndex, FunctionType.FOLDER))
        self.actionUse_Mapping.triggered.connect(partial(self.function_tabWidget.setCurrentIndex, FunctionType.MAPPING))
        self.actionCrop_Video.triggered.connect(partial(self.function_tabWidget.setCurrentIndex, FunctionType.VIDEO))

        # Browser-style navigation buttons
        self.back_button.clicked.connect(self.navigate_back)
        self.forward_button.clicked.connect(self.navigate_forward)
        self.refresh_button.clicked.connect(self.refresh_current_view)
        self.info_button.clicked.connect(self.show_settings)

        # Connect unified address bar signals
        self.connect_address_bar_signals()

        # Tab change event
        self.function_tabWidget.currentChanged.connect(self.check_tab_selection)
        self.function_tabWidget.currentChanged.connect(self.video_tab_widget.player.pause)

        # Error handling connections
        self.folder_worker.error.connect(self.folder_tab_widget.disable_buttons)
        self.photo_worker.error.connect(self.photo_tab_widget.disable_buttons)
        self.mapping_worker.error.connect(self.mapping_tab_widget.disable_buttons)
        self.video_worker.error.connect(self.video_tab_widget.disable_buttons)

    def connect_address_bar_signals(self):
        """Connect signals for the unified address bar"""
        # Connect tab change to update address bar context
        self.function_tabWidget.currentChanged.connect(self.update_address_bar_context)

        # Connect the unified address bar to update the current tab's input field
        self.unified_address_bar.textChanged.connect(self.unified_address_changed)
        self.unified_address_bar.textChanged.connect(self.update_all_button_states)  # Add this line

        # Connect secondary input for mapping tab
        self.secondary_input.textChanged.connect(self.secondary_input_changed)
        self.secondary_input.textChanged.connect(self.update_all_button_states)  # Add this line

        # Connect destination input
        self.destination_input.textChanged.connect(self.destination_input_changed)
        self.destination_input.textChanged.connect(self.update_all_button_states)  # Add this line

        # Connect context-aware buttons
        self.context_button.clicked.connect(self.handle_context_button)
        self.secondary_button.clicked.connect(self.handle_secondary_button)
        self.destination_button.clicked.connect(self.handle_destination_button)

    def update_all_button_states(self):
        """Update button states for all tab widgets when address bars change"""
        # Update button states for all tabs
        for tab_widget in [
            self.photo_tab_widget,
            self.folder_tab_widget,
            self.mapping_tab_widget,
            self.video_tab_widget
        ]:
            tab_widget.tab_state_manager.update_button_states()

    def unified_address_changed(self, text: str):
        """Handle changes to the unified address bar"""
        current_index = self.function_tabWidget.currentIndex()

        # Update the appropriate input field in the current tab
        if cleaned_text := ut.sanitize_path(text) if text.strip() else "":
            match current_index:
                case FunctionType.PHOTO:
                    self.photo_tab_widget.input_path = cleaned_text
                case FunctionType.FOLDER:
                    self.folder_tab_widget.input_path = cleaned_text
                case FunctionType.MAPPING:
                    self.mapping_tab_widget.input_path = cleaned_text
                case FunctionType.VIDEO:
                    self.video_tab_widget.input_path = cleaned_text
        else:
            # Clear the path if text is empty or invalid
            match current_index:
                case FunctionType.PHOTO:
                    self.photo_tab_widget.input_path = ""
                case FunctionType.FOLDER:
                    self.folder_tab_widget.input_path = ""
                case FunctionType.MAPPING:
                    self.mapping_tab_widget.input_path = ""
                case FunctionType.VIDEO:
                    self.video_tab_widget.input_path = ""

        # Trigger preview update for the current tab (only if valid)
        if text.strip() and self.unified_address_bar.state == LineEditState.VALID_INPUT:
            self.trigger_preview_update()

    def trigger_preview_update(self):
        """Trigger preview update for the current tab when path is valid"""
        current_index = self.function_tabWidget.currentIndex()

        # Check if the path is valid before updating
        if self.unified_address_bar.state == LineEditState.VALID_INPUT:
            match current_index:
                case FunctionType.PHOTO:
                    if self.photo_tab_widget.input_path:
                        self.display_worker.crop(FunctionType.PHOTO)
                case FunctionType.FOLDER:
                    if self.folder_tab_widget.input_path:
                        self.folder_tab_widget.load_data()
                        self.display_worker.crop(FunctionType.FOLDER)
                case FunctionType.MAPPING:
                    if self.mapping_tab_widget.input_path:
                        # Load folder data for mapping tab too
                        self.mapping_tab_widget.load_data()
                        self.display_worker.crop(FunctionType.MAPPING)
                case FunctionType.VIDEO:
                    if self.video_tab_widget.input_path:
                        self.display_worker.crop(FunctionType.VIDEO)

    def secondary_input_changed(self, text: str):
        """Handle changes to the secondary input (only for mapping tab)"""
        current_index = self.function_tabWidget.currentIndex()

        if current_index == FunctionType.MAPPING:
            if cleaned_text := ut.sanitize_path(text) if text.strip() else "":
                self.mapping_tab_widget.table_path = cleaned_text
            else:
                self.mapping_tab_widget.table_path = ""

    def destination_input_changed(self, text: str):
        """Handle changes to the destination input"""
        current_index = self.function_tabWidget.currentIndex()

        if cleaned_text := ut.sanitize_path(text) if text.strip() else "":
            match current_index:
                case FunctionType.PHOTO:
                    self.photo_tab_widget.destination_path = cleaned_text
                case FunctionType.FOLDER:
                    self.folder_tab_widget.destination_path = cleaned_text
                case FunctionType.MAPPING:
                    self.mapping_tab_widget.destination_path = cleaned_text
                case FunctionType.VIDEO:
                    self.video_tab_widget.destination_path = cleaned_text
        else:
            # Clear destination path if text is empty or invalid
            match current_index:
                case FunctionType.PHOTO:
                    self.photo_tab_widget.destination_path = ""
                case FunctionType.FOLDER:
                    self.folder_tab_widget.destination_path = ""
                case FunctionType.MAPPING:
                    self.mapping_tab_widget.destination_path = ""
                case FunctionType.VIDEO:
                    self.video_tab_widget.destination_path = ""

    def handle_context_button(self):
        """Handle clicks on the context-aware button"""
        match self.function_tabWidget.currentIndex():
            case FunctionType.PHOTO:
                self.open_file_dialog(PathType.IMAGE, self.unified_address_bar)
            case FunctionType.FOLDER:
                self.open_folder_dialog(self.unified_address_bar)
            case FunctionType.MAPPING:
                self.open_folder_dialog(self.unified_address_bar)
            case FunctionType.VIDEO:
                self.open_file_dialog(PathType.VIDEO, self.unified_address_bar)

    def handle_secondary_button(self):
        """Handle clicks on the secondary button (only for mapping tab)"""
        match self.function_tabWidget.currentIndex():
            case FunctionType.MAPPING:
                self.open_file_dialog(PathType.TABLE, self.secondary_input)
            case _:
                pass

    def handle_destination_button(self):
        """Handle clicks on the destination button"""
        self.open_folder_dialog(self.destination_input)

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
        current_index = self.function_tabWidget.currentIndex()

        if tab_widget := self._get_current_tab_widget():
            if path_type == PathType.IMAGE:
                tab_widget.input_path = file_path
                if current_index == FunctionType.PHOTO:
                    self.unified_address_bar.setText(file_path)
            elif path_type == PathType.VIDEO:
                tab_widget.input_path = file_path
                self.video_tab_widget.player.setSource(QUrl.fromLocalFile(file_path))
                self.video_tab_widget.reset_video_widgets()
            elif path_type == PathType.TABLE:
                tab_widget.table_path = file_path
                data = prc.load_table(Path(file_path))
                self.mapping_tab_widget.process_data(data)
            elif path_type == PathType.FOLDER:
                # New: Handle folder paths for mapping tab
                tab_widget.input_path = file_path
                if current_index == FunctionType.MAPPING:
                    # Load the folder data into the mapping tab's tree view
                    self.mapping_tab_widget.load_data()

    def _get_current_tab_widget(self) -> TabWidget | None:
        """Get the currently active tab widget"""
        current_index = self.function_tabWidget.currentIndex()

        widget_map = {
            FunctionType.PHOTO: self.photo_tab_widget,
            FunctionType.FOLDER: self.folder_tab_widget,
            FunctionType.MAPPING: self.mapping_tab_widget,
            FunctionType.VIDEO: self.video_tab_widget
        }

        return widget_map.get(current_index)

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
                current_index = self.function_tabWidget.currentIndex()
                if target == self.unified_address_bar:
                    if current_index == FunctionType.FOLDER:
                        self.folder_tab_widget.load_data()
                    elif current_index == FunctionType.MAPPING:
                        # Also load data for mapping tab
                        self.mapping_tab_widget.load_data()

        except Exception as e:
            # Log error internally without exposing details
            ut.show_error_box(f"An error occurred opening the directory\n{e}")

    # Browser-style navigation methods
    def navigate_back(self):
        """Navigate to the previous tab"""
        current_index = self.function_tabWidget.currentIndex()
        if current_index > 0:
            self.function_tabWidget.setCurrentIndex(current_index - 1)

    def navigate_forward(self):
        """Navigate to the next tab"""
        current_index = self.function_tabWidget.currentIndex()
        if current_index < self.function_tabWidget.count() - 1:
            self.function_tabWidget.setCurrentIndex(current_index + 1)

    def refresh_current_view(self):
        """Refresh the current tab's view"""
        # Handle refresh based on the current tab
        match self.function_tabWidget.currentIndex():
            case FunctionType.PHOTO:
                # Refresh photo preview
                if self.unified_address_bar.text() and self.unified_address_bar.state == LineEditState.VALID_INPUT:
                    self.display_worker.crop(FunctionType.PHOTO)
            case FunctionType.FOLDER:
                # Refresh folder view
                self.folder_tab_widget.load_data()
            case FunctionType.MAPPING:
                # Refresh mapping preview
                if self.secondary_input.text() and self.secondary_input.state == LineEditState.VALID_INPUT:
                    if file_path := Path(self.secondary_input.text()):
                        data = prc.load_table(file_path)
                        self.mapping_tab_widget.process_data(data)
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
        match self.function_tabWidget.currentIndex():
            case FunctionType.PHOTO:
                path = self.photo_tab_widget.input_path
            case FunctionType.FOLDER:
                path = self.folder_tab_widget.input_path
            case FunctionType.MAPPING:
                path = self.mapping_tab_widget.table_path
            case FunctionType.VIDEO:
                path = self.video_tab_widget.input_path
            case _:
                return None

        # Update the address bar without triggering the text_changed event
        self.unified_address_bar.blockSignals(True)
        self.unified_address_bar.setText(path)
        self.unified_address_bar.blockSignals(False)
        return None

    def update_current_tab_path(self):
        """Update the current tab's path with address bar text"""
        path = self.unified_address_bar.text()

        # Update the appropriate input field based on the current tab
        match self.function_tabWidget.currentIndex():
            case FunctionType.PHOTO:
                self.photo_tab_widget.input_path = path
            case FunctionType.FOLDER:
                self.folder_tab_widget.input_path = path
            case FunctionType.MAPPING:
                self.mapping_tab_widget.table_path = path
            case FunctionType.VIDEO:
                self.video_tab_widget.input_path = path
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
        self.video_tab_widget.player.stop()

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
        self.function_tabWidget.setTabsClosable(True)
        self.function_tabWidget.tabCloseRequested.connect(self.handle_tab_close)

        # Initialize navigation button states
        self.update_navigation_button_states()

    def handle_tab_close(self, index: int):
        """
        Handle tab close button clicks.
        For a browser-like experience, we don't close tabs but reset their state.
        """
        # Doesn't close the tab, just resets its state

        # Show a message in the status bar
        self.statusbar.showMessage(f"Tab {self.function_tabWidget.tabText(index)} reset", 2000)

        # Reset the appropriate tab widget
        match index:
            case FunctionType.PHOTO:
                self.photo_tab_widget.input_path = ''
                self.photo_tab_widget.destination_path = ''
            case FunctionType.FOLDER:
                self.folder_tab_widget.input_path = ''
                self.folder_tab_widget.destination_path = ''
            case FunctionType.MAPPING:
                self.mapping_tab_widget.input_path = ''
                self.mapping_tab_widget.table_path = ''
                self.mapping_tab_widget.destination_path = ''
            case FunctionType.VIDEO:
                self.video_tab_widget.input_path = ''
                self.video_tab_widget.destination_path = ''
                self.video_tab_widget.player.stop()

        # Update the address bar
        self.update_address_bar()

    def update_navigation_button_states(self):
        """Update the state of navigation buttons"""
        current_index = self.function_tabWidget.currentIndex()
        tab_count = self.function_tabWidget.count()

        # Enable/disable back button
        self.back_button.setEnabled(current_index > 0)

        # Enable/disable forward button
        self.forward_button.setEnabled(current_index < tab_count - 1)

    def check_tab_selection(self) -> None:
        """
        Checks the current selection of the function tab widget and handles the tab states accordingly.
        """
        # Update navigation buttons
        self.update_navigation_button_states()

        # Update unified address bar context
        self.update_address_bar_context()

        # Force validation of path inputs
        self.unified_address_bar.validate_path()
        self.destination_input.validate_path()
        if self.secondary_input_container.isVisible():
            self.secondary_input.validate_path()

        # Process tab selection as before
        match self.function_tabWidget.currentIndex():
            case FunctionType.PHOTO:
                self.handle_function_tab_state(
                    self.photo_tab_widget, self.folder_tab_widget, self.mapping_tab_widget, self.video_tab_widget
                )
            case FunctionType.FOLDER:
                self.handle_function_tab_state(
                    self.folder_tab_widget, self.mapping_tab_widget, self.video_tab_widget, self.photo_tab_widget
                )
            case FunctionType.MAPPING:
                self.handle_function_tab_state(
                    self.mapping_tab_widget, self.video_tab_widget, self.photo_tab_widget, self.folder_tab_widget
                )
            case FunctionType.VIDEO:
                self.handle_function_tab_state(
                    self.video_tab_widget, self.photo_tab_widget, self.folder_tab_widget, self.mapping_tab_widget
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

        match self.function_tabWidget.currentIndex():
            case FunctionType.PHOTO:
                callback(self.photo_tab_widget.controlWidget)
            case FunctionType.FOLDER:
                callback(self.folder_tab_widget.controlWidget)
            case FunctionType.MAPPING:
                callback(self.mapping_tab_widget.controlWidget)
            case FunctionType.VIDEO:
                callback(self.video_tab_widget.controlWidget)
            case _:
                pass

    # Drag and drop event handlers
    def dragEnterEvent(self, a0: QDragEnterEvent | None) -> None:
        """Handle drag enter events for browser-like drag and drop"""
        try:
            assert isinstance(a0, QDragEnterEvent)
        except AssertionError:
            return
        ut.check_mime_data(a0)

        # Show a status message
        self.statusbar.showMessage("Drop files here to open", 2000)

    def dragMoveEvent(self, a0: QDragMoveEvent | None) -> None:
        """Handle drag move events"""
        try:
            assert isinstance(a0, QDragMoveEvent)
        except AssertionError:
            return
        ut.check_mime_data(a0)

    def dropEvent(self, a0: QDropEvent | None) -> None:
        """
        Handle drop events with enhanced security.
        """
        try:
            assert isinstance(a0, QDropEvent)
        except AssertionError:
            return

        if (mime_data := a0.mimeData()) is None:
            return

        if not mime_data.hasUrls():
            a0.ignore()
            return

        a0.setDropAction(Qt.DropAction.CopyAction)

        # Get the dropped URL and convert to a local file path
        try:
            url = mime_data.urls()[0]
            if not url.isLocalFile():
                ut.show_error_box("Only local files can be dropped")
                a0.ignore()
                return
        except IndexError:
            ut.show_error_box("File may be corrupted or not accessible")
            a0.ignore()
            return

        file_path_str = url.toLocalFile()

        # Validate the path with our improved sanitize_path function
        if not (safe_path_str := ut.sanitize_path(file_path_str)):
            a0.ignore()
            return

        # Create a Path object from the sanitized path
        file_path = Path(safe_path_str).resolve()

        # Handle the file based on its type with proper error handling
        try:
            if file_path.is_dir():
                self._handle_dropped_directory(file_path)
            elif file_path.is_file():
                self._handle_dropped_file(file_path)
            else:
                ut.show_error_box("Dropped item is neither a file nor a directory")
                a0.ignore()
                return

            a0.accept()
            self.statusbar.showMessage(f"Opened {file_path.name}", 2000)

        except Exception as e:
            # Log error internally without exposing details
            logger.exception(f"Error processing dropped file: {e}")
            ut.show_error_box("An error occurred processing the dropped item")
            a0.ignore()

    def _handle_dropped_directory(self, dir_path: Path) -> None:
        """
        Securely handle a dropped directory.
        """
        # Verify it's a valid directory
        if not dir_path.is_dir() or not os.access(dir_path, os.R_OK):
            ut.show_error_box("Directory is not accessible")
            return

        # Check directory contents to determine appropriate tab
        try:
            # Look for table files to determine if this is a mapping operation
            has_table_files = any(
                file_manager.is_valid_type(f, FileCategory.TABLE)
                for f in dir_path.iterdir()
                if f.is_file() and os.access(f, os.R_OK)
            )
            self.blockSignals(True)

            if has_table_files:
                # Handle as mapping tab
                self.function_tabWidget.setCurrentIndex(FunctionType.MAPPING)
                self.mapping_tab_widget.input_path = dir_path.as_posix()
                self.unified_address_bar.setText(self.mapping_tab_widget.input_path)

                # Update display
                self.display_worker.current_paths[FunctionType.MAPPING] = None
                self.display_worker.crop(FunctionType.MAPPING)
            else:
                # Handle as folder tab
                self.function_tabWidget.setCurrentIndex(FunctionType.FOLDER)
                self.folder_tab_widget.input_path = dir_path.as_posix()
                self.unified_address_bar.setText(self.folder_tab_widget.input_path)

                # Update display
                self.display_worker.current_paths[FunctionType.FOLDER] = None
                self.folder_tab_widget.load_data()
                self.display_worker.crop(FunctionType.FOLDER)

            self.blockSignals(False)
        except OSError as e:
            # Log error internally without exposing details
            logger.exception(f"Error handling dropped directory: {e}")
            ut.show_error_box("An error occurred processing the directory")

    def _handle_dropped_file(self, file_path: Path) -> None:
        """
        Securely handle a dropped file.
        """
        # Verify it's a valid file
        if not file_path.is_file() or not os.access(file_path, os.R_OK):
            ut.show_error_box("File is not accessible")
            return

        try:
            self.blockSignals(True)
            # Determine the file type and handle accordingly
            if file_manager.is_valid_type(file_path, FileCategory.PHOTO):
                # Photo file - verify PHOTO signature
                if SignatureChecker.verify_file_type(file_path, FileCategory.PHOTO):
                    self._handle_image_drop(file_path)
                else:
                    ut.show_error_box("File content doesn't match its extension (expected photo)")

            elif file_manager.is_valid_type(file_path, FileCategory.RAW):
                # RAW file - verify RAW signature
                if SignatureChecker.verify_file_type(file_path, FileCategory.RAW):
                    self._handle_image_drop(file_path)
                else:
                    ut.show_error_box("File content doesn't match its extension (expected RAW)")

            elif file_manager.is_valid_type(file_path, FileCategory.TIFF):
                # TIFF file - verify TIFF signature
                if SignatureChecker.verify_file_type(file_path, FileCategory.TIFF):
                    self._handle_image_drop(file_path)
                else:
                    ut.show_error_box("File content doesn't match its extension (expected TIFF)")

            elif file_manager.is_valid_type(file_path, FileCategory.VIDEO):
                # Video file - verify VIDEO signature
                if SignatureChecker.verify_file_type(file_path, FileCategory.VIDEO):
                    # Handle with the video tab
                    self.function_tabWidget.setCurrentIndex(FunctionType.VIDEO)
                    self.video_tab_widget.input_path = file_path.as_posix()
                    self.unified_address_bar.setText(self.video_tab_widget.input_path)
                    self.video_tab_widget.open_dropped_video()
                else:
                    ut.show_error_box("File content doesn't match its extension (expected video)")

            elif file_manager.is_valid_type(file_path, FileCategory.TABLE):
                # Table file - verify TABLE signature
                if SignatureChecker.verify_file_type(file_path, FileCategory.TABLE):
                    # Handle with the mapping tab
                    self.function_tabWidget.setCurrentIndex(FunctionType.MAPPING)
                    self.mapping_tab_widget.table_path = file_path.as_posix()
                    self.secondary_input.setText(self.mapping_tab_widget.table_path)

                    # Process the table data
                    data = prc.load_table(file_path)
                    self.mapping_tab_widget.process_data(data)
                else:
                    ut.show_error_box("File content doesn't match its extension (expected table)")

            else:
                ut.show_error_box(f"Unsupported file type: {file_path.suffix}")

            self.blockSignals(False)
        except (OSError, ValueError, TypeError) as e:
            # Log error internally without exposing details
            logger.exception(f"Error handling dropped file: {e}")
            ut.show_error_box("An error occurred processing the file")

    def _handle_image_drop(self, file_path: Path) -> None:
        """Handle dropping an image file (photo, RAW, or TIFF)"""
        self.function_tabWidget.setCurrentIndex(FunctionType.PHOTO)
        self.photo_tab_widget.input_path = file_path.as_posix()
        self.unified_address_bar.setText(self.photo_tab_widget.input_path)
        self.display_worker.crop(FunctionType.PHOTO)

    def handle_path_main(self, file_path: Path) -> None:
        """
        Handles a file path by checking the file extensions in the directory.
        """
        extensions = {y.suffix.lower() for y in file_path.iterdir()}
        if extensions & file_manager.get_extensions(FileCategory.TABLE):
            self.handle_path(file_path, FunctionType.MAPPING, self.secondary_input)
        else:
            self.handle_path(file_path, FunctionType.FOLDER, self.unified_address_bar)

    def handle_path(self, file_path: Path,
                    tab_index: FunctionType,
                    line_edit: PathLineEdit) -> None:
        """
        Handles a file path by setting the function tab widget to the specified tab index.
        """
        self.function_tabWidget.setCurrentIndex(tab_index.value)
        line_edit.setText(file_path.as_posix())

        try:
            assert isinstance(self.mapping_tab_widget, UiMappingTabWidget)
            assert isinstance(self.folder_tab_widget, UiFolderTabWidget)
            assert isinstance(self.photo_tab_widget, UiPhotoTabWidget)
        except AssertionError:
            return

        if self.photo_tab_widget.selection_state == self.photo_tab_widget.SELECTED:
            self.handle_function_tab_state(self.photo_tab_widget, self.folder_tab_widget,
                                           self.mapping_tab_widget, self.video_tab_widget)
        elif self.folder_tab_widget.selection_state == self.folder_tab_widget.SELECTED:
            self.handle_function_tab_state(self.folder_tab_widget, self.photo_tab_widget,
                                           self.mapping_tab_widget, self.video_tab_widget)
            self.folder_tab_widget.load_data()
        elif self.mapping_tab_widget.selection_state == self.mapping_tab_widget.SELECTED:
            self.handle_function_tab_state(self.mapping_tab_widget, self.photo_tab_widget,
                                           self.folder_tab_widget, self.video_tab_widget)

    def handle_file(self, file_path: Path) -> None:
        """
        Handles a file based on its file extension by calling the appropriate handler method.
        """
        if (
            file_manager.is_valid_type(file_path, FileCategory.PHOTO) or
            file_manager.is_valid_type(file_path, FileCategory.TIFF) or
            file_manager.is_valid_type(file_path, FileCategory.RAW)
        ):
            self.handle_image_file(file_path)
        elif file_manager.is_valid_type(file_path, FileCategory.VIDEO):
            self.handle_video_file(file_path)
        elif file_manager.is_valid_type(file_path, FileCategory.TABLE):
            self.handle_table_file(file_path)

    def handle_image_file(self, file_path: Path) -> None:
        """
        Handles an image by validating the file path and calling the handle_path method with the appropriate arguments.
        """
        try:
            assert isinstance(self.photo_tab_widget, UiPhotoTabWidget)
        except AssertionError:
            return
        self.handle_path(file_path, FunctionType.PHOTO, self.unified_address_bar)

    def handle_video_file(self, file_path: Path) -> None:
        """
        Handles a video by setting the function tab widget to the video tab, and configuring the video player.
        """
        self.handle_function_tab_state(self.video_tab_widget, self.folder_tab_widget, self.photo_tab_widget,
                                      self.mapping_tab_widget)
        self.function_tabWidget.setCurrentIndex(FunctionType.VIDEO)
        try:
            assert isinstance(self.video_tab_widget, UiVideoTabWidget)
        except AssertionError:
            return
        self.unified_address_bar.setText(file_path.as_posix())
        self.video_tab_widget.mediacontrolWidget_1.playButton.setEnabled(True)
        self.video_tab_widget.mediacontrolWidget_1.playButton.setIcon(QIcon(GuiIcon.MULTIMEDIA_PLAY))
        self.video_tab_widget.mediacontrolWidget_2.playButton.setEnabled(True)
        self.video_tab_widget.mediacontrolWidget_2.playButton.setIcon(QIcon(GuiIcon.MULTIMEDIA_PLAY))
        self.video_tab_widget.open_dropped_video()

    def handle_table_file(self, file_path: Path) -> None:
        """
        Handles a table file by setting the function tab widget to the mapping tab, and opening the table.
        """
        self.function_tabWidget.setCurrentIndex(FunctionType.MAPPING)
        self.mapping_tab_widget.table_path = file_path.as_posix()
        data = prc.load_table(file_path)
        self.mapping_tab_widget.process_data(data)

    @staticmethod
    def all_filled(*line_edits: PathLineEdit | NumberLineEdit | QComboBox) -> bool:
        x = all(edit.state == LineEditState.VALID_INPUT
                for edit in line_edits if isinstance(edit, PathLineEdit | NumberLineEdit))
        y = all(edit.currentText() for edit in line_edits if isinstance(edit, QComboBox))
        return x and y

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
        check_button_state  = partial(self.all_filled,
                                      self.unified_address_bar,
                                      self.destination_input,
                                      *common_line_edits)

        match tab_widget:
            case tab_widget if isinstance(tab_widget, UiPhotoTabWidget | UiFolderTabWidget):
                ut.change_widget_state(check_button_state(),
                    tab_widget.cropButton,
                )
            case tab_widget if isinstance(tab_widget, UiMappingTabWidget):
                ut.change_widget_state(
                    check_button_state(self.secondary_input, tab_widget.comboBox_1, tab_widget.comboBox_2),
                    self.mapping_tab_widget.cropButton
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
