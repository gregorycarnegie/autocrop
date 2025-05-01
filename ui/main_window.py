import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Union, Optional

from PyQt6 import QtCore, QtWidgets, QtGui
from PyQt6.QtSvg import QSvgRenderer

from core import face_tools as ft
from core import processing as prc
from core.croppers import FolderCropper, PhotoCropper, MappingCropper, VideoCropper, DisplayCropper
from core.enums import FunctionType, Preset
from file_types import file_manager, FileCategory, SignatureChecker
from line_edits import NumberLineEdit, PathLineEdit, LineEditState, PathType
from ui import utils as ut
from .control_widget import UiCropControlWidget
from .crop_widget import UiCropWidget
from .enums import GuiIcon
from .folder_tab import UiFolderTabWidget
from .mapping_tab import UiMappingTabWidget
from .photo_tab import UiPhotoTabWidget
from .splash_screen import UiClickableSplashScreen
from .video_tab import UiVideoTabWidget

type TabWidget = Union[UiPhotoTabWidget, UiFolderTabWidget, UiMappingTabWidget, UiVideoTabWidget]


class UiMainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)

        # Start with a splash screen
        splash = UiClickableSplashScreen()
        splash.show_message("Loading face detection models...")
        QtWidgets.QApplication.processEvents()

        face_detection_tools = self.get_face_detection_tools(splash)

        # Single-threaded workers
        self.display_worker = DisplayCropper(face_detection_tools[0])
        self.photo_worker = PhotoCropper(face_detection_tools[0])
        self.video_worker = VideoCropper(face_detection_tools[0])

        # Multithreaded workers
        self.folder_worker = FolderCropper(face_detection_tools)
        self.mapping_worker = MappingCropper(face_detection_tools)

        # Create the central widget
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName(u"centralwidget")
        self.main_layout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        self.actionAbout_Face_Cropper = QtGui.QAction(self)
        self.actionUse_Mapping = QtGui.QAction(self)
        self.actionCrop_File = QtGui.QAction(self)
        self.actionCrop_Folder = QtGui.QAction(self)
        self.actionSquare = QtGui.QAction(self)
        self.actionGolden_Ratio = QtGui.QAction(self)
        self.action2_3_Ratio = QtGui.QAction(self)
        self.action3_4_Ratio = QtGui.QAction(self)
        self.action4_5_Ratio = QtGui.QAction(self)
        self.actionCrop_Video = QtGui.QAction(self)

        self.menubar = QtWidgets.QMenuBar(self)
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuTools = QtWidgets.QMenu(self.menubar)
        self.menuInfo = QtWidgets.QMenu(self.menubar)

        self.address_bar_widget = QtWidgets.QWidget()
        self.back_button = QtWidgets.QPushButton()
        self.forward_button = QtWidgets.QPushButton()
        self.refresh_button = QtWidgets.QPushButton()
        self.unified_address_bar = PathLineEdit(path_type=PathType.IMAGE)  # Default to image
        self.context_button = QtWidgets.QPushButton()
        self.secondary_input_container = QtWidgets.QWidget()
        self.secondary_input = PathLineEdit(path_type=PathType.TABLE)
        self.secondary_button = QtWidgets.QPushButton()
        self.destination_container = QtWidgets.QWidget()
        self.destination_label = QtWidgets.QLabel("Save to:")
        self.destination_input = PathLineEdit(path_type=PathType.FOLDER)
        self.destination_button = QtWidgets.QPushButton()
        self.info_button = QtWidgets.QPushButton()

        self.function_tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.photo_tab = QtWidgets.QWidget()
        self.verticalLayout_2 = ut.setup_vbox(u"verticalLayout_2", self.photo_tab)
        self.photo_tab_widget = UiPhotoTabWidget(self.photo_worker, u"photo_tab_widget", self.photo_tab)
        self.folder_tab = QtWidgets.QWidget()
        self.verticalLayout_3 = ut.setup_vbox(u"verticalLayout_3", self.folder_tab)
        self.folder_tab_widget = UiFolderTabWidget(self.folder_worker, u"folder_tab_widget", self.folder_tab)
        self.mapping_tab = QtWidgets.QWidget()
        self.verticalLayout_4 = ut.setup_vbox(u"verticalLayout_4", self.mapping_tab)
        self.mapping_tab_widget = UiMappingTabWidget(self.mapping_worker, u"mapping_tab_widget", self.mapping_tab)
        self.video_tab = QtWidgets.QWidget()
        self.verticalLayout_5 = ut.setup_vbox(u"verticalLayout_5", self.video_tab)
        self.video_tab_widget = UiVideoTabWidget(self.video_worker, u"video_tab_widget", self.video_tab)

        self.setObjectName(u"MainWindow")
        self.resize(1256, 652)
        icon = QtGui.QIcon()
        icon.addFile(GuiIcon.LOGO, QtCore.QSize(), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.setWindowIcon(icon)
        
        # Create the main menu
        self.create_main_menu()
        
        # Create an address bar (browser-like)
        self.create_address_bar()
        
        # Create tab widget (browser-like)
        self.create_tab_widgets()

        self.folder_worker.progressBars = [self.folder_tab_widget.progressBar]
        self.mapping_worker.progressBars = [self.mapping_tab_widget.progressBar]
        self.video_worker.progressBars = [self.video_tab_widget.progressBar, self.video_tab_widget.progressBar_2]
        
        # Create a status bar
        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName(u"statusbar")
        self.setStatusBar(self.statusbar)
        
        # Connect signals
        self.connect_widgets()
        
        self.setCentralWidget(self.centralwidget)
        self.retranslateUi()
        
        # Set the initial tab
        self.function_tabWidget.setCurrentIndex(0)

        self.initialize_clear_button_states()
        
        QtCore.QMetaObject.connectSlotsByName(self)

    def create_main_menu(self):
        """Create the main menu for the application"""
        # Create actions
        self.actionAbout_Face_Cropper.setObjectName(u"actionAbout_Face_Cropper")
        icon0 = QtGui.QIcon.fromTheme("help-browser")
        self.actionAbout_Face_Cropper.setIcon(icon0)

        self.actionUse_Mapping.setObjectName(u"actionUse_Mapping")
        icon1 = QtGui.QIcon()
        icon1.addFile(GuiIcon.EXCEL, QtCore.QSize(), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.actionUse_Mapping.setIcon(icon1)

        self.actionCrop_File.setObjectName(u"actionCrop_File")
        icon2 = QtGui.QIcon()
        icon2.addFile(GuiIcon.PICTURE, QtCore.QSize(), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.actionCrop_File.setIcon(icon2)

        self.actionCrop_Folder.setObjectName(u"actionCrop_Folder")
        icon3 = QtGui.QIcon()
        icon3.addFile(GuiIcon.FOLDER, QtCore.QSize(), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.actionCrop_Folder.setIcon(icon3)

        self.actionSquare.setObjectName(u"actionSquare")
        self.actionGolden_Ratio.setObjectName(u"actionGolden_Ratio")
        self.action2_3_Ratio.setObjectName(u"action2_3_Ratio")
        self.action3_4_Ratio.setObjectName(u"action3_4_Ratio")
        self.action4_5_Ratio.setObjectName(u"action4_5_Ratio")

        self.decorate_action(self.actionSquare, 1, 1, "#3498db")
        self.decorate_action(self.actionGolden_Ratio, 100, 162, "#f39c12")
        self.decorate_action(self.action2_3_Ratio, 2, 3, "#2ecc71")
        self.decorate_action(self.action3_4_Ratio, 3, 4, "#e74c3c")
        self.decorate_action(self.action4_5_Ratio, 4, 5, "#9b59b6")

        self.actionCrop_Video.setObjectName(u"actionCrop_Video")
        icon4 = QtGui.QIcon()
        icon4.addFile(GuiIcon.CLAPPERBOARD, QtCore.QSize(), QtGui.QIcon.Mode.Normal,
                      QtGui.QIcon.State.Off)
        self.actionCrop_Video.setIcon(icon4)
        
        # Create menu bar
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1256, 22))
        
        # Create menus
        self.menuFile.setObjectName(u"menuFile")
        self.menuTools.setObjectName(u"menuTools")
        self.menuInfo.setObjectName(u"menuInfo")
        
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
        address_bar_layout = QtWidgets.QHBoxLayout(self.address_bar_widget)
        address_bar_layout.setContentsMargins(10, 5, 10, 5)
        address_bar_layout.setSpacing(10)
        
        # Navigation buttons
        self.back_button.setIcon(QtGui.QIcon.fromTheme("go-previous"))
        self.back_button.setObjectName("backButton")
        self.back_button.setToolTip("Back")
        self.back_button.setFixedSize(36, 36)

        self.forward_button.setIcon(QtGui.QIcon.fromTheme("go-next"))
        self.forward_button.setObjectName("forwardButton")
        self.forward_button.setToolTip("Forward")
        self.forward_button.setFixedSize(36, 36)

        self.refresh_button.setIcon(QtGui.QIcon.fromTheme("view-refresh"))
        self.refresh_button.setObjectName("refreshButton")
        self.refresh_button.setToolTip("Refresh")
        self.refresh_button.setFixedSize(36, 36)
        
        # Unified address bar (dynamic path field)
        self.unified_address_bar.setObjectName("unifiedAddressBar")
        self.unified_address_bar.setPlaceholderText("Enter path...")
        
        # Context-aware open button with changing icon
        self.context_button.setObjectName("contextButton")
        self.context_button.setToolTip("Open File")
        self.context_button.setIcon(QtGui.QIcon(GuiIcon.PICTURE))  # Default icon
        self.context_button.setFixedSize(36, 36)
        
        # Secondary input for mapping tab (initially hidden)
        self.secondary_input_container.setObjectName("secondaryInputContainer")
        self.secondary_input_container.setVisible(False)  # Hidden by default
        
        secondary_layout = QtWidgets.QHBoxLayout(self.secondary_input_container)
        secondary_layout.setContentsMargins(0, 0, 0, 0)
        secondary_layout.setSpacing(5)

        self.secondary_input.setObjectName("secondaryInput")
        self.secondary_input.setPlaceholderText("Select table file...")

        self.secondary_button.setObjectName("secondaryButton")
        self.secondary_button.setIcon(QtGui.QIcon(GuiIcon.EXCEL))
        self.secondary_button.setFixedSize(36, 36)
        self.secondary_button.setToolTip("Open Table File")
        
        secondary_layout.addWidget(self.secondary_input)
        secondary_layout.addWidget(self.secondary_button)
        
        # Destination section (always visible)
        self.destination_container.setObjectName("destinationContainer")
        
        destination_layout = QtWidgets.QHBoxLayout(self.destination_container)
        destination_layout.setContentsMargins(0, 0, 0, 0)
        destination_layout.setSpacing(5)

        self.destination_label.setObjectName("destinationLabel")

        self.destination_input.setObjectName("destinationInput")
        self.destination_input.setPlaceholderText("Select destination folder...")

        self.destination_button.setObjectName("destinationButton")
        self.destination_button.setIcon(QtGui.QIcon(GuiIcon.FOLDER))
        self.destination_button.setFixedSize(36, 36)
        self.destination_button.setToolTip("Select Destination Folder")
        
        destination_layout.addWidget(self.destination_label)
        destination_layout.addWidget(self.destination_input)
        destination_layout.addWidget(self.destination_button)
        
        # Info button (on the right)
        self.info_button.setIcon(QtGui.QIcon.fromTheme("help-browser"))
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
        self.function_tabWidget.setObjectName(u"function_tabWidget")
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
        icon2 = QtGui.QIcon()
        icon2.addFile(GuiIcon.PICTURE, QtCore.QSize(), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)

        self.photo_tab.setObjectName(u"photo_tab")

        # Hide the redundant input fields that will be handled by unified address bar
        self.photo_tab_widget.horizontalLayout_2.setParent(None)  # Remove input layout
        self.photo_tab_widget.horizontalLayout_3.setParent(None)  # Remove destination layout

        self.verticalLayout_2.addWidget(self.photo_tab_widget)
        self.function_tabWidget.addTab(self.photo_tab, icon2, "")
        
    def create_folder_tab(self):
        """Create folder tab without redundant input fields"""
        icon3 = QtGui.QIcon()
        icon3.addFile(GuiIcon.FOLDER, QtCore.QSize(), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)

        self.folder_tab.setObjectName(u"folder_tab")

        # Modify the setup to remove redundant input fields
        # Access the verticalLayout_200 in page_1 to remove input and destination layouts
        input_layout = None
        destination_layout = None

        # Find the input and destination layouts
        for i in range(self.folder_tab_widget.verticalLayout_200.count()):
            item = self.folder_tab_widget.verticalLayout_200.itemAt(i)
            if isinstance(item, QtWidgets.QHBoxLayout) and hasattr(item, "objectName"):
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
        icon1 = QtGui.QIcon()
        icon1.addFile(GuiIcon.EXCEL, QtCore.QSize(), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)

        self.mapping_tab.setObjectName(u"mapping_tab")

        self.verticalLayout_4.addWidget(self.mapping_tab_widget)
        self.function_tabWidget.addTab(self.mapping_tab, icon1, "")
        
    def create_video_tab(self):
        """Create video tab without redundant input fields"""
        icon4 = QtGui.QIcon()
        icon4.addFile(GuiIcon.CLAPPERBOARD, QtCore.QSize(), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)

        self.video_tab.setObjectName(u"video_tab")

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
                    w.imageWidget.setImage(img) if _ft == ft_expected else None
            )

        self.verticalLayout_5.addWidget(self.video_tab_widget)
        self.function_tabWidget.addTab(self.video_tab, icon4, "")

    def decorate_action(self,action: QtGui.QAction, width: float, height: float, color: str):
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

        svg_bytes = QtCore.QByteArray(svg_data.encode())
        renderer = QSvgRenderer(svg_bytes)

        # Create pixmap at multiple sizes for better scaling
        icon = QtGui.QIcon()
        for size in [16, 24, 32, 48, 64]:
            pixmap = QtGui.QPixmap(size, size)
            pixmap.fill(QtCore.Qt.GlobalColor.transparent)
            painter = QtGui.QPainter(pixmap)
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
                self.context_button.setIcon(QtGui.QIcon(GuiIcon.PICTURE))
                self.context_button.setToolTip("Open Image")
                
                # Hide secondary input (not needed for photo tab)
                self.secondary_input_container.setVisible(False)
                
                # Get the current path from tab widget if it exists
                self.unified_address_bar.blockSignals(True)
                self.unified_address_bar.setText(self.photo_tab_widget.input_path)
                self.unified_address_bar.blockSignals(False)
                
                # Get a destination path
                self.destination_input.blockSignals(True)
                self.destination_input.setText(self.photo_tab_widget.destination_path)
                self.destination_input.blockSignals(False)
            
            case FunctionType.FOLDER:
                self.unified_address_bar.set_path_type(PathType.FOLDER)
                self.unified_address_bar.setPlaceholderText("Enter folder path...")
                self.context_button.setIcon(QtGui.QIcon(GuiIcon.FOLDER))
                self.context_button.setToolTip("Select Folder")
                
                # Hide secondary input (not needed for folder tab)
                self.secondary_input_container.setVisible(False)
                
                # Get the current path from tab widget if it exists
                self.unified_address_bar.blockSignals(True)
                self.unified_address_bar.setText(self.folder_tab_widget.input_path)
                self.unified_address_bar.blockSignals(False)
                
                # Get a destination path
                self.destination_input.blockSignals(True)
                self.destination_input.setText(self.folder_tab_widget.destination_path)
                self.destination_input.blockSignals(False)
            
            case FunctionType.MAPPING:
                self.unified_address_bar.set_path_type(PathType.FOLDER)
                self.unified_address_bar.setPlaceholderText("Enter source folder path...")
                self.context_button.setIcon(QtGui.QIcon(GuiIcon.FOLDER))
                self.context_button.setToolTip("Select Source Folder")
                
                # Show and configure secondary input for mapping tab
                self.secondary_input_container.setVisible(True)
                self.secondary_input.set_path_type(PathType.TABLE)
                self.secondary_input.setPlaceholderText("Enter table file path...")
                
                # Get current paths from tab widget if they exist
                self.unified_address_bar.blockSignals(True)
                self.unified_address_bar.setText(self.mapping_tab_widget.input_path)
                self.unified_address_bar.blockSignals(False)
                
                self.secondary_input.blockSignals(True)
                self.secondary_input.setText(self.mapping_tab_widget.table_path)
                self.secondary_input.blockSignals(False)
                
                # Get a destination path
                self.destination_input.blockSignals(True)
                self.destination_input.setText(self.mapping_tab_widget.destination_path)
                self.destination_input.blockSignals(False)
            
            case FunctionType.VIDEO:
                self.unified_address_bar.set_path_type(PathType.VIDEO)
                self.unified_address_bar.setPlaceholderText("Enter video file path...")
                self.context_button.setIcon(QtGui.QIcon(GuiIcon.CLAPPERBOARD))
                self.context_button.setToolTip("Open Video")
                
                # Hide secondary input (not needed for video tab)
                self.secondary_input_container.setVisible(False)
                
                # Get the current path from tab widget if it exists
                self.unified_address_bar.blockSignals(True)
                self.unified_address_bar.setText(self.video_tab_widget.input_path)
                self.unified_address_bar.blockSignals(False)
                
                # Get a destination path
                self.destination_input.blockSignals(True)
                self.destination_input.setText(self.video_tab_widget.destination_path)
                self.destination_input.blockSignals(False)
        
        # Force update of clear button visibility after changing tab
        # This is important since we block signals during setText operations
        self.unified_address_bar.update_clear_button(self.unified_address_bar.text())
        self.destination_input.update_clear_button(self.destination_input.text())
        if self.secondary_input_container.isVisible():
            self.secondary_input.update_clear_button(self.secondary_input.text())


    # retranslateUi
    def retranslateUi(self):
        self.setWindowTitle(QtCore.QCoreApplication.translate("self", u"Autocrop", None))
        self.actionAbout_Face_Cropper.setText(QtCore.QCoreApplication.translate("self", u"About Autocrop", None))
        self.actionUse_Mapping.setText(QtCore.QCoreApplication.translate("self", u"Use Mapping", None))
        self.actionCrop_File.setText(QtCore.QCoreApplication.translate("self", u"Crop File", None))
        self.actionCrop_Folder.setText(QtCore.QCoreApplication.translate("self", u"Crop Folder", None))
        self.actionSquare.setText(QtCore.QCoreApplication.translate("self", u"Square", None))
        self.actionGolden_Ratio.setText(QtCore.QCoreApplication.translate("self", u"Golden Ratio", None))
        self.action2_3_Ratio.setText(QtCore.QCoreApplication.translate("self", u"2:3 Ratio", None))
        self.action3_4_Ratio.setText(QtCore.QCoreApplication.translate("self", u"3:4 Ratio", None))
        self.action4_5_Ratio.setText(QtCore.QCoreApplication.translate("self", u"4:5 Ratio", None))
        self.actionCrop_Video.setText(QtCore.QCoreApplication.translate("self", u"Crop Video", None))
        self.function_tabWidget.setTabText(self.function_tabWidget.indexOf(self.photo_tab),
                                          QtCore.QCoreApplication.translate("self", u"Photo Crop", None))
        self.function_tabWidget.setTabText(self.function_tabWidget.indexOf(self.folder_tab),
                                          QtCore.QCoreApplication.translate("self", u"Folder Crop", None))
        self.function_tabWidget.setTabText(self.function_tabWidget.indexOf(self.mapping_tab),
                                          QtCore.QCoreApplication.translate("self", u"Mapping Crop", None))
        self.function_tabWidget.setTabText(self.function_tabWidget.indexOf(self.video_tab),
                                          QtCore.QCoreApplication.translate("self", u"Video Crop", None))
        self.menuFile.setTitle(QtCore.QCoreApplication.translate("self", u"Presets", None))
        self.menuTools.setTitle(QtCore.QCoreApplication.translate("self", u"Tools", None))
        self.menuInfo.setTitle(QtCore.QCoreApplication.translate("self", u"Info", None))
        
        # Address bar elements
        self.unified_address_bar.setPlaceholderText(QtCore.QCoreApplication.translate("self", 
                                            u"Enter file path or drag and drop files here", None))
        self.back_button.setToolTip(QtCore.QCoreApplication.translate("self", u"Back", None))
        self.forward_button.setToolTip(QtCore.QCoreApplication.translate("self", u"Forward", None))
        self.refresh_button.setToolTip(QtCore.QCoreApplication.translate("self", u"Refresh Preview", None))
        self.info_button.setToolTip(QtCore.QCoreApplication.translate("self", u"Settings", None))
        
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
        
        # Connect secondary input for mapping tab
        self.secondary_input.textChanged.connect(self.secondary_input_changed)
        
        # Connect destination input
        self.destination_input.textChanged.connect(self.destination_input_changed)
        
        # Connect context-aware buttons
        self.context_button.clicked.connect(self.handle_context_button)
        self.secondary_button.clicked.connect(self.handle_secondary_button)
        self.destination_button.clicked.connect(self.handle_destination_button)

    def unified_address_changed(self, text):
        """Handle changes to the unified address bar"""
        
        # Update the appropriate input field in the current tab
        match self.function_tabWidget.currentIndex():
            case FunctionType.PHOTO:
                self.photo_tab_widget.input_path = text
            case FunctionType.FOLDER:
                self.folder_tab_widget.input_path = text
            case FunctionType.MAPPING:
                self.mapping_tab_widget.input_path = text
            case FunctionType.VIDEO:
                self.video_tab_widget.input_path = text

    def secondary_input_changed(self, text):
        """Handle changes to the secondary input (only for mapping tab)"""
        match self.function_tabWidget.currentIndex():
            case FunctionType.MAPPING:
                self.mapping_tab_widget.table_path = text
            case _:
                pass

    def destination_input_changed(self, text):
        """Handle changes to the destination input"""
        match self.function_tabWidget.currentIndex():
            case FunctionType.PHOTO:
                self.photo_tab_widget.destination_path = text
            case FunctionType.FOLDER:
                self.folder_tab_widget.destination_path = text
            case FunctionType.MAPPING:
                self.mapping_tab_widget.destination_path = text
            case FunctionType.VIDEO:
                self.video_tab_widget.destination_path = text

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

    def open_file_dialog(self, path_type: PathType) -> None:
        """Securely open a file dialog and validate the selected path"""
        try:
            # Use QFileDialog with security considerations
            options = QtWidgets.QFileDialog.Option.ReadOnly  # Prevents modification during selection
            
            category = (
                FileCategory.PHOTO if path_type == PathType.IMAGE else
                FileCategory.VIDEO if path_type == PathType.VIDEO else
                FileCategory.TABLE
            )
            # Get the appropriate default directory and filter based on the path type
            default_dir = file_manager.get_default_directory(category).as_posix()
            
            filter_string = file_manager.get_filter_string(category)
            
            # Open the dialogue with the appropriate title
            title = f"Open {'Image' if path_type == PathType.IMAGE else 'Video' if path_type == PathType.VIDEO else 'Table'}"
            
            # Use QFileDialog.getOpenFileName which is more secure than the older methods
            f_name, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, title, default_dir, filter_string, options=options
            )
            
            # Validate the selected path
            if f_name := ut.sanitize_path(f_name):
                # Create a Path object for additional validation
                path_obj = Path(f_name).resolve()
                
                # Verify the file exists and is of the expected type
                if not path_obj.is_file():
                    ut.show_error_box("Selected path is not a valid file")
                    return
                    
                # Verify file is of expected type
                expected_category = (FileCategory.PHOTO if path_type == PathType.IMAGE else
                                FileCategory.VIDEO if path_type == PathType.VIDEO else
                                FileCategory.TABLE)
                
                if not file_manager.is_valid_type(path_obj, expected_category):
                    ut.show_error_box(f"Selected file is not a valid {expected_category.name.lower()}")
                    return
                
        # Verify content matches the expected type
                if not SignatureChecker.verify_file_type(path_obj, category):
                    ut.show_error_box("File content doesn't match its extension. The file may be corrupted or modified.")
                    return
                
                # Update the appropriate input path based on the file type
                if path_type == PathType.IMAGE:
                    self.input_path = f_name
                    # Update the unified address bar if this is the active tab
                    main_window = self.parent().parent().parent()
                    if main_window.function_tabWidget.currentIndex() == FunctionType.PHOTO:
                        main_window.unified_address_bar.setText(f_name)
                elif path_type == PathType.VIDEO:
                    self.input_path = f_name
                    # Update relevant UI for video
                    self.player.setSource(QtCore.QUrl.fromLocalFile(f_name))
                    self.reset_video_widgets()
                elif path_type == PathType.TABLE:
                    self.table_path = f_name
                    # Process table data
                    data = prc.load_table(path_obj)
                    self.process_data(data)
        
        except Exception as e:
            # Log error internally without exposing details
            ut.show_error_box(f"An error occurred opening the file\n{e}")

    def open_folder_dialog(self, target_input: PathLineEdit) -> None:
        """Securely open a folder dialogue and validate the selected path"""
        try:
            # Use QFileDialog with options that improve security
            options = QtWidgets.QFileDialog.Option.ShowDirsOnly | QtWidgets.QFileDialog.Option.DontResolveSymlinks
            
            # Get appropriate default directory
            default_dir = file_manager.get_default_directory(FileCategory.PHOTO).as_posix()
            
            # Open the dialogue with the appropriate title
            f_name = QtWidgets.QFileDialog.getExistingDirectory(
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
                target_input.setText(str(path_obj))
                
                # If this is a source directory, refresh any view that depends on it
                if target_input == self.unified_address_bar and self.function_tabWidget.currentIndex() == FunctionType.FOLDER:
                    self.folder_tab_widget.load_data()
        
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
    
    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
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
                QtWidgets.QApplication.processEvents()
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

    def adjust_ui(self, app: QtWidgets.QApplication):
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
                control.widthLineEdit.setText(u'1000')
                control.heightLineEdit.setText(u'1000')

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
    def dragEnterEvent(self, a0: Optional[QtGui.QDragEnterEvent]) -> None:
        """Handle drag enter events for browser-like drag and drop"""
        try:
            assert isinstance(a0, QtGui.QDragEnterEvent)
        except AssertionError:
            return
        ut.check_mime_data(a0)
        
        # Show a status message
        self.statusbar.showMessage("Drop files here to open", 2000)

    def dragMoveEvent(self, a0: Optional[QtGui.QDragMoveEvent]) -> None:
        """Handle drag move events"""
        try:
            assert isinstance(a0, QtGui.QDragMoveEvent)
        except AssertionError:
            return
        ut.check_mime_data(a0)

    def dropEvent(self, a0: Optional[QtGui.QDropEvent]) -> None:
        """
        Handle drop events with enhanced security.
        """
        try:
            assert isinstance(a0, QtGui.QDropEvent)
        except AssertionError:
            return

        if (mime_data := a0.mimeData()) is None:
            return

        if not mime_data.hasUrls():
            a0.ignore()
            return

        a0.setDropAction(QtCore.Qt.DropAction.CopyAction)
        
        # Get the dropped URL and convert to a local file path
        url = mime_data.urls()[0]
        if not url.isLocalFile():
            ut.show_error_box("Only local files can be dropped")
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
            print(f"Error processing dropped file: {e}")
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
            
            if has_table_files:
                # Handle as mapping tab
                self.function_tabWidget.setCurrentIndex(FunctionType.MAPPING)
                self.mapping_tab_widget.input_path = str(dir_path)
                self.unified_address_bar.setText(str(dir_path))
                
                # Update display
                self.display_worker.current_paths[FunctionType.MAPPING] = None
                self.display_worker.crop(FunctionType.MAPPING)
            else:
                # Handle as folder tab
                self.function_tabWidget.setCurrentIndex(FunctionType.FOLDER)
                self.folder_tab_widget.input_path = str(dir_path)
                self.unified_address_bar.setText(str(dir_path))
                
                # Update display
                self.display_worker.current_paths[FunctionType.FOLDER] = None
                self.folder_tab_widget.load_data()
                self.display_worker.crop(FunctionType.FOLDER)
        except Exception as e:
            # Log error internally without exposing details
            print(f"Error handling dropped directory: {e}")
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
            # Determine the file type and handle accordingly
            if ((file_manager.is_valid_type(file_path, FileCategory.PHOTO) or
                file_manager.is_valid_type(file_path, FileCategory.RAW) or
                file_manager.is_valid_type(file_path, FileCategory.TIFF)) and
                SignatureChecker.verify_file_type(file_path, FileCategory.PHOTO)):
                
                # Handle with the photo tab
                self.function_tabWidget.setCurrentIndex(FunctionType.PHOTO)
                self.photo_tab_widget.input_path = str(file_path)
                self.unified_address_bar.setText(str(file_path))
                self.display_worker.crop(FunctionType.PHOTO)
                
            elif (file_manager.is_valid_type(file_path, FileCategory.VIDEO) and
                  SignatureChecker.verify_file_type(file_path, FileCategory.VIDEO)):
                # Handle with the video tab
                self.function_tabWidget.setCurrentIndex(FunctionType.VIDEO)
                self.video_tab_widget.input_path = str(file_path)
                self.unified_address_bar.setText(str(file_path))
                self.video_tab_widget.open_dropped_video()
                
            elif (file_manager.is_valid_type(file_path, FileCategory.TABLE) and
                  SignatureChecker.verify_file_type(file_path, FileCategory.TABLE)):
                # Handle with the mapping tab
                self.function_tabWidget.setCurrentIndex(FunctionType.MAPPING)
                self.mapping_tab_widget.table_path = str(file_path)
                self.secondary_input.setText(str(file_path))
                
                # Process the table data
                data = prc.load_table(file_path)
                self.mapping_tab_widget.process_data(data)
                
            else:
                ut.show_error_box(f"Unsupported file type: {file_path.suffix}")
        except Exception as e:
            # Log error internally without exposing details
            print(f"Error handling dropped file: {e}")
            ut.show_error_box("An error occurred processing the file")
        
    def handle_path_main(self, file_path: Path) -> None:
        """
        Handles a file path by checking the file extensions in the directory, validating the mapping and folder tabs, and calling the handle_path method with the appropriate arguments.
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
        Handles a file path by setting the function tab widget to the specified tab index, updating the line edit with the file path, and handling the selection state of the tabs.
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
        Handles an image file by validating the file path and calling the handle_path method with the appropriate arguments.
        """
        try:
            assert isinstance(self.photo_tab_widget, UiPhotoTabWidget)
        except AssertionError:
            return
        self.handle_path(file_path, FunctionType.PHOTO, self.unified_address_bar)

    def handle_video_file(self, file_path: Path) -> None:
        """
        Handles a video file by setting the function tab widget to the video tab, validating the file path, and configuring the video player.
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
        self.video_tab_widget.mediacontrolWidget_1.playButton.setIcon(QtGui.QIcon(GuiIcon.MULTIMEDIA_PLAY))
        self.video_tab_widget.mediacontrolWidget_2.playButton.setEnabled(True)
        self.video_tab_widget.mediacontrolWidget_2.playButton.setIcon(QtGui.QIcon(GuiIcon.MULTIMEDIA_PLAY))
        self.video_tab_widget.open_dropped_video()

    def handle_table_file(self, file_path: Path) -> None:
        """
        Handles a table file by setting the function tab widget to the mapping tab, validating the file path, and opening the table.
        """
        self.function_tabWidget.setCurrentIndex(FunctionType.MAPPING)
        self.mapping_tab_widget.table_path = file_path.as_posix()
        data = prc.load_table(file_path)
        self.mapping_tab_widget.process_data(data)

    @staticmethod
    def all_filled(*line_edits: Union[PathLineEdit, NumberLineEdit, QtWidgets.QComboBox]) -> bool:
        x = all(edit.state == LineEditState.VALID_INPUT
                for edit in line_edits if isinstance(edit, (PathLineEdit, NumberLineEdit)))
        y = all(edit.currentText() for edit in line_edits if isinstance(edit, QtWidgets.QComboBox))
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
            case tab_widget if isinstance(tab_widget, (UiPhotoTabWidget, UiFolderTabWidget)):
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
