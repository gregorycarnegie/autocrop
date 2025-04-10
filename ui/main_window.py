from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Union, Optional

from PyQt6 import QtCore, QtWidgets, QtGui

from core import face_tools as ft
from core import processing as prc
from core.croppers import FolderCropper, PhotoCropper, MappingCropper, VideoCropper, DisplayCropper
from core.enums import FunctionType, Preset
from file_types import file_manager, FileCategory
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

        # Single-threaded
        self.display_worker = DisplayCropper(face_detection_tools[0])
        self.photo_worker = PhotoCropper(face_detection_tools[0])
        self.video_worker = VideoCropper(face_detection_tools[0])
        
        # Multi-threaded
        self.folder_worker = FolderCropper(face_detection_tools)
        self.mapping_worker = MappingCropper(face_detection_tools)

        self.setObjectName(u"MainWindow")
        self.resize(1256, 652)
        icon = QtGui.QIcon()
        icon.addFile(GuiIcon.LOGO, QtCore.QSize(), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.setWindowIcon(icon)
        
        # Create main menu
        self.create_main_menu()
        
        # Create central widget
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName(u"centralwidget")
        self.main_layout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # Create address bar (browser-like)
        self.create_address_bar()
        
        # Create tab widget (browser-like)
        self.create_tab_widget()
        
        # Create status bar
        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName(u"statusbar")
        self.setStatusBar(self.statusbar)
        
        # Connect signals
        self.connect_widgets()
        
        self.setCentralWidget(self.centralwidget)
        self.retranslateUi()
        
        # Set initial tab
        self.function_tabWidget.setCurrentIndex(0)
        
        QtCore.QMetaObject.connectSlotsByName(self)

    def create_main_menu(self):
        """Create the main menu for the application"""
        # Create actions
        self.actionAbout_Face_Cropper = QtGui.QAction(self)
        self.actionAbout_Face_Cropper.setObjectName(u"actionAbout_Face_Cropper")
        
        self.actionUse_Mapping = QtGui.QAction(self)
        self.actionUse_Mapping.setObjectName(u"actionUse_Mapping")
        icon1 = QtGui.QIcon()
        icon1.addFile(GuiIcon.EXCEL, QtCore.QSize(), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.actionUse_Mapping.setIcon(icon1)
        
        self.actionCrop_File = QtGui.QAction(self)
        self.actionCrop_File.setObjectName(u"actionCrop_File")
        icon2 = QtGui.QIcon()
        icon2.addFile(GuiIcon.PICTURE, QtCore.QSize(), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.actionCrop_File.setIcon(icon2)
        
        self.actionCrop_Folder = QtGui.QAction(self)
        self.actionCrop_Folder.setObjectName(u"actionCrop_Folder")
        icon3 = QtGui.QIcon()
        icon3.addFile(GuiIcon.FOLDER, QtCore.QSize(), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.actionCrop_Folder.setIcon(icon3)
        
        self.actionSquare = QtGui.QAction(self)
        self.actionSquare.setObjectName(u"actionSquare")
        
        self.actionGolden_Ratio = QtGui.QAction(self)
        self.actionGolden_Ratio.setObjectName(u"actionGolden_Ratio")
        
        self.action2_3_Ratio = QtGui.QAction(self)
        self.action2_3_Ratio.setObjectName(u"action2_3_Ratio")
        
        self.action3_4_Ratio = QtGui.QAction(self)
        self.action3_4_Ratio.setObjectName(u"action3_4_Ratio")
        
        self.action4_5_Ratio = QtGui.QAction(self)
        self.action4_5_Ratio.setObjectName(u"action4_5_Ratio")
        
        self.actionCrop_Video = QtGui.QAction(self)
        self.actionCrop_Video.setObjectName(u"actionCrop_Video")
        icon4 = QtGui.QIcon()
        icon4.addFile(GuiIcon.CLAPPERBOARD, QtCore.QSize(), QtGui.QIcon.Mode.Normal,
                      QtGui.QIcon.State.Off)
        self.actionCrop_Video.setIcon(icon4)
        
        # Create menu bar
        self.menubar = QtWidgets.QMenuBar(self)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1256, 22))
        
        # Create menus
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName(u"menuFile")
        
        self.menuTools = QtWidgets.QMenu(self.menubar)
        self.menuTools.setObjectName(u"menuTools")
        
        self.menuInfo = QtWidgets.QMenu(self.menubar)
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
        """Create a browser-like address bar layout"""
        # Address bar container
        self.address_bar_widget = QtWidgets.QWidget()
        self.address_bar_widget.setObjectName("addressBarWidget")
        self.address_bar_widget.setMinimumHeight(48)
        self.address_bar_widget.setMaximumHeight(48)
        
        # Address bar layout
        address_bar_layout = QtWidgets.QHBoxLayout(self.address_bar_widget)
        address_bar_layout.setContentsMargins(10, 5, 10, 5)
        address_bar_layout.setSpacing(10)
        
        # Navigation buttons (like browser back/forward)
        self.back_button = QtWidgets.QPushButton()
        self.back_button.setIcon(QtGui.QIcon.fromTheme("go-previous"))
        self.back_button.setObjectName("backButton")
        self.back_button.setToolTip("Back")
        self.back_button.setFixedSize(36, 36)
        
        self.forward_button = QtWidgets.QPushButton()
        self.forward_button.setIcon(QtGui.QIcon.fromTheme("go-next"))
        self.forward_button.setObjectName("forwardButton")
        self.forward_button.setToolTip("Forward")
        self.forward_button.setFixedSize(36, 36)
        
        self.refresh_button = QtWidgets.QPushButton()
        self.refresh_button.setIcon(QtGui.QIcon.fromTheme("view-refresh"))
        self.refresh_button.setObjectName("refreshButton")
        self.refresh_button.setToolTip("Refresh")
        self.refresh_button.setFixedSize(36, 36)
        
        # Address bar (unified search/path field like in a browser)
        self.address_bar = PathLineEdit(path_type=PathType.IMAGE)
        self.address_bar.setObjectName("addressBar")
        self.address_bar.setPlaceholderText("Enter file path or drag and drop files here")
        
        # Settings button (like browser's menu button)
        self.settings_button = QtWidgets.QPushButton()
        self.settings_button.setIcon(QtGui.QIcon.fromTheme("preferences-system"))
        self.settings_button.setObjectName("settingsButton")
        self.settings_button.setToolTip("Settings")
        self.settings_button.setFixedSize(36, 36)
        
        # Add widgets to layout
        address_bar_layout.addWidget(self.back_button)
        address_bar_layout.addWidget(self.forward_button)
        address_bar_layout.addWidget(self.refresh_button)
        address_bar_layout.addWidget(self.address_bar)
        address_bar_layout.addWidget(self.settings_button)
        
        # Add to main layout
        self.main_layout.addWidget(self.address_bar_widget)
        
    def create_tab_widget(self):
        """Create browser-like tab widget layout"""
        # Create the tab widget
        self.function_tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.function_tabWidget.setObjectName(u"function_tabWidget")
        self.function_tabWidget.setMovable(True)
        
        # Add tabs
        self.create_photo_tab()
        self.create_folder_tab()
        self.create_mapping_tab()
        self.create_video_tab()
        
        # Add tab widget to main layout
        self.main_layout.addWidget(self.function_tabWidget)
        
    def create_photo_tab(self):
        """Create photo tab"""
        icon2 = QtGui.QIcon()
        icon2.addFile(GuiIcon.PICTURE, QtCore.QSize(), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        
        self.photo_tab = QtWidgets.QWidget()
        self.photo_tab.setObjectName(u"photo_tab")
        self.verticalLayout_2 = ut.setup_vbox(u"verticalLayout_2", self.photo_tab)
        self.photo_tab_widget = UiPhotoTabWidget(self.photo_worker, u"photo_tab_widget", self.photo_tab)
        self.verticalLayout_2.addWidget(self.photo_tab_widget)
        self.function_tabWidget.addTab(self.photo_tab, icon2, "")
        
    def create_folder_tab(self):
        """Create folder tab"""
        icon3 = QtGui.QIcon()
        icon3.addFile(GuiIcon.FOLDER, QtCore.QSize(), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        
        self.folder_tab = QtWidgets.QWidget()
        self.folder_tab.setObjectName(u"folder_tab")
        self.verticalLayout_3 = ut.setup_vbox(u"verticalLayout_3", self.folder_tab)
        self.folder_tab_widget = UiFolderTabWidget(self.folder_worker, u"folder_tab_widget", self.folder_tab)
        self.verticalLayout_3.addWidget(self.folder_tab_widget)
        self.function_tabWidget.addTab(self.folder_tab, icon3, "")
        
    def create_mapping_tab(self):
        """Create mapping tab"""
        icon1 = QtGui.QIcon()
        icon1.addFile(GuiIcon.EXCEL, QtCore.QSize(), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        
        self.mapping_tab = QtWidgets.QWidget()
        self.mapping_tab.setObjectName(u"mapping_tab")
        self.verticalLayout_4 = ut.setup_vbox(u"verticalLayout_4", self.mapping_tab)
        self.mapping_tab_widget = UiMappingTabWidget(self.mapping_worker, u"mapping_tab_widget", self.mapping_tab)
        self.verticalLayout_4.addWidget(self.mapping_tab_widget)
        self.function_tabWidget.addTab(self.mapping_tab, icon1, "")
        
    def create_video_tab(self):
        """Create video tab"""
        icon4 = QtGui.QIcon()
        icon4.addFile(GuiIcon.CLAPPERBOARD, QtCore.QSize(), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        
        self.video_tab = QtWidgets.QWidget()
        self.video_tab.setObjectName(u"video_tab")
        self.verticalLayout_5 = ut.setup_vbox(u"verticalLayout_5", self.video_tab)
        self.video_tab_widget = UiVideoTabWidget(self.video_worker, u"video_tab_widget", self.video_tab)
        
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
        
    # retranslateUi
    def retranslateUi(self):
        self.setWindowTitle(QtCore.QCoreApplication.translate("self", u"Autocrop Browser", None))
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
        self.address_bar.setPlaceholderText(QtCore.QCoreApplication.translate("self", 
                                            u"Enter file path or drag and drop files here", None))
        self.back_button.setToolTip(QtCore.QCoreApplication.translate("self", u"Back", None))
        self.forward_button.setToolTip(QtCore.QCoreApplication.translate("self", u"Forward", None))
        self.refresh_button.setToolTip(QtCore.QCoreApplication.translate("self", u"Refresh Preview", None))
        self.settings_button.setToolTip(QtCore.QCoreApplication.translate("self", u"Settings", None))
        
    def connect_widgets(self):
        """Connect widget signals to handlers"""
        # CONNECTIONS
        self.connect_combo_boxes(self.mapping_tab_widget)

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
        self.settings_button.clicked.connect(self.show_settings)
        
        # Connect address bar to current tab's input field
        self.address_bar.textChanged.connect(self.update_current_tab_path)
        
        # Connect tab change events
        self.function_tabWidget.currentChanged.connect(self.check_tab_selection)
        self.function_tabWidget.currentChanged.connect(self.update_address_bar)
        self.function_tabWidget.currentChanged.connect(self.video_tab_widget.player.pause)

        # Error handling connections
        self.folder_worker.error.connect(self.folder_tab_widget.disable_buttons)
        self.photo_worker.error.connect(self.photo_tab_widget.disable_buttons)
        self.mapping_worker.error.connect(self.mapping_tab_widget.disable_buttons)
        self.video_worker.error.connect(self.video_tab_widget.disable_buttons)

        # Focus connections
        self.actionCrop_File.triggered.connect(self.function_tabWidget.setFocus)
        self.actionCrop_Folder.triggered.connect(self.function_tabWidget.setFocus)
        self.actionCrop_Video.triggered.connect(self.function_tabWidget.setFocus)
        self.actionUse_Mapping.triggered.connect(self.function_tabWidget.setFocus)
        
    # Browser-style navigation methods
    def navigate_back(self):
        """Navigate to previous tab"""
        current_index = self.function_tabWidget.currentIndex()
        if current_index > 0:
            self.function_tabWidget.setCurrentIndex(current_index - 1)
    
    def navigate_forward(self):
        """Navigate to next tab"""
        current_index = self.function_tabWidget.currentIndex()
        if current_index < self.function_tabWidget.count() - 1:
            self.function_tabWidget.setCurrentIndex(current_index + 1)
    
    def refresh_current_view(self):
        """Refresh the current tab's view"""
        # Get the current tab widget
        current_index = self.function_tabWidget.currentIndex()
        tab_widget = None
        
        if current_index == FunctionType.PHOTO:
            tab_widget = self.photo_tab_widget
            # Refresh photo preview
            if tab_widget.inputLineEdit.text() and tab_widget.inputLineEdit.state == LineEditState.VALID_INPUT:
                self.display_worker.crop(FunctionType.PHOTO)
        elif current_index == FunctionType.FOLDER:
            tab_widget = self.folder_tab_widget
            # Refresh folder view
            tab_widget.load_data()
        elif current_index == FunctionType.MAPPING:
            tab_widget = self.mapping_tab_widget
            # Refresh mapping preview
            if tab_widget.tableLineEdit.text() and tab_widget.tableLineEdit.state == LineEditState.VALID_INPUT:
                if file_path := Path(tab_widget.tableLineEdit.text()):
                    data = prc.open_table(file_path)
                    tab_widget.process_data(data)
        elif current_index == FunctionType.VIDEO:
            tab_widget = self.video_tab_widget
            # Refresh video preview
            self.display_worker.crop(FunctionType.VIDEO)
            
        # Update status bar with message
        self.statusbar.showMessage("View refreshed", 2000)
    
    def show_settings(self):
        """Show settings dialog"""
        # For now, just show the About dialog as a placeholder
        ut.load_about_form()
    
    def update_address_bar(self):
        """Update address bar with current tab's path"""
        current_index = self.function_tabWidget.currentIndex()
        path = ""
        
        # Get the appropriate path based on the current tab
        if current_index == FunctionType.PHOTO:
            path = self.photo_tab_widget.inputLineEdit.text()
        elif current_index == FunctionType.FOLDER:
            path = self.folder_tab_widget.inputLineEdit.text()
        elif current_index == FunctionType.MAPPING:
            path = self.mapping_tab_widget.tableLineEdit.text()
        elif current_index == FunctionType.VIDEO:
            path = self.video_tab_widget.inputLineEdit.text()
            
        # Update address bar without triggering text changed event
        self.address_bar.blockSignals(True)
        self.address_bar.setText(path)
        self.address_bar.blockSignals(False)
    
    def update_current_tab_path(self):
        """Update current tab's path with address bar text"""
        current_index = self.function_tabWidget.currentIndex()
        path = self.address_bar.text()
        
        # Update the appropriate input field based on the current tab
        if current_index == FunctionType.PHOTO:
            self.photo_tab_widget.inputLineEdit.setText(path)
        elif current_index == FunctionType.FOLDER:
            self.folder_tab_widget.inputLineEdit.setText(path)
        elif current_index == FunctionType.MAPPING:
            self.mapping_tab_widget.tableLineEdit.setText(path)
        elif current_index == FunctionType.VIDEO:
            self.video_tab_widget.inputLineEdit.setText(path)
    
    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """
        Handle window close event by performing proper cleanup.
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
            w.inputLineEdit.text(),
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
        return w.inputLineEdit.text()

    @staticmethod
    def connect_widget_signals(widget: TabWidget, crop_method: Callable) -> None:
        """Connect signals with minimal overhead"""
        signals = (
            widget.inputLineEdit.textChanged,
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
        size = screen.size()
        width, height = size.width(), size.height()
        base_font_size = 10  # Reduced from 12 for more browser-like appearance

        # Adjust based on screen resolution
        if width >= 3_840:
            self.resize(width >> 1, height >> 1)
        else:
            self.resize(3 * width >> 2, 3 * height >> 2)

        if height > 1_080:
            scale_factor = height / 1_080
            base_font_size = int(base_font_size * scale_factor)

        font = app.font()
        font.setPointSize(base_font_size)
        app.setFont(font)
        
        # Set appropriate tab sizes
        self.function_tabWidget.setTabsClosable(True)  # Add closable tabs like a browser
        self.function_tabWidget.tabCloseRequested.connect(self.handle_tab_close)
        
        # Initialize navigation button states
        self.update_navigation_button_states()

    def handle_tab_close(self, index: int):
        """
        Handle tab close button clicks.
        For a browser-like experience, we don't actually close tabs but reset their state.
        """
        # Don't actually close the tab, just reset its state
        current_tab = self.function_tabWidget.widget(index)
        
        # Show a message in the status bar
        self.statusbar.showMessage(f"Tab {self.function_tabWidget.tabText(index)} reset", 2000)
        
        # Reset the appropriate tab widget
        if index == FunctionType.PHOTO:
            self.photo_tab_widget.inputLineEdit.clear()
            self.photo_tab_widget.destinationLineEdit.clear()
        elif index == FunctionType.FOLDER:
            self.folder_tab_widget.inputLineEdit.clear()
            self.folder_tab_widget.destinationLineEdit.clear()
        elif index == FunctionType.MAPPING:
            self.mapping_tab_widget.inputLineEdit.clear()
            self.mapping_tab_widget.tableLineEdit.clear()
            self.mapping_tab_widget.destinationLineEdit.clear()
        elif index == FunctionType.VIDEO:
            self.video_tab_widget.inputLineEdit.clear()
            self.video_tab_widget.destinationLineEdit.clear()
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
        
        # Update address bar
        self.update_address_bar()
        
        # Process tab selection as before
        match self.function_tabWidget.currentIndex():
            case FunctionType.PHOTO:
                self.handle_function_tab_state(self.photo_tab_widget, self.folder_tab_widget, self.mapping_tab_widget,
                                              self.video_tab_widget)
            case FunctionType.FOLDER:
                self.handle_function_tab_state(self.folder_tab_widget, self.mapping_tab_widget, self.video_tab_widget,
                                              self.photo_tab_widget)
            case FunctionType.MAPPING:
                self.handle_function_tab_state(self.mapping_tab_widget, self.video_tab_widget, self.photo_tab_widget,
                                              self.folder_tab_widget)
            case FunctionType.VIDEO:
                self.handle_function_tab_state(self.video_tab_widget, self.photo_tab_widget, self.folder_tab_widget,
                                              self.mapping_tab_widget)
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
            case 0:
                callback(self.photo_tab_widget.controlWidget)
            case 1:
                callback(self.folder_tab_widget.controlWidget)
            case 2:
                callback(self.mapping_tab_widget.controlWidget)
            case 3:
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
        
        # Show status message
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
        Handle drop events with browser-like behavior
        """
        try:
            assert isinstance(a0, QtGui.QDropEvent)
        except AssertionError:
            return

        if (x := a0.mimeData()) is None:
            return

        if not x.hasUrls():
            a0.ignore()
            return

        a0.setDropAction(QtCore.Qt.DropAction.CopyAction)
        file_path = Path(x.urls()[0].toLocalFile())
        
        # Update the address bar with the dropped path
        self.address_bar.setText(file_path.as_posix())
        
        if file_path.is_dir():
            self.handle_path_main(file_path)
        elif file_path.is_file():
            self.handle_file(file_path)
            
        a0.accept()
        
        # Show status message
        self.statusbar.showMessage(f"Opened {file_path.name}", 2000)
        
    def handle_path_main(self, file_path: Path) -> None:
        """
        Handles a file path by checking the file extensions in the directory, validating the mapping and folder tabs, and calling the handle_path method with the appropriate arguments.
        """
        extensions = {y.suffix.lower() for y in file_path.iterdir()}
        mask = {ext in extensions for ext in file_manager.get_extensions(FileCategory.TABLE)}
        try:
            assert isinstance(self.mapping_tab_widget, UiMappingTabWidget)
            assert isinstance(self.folder_tab_widget, UiFolderTabWidget)
        except AssertionError:
            return
        if any(mask):
            self.handle_path(file_path, FunctionType.MAPPING, self.mapping_tab_widget.inputLineEdit)
        else:
            self.handle_path(file_path, FunctionType.FOLDER, self.folder_tab_widget.inputLineEdit)

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
            self.handle_function_tab_state(self.photo_tab_widget, self.folder_tab_widget, self.mapping_tab_widget,
                                          self.video_tab_widget)
        elif self.folder_tab_widget.selection_state == self.folder_tab_widget.SELECTED:
            self.handle_function_tab_state(self.folder_tab_widget, self.photo_tab_widget, self.mapping_tab_widget,
                                          self.video_tab_widget)
            self.folder_tab_widget.load_data()
        elif self.mapping_tab_widget.selection_state == self.mapping_tab_widget.SELECTED:
            self.handle_function_tab_state(self.mapping_tab_widget, self.photo_tab_widget, self.folder_tab_widget,
                                          self.video_tab_widget)

    def handle_file(self, file_path: Path) -> None:
        """
        Handles a file based on its file extension by calling the appropriate handler method.
        """
        if file_manager.is_valid_type(file_path, FileCategory.PHOTO) or file_manager.is_valid_type(file_path, FileCategory.TIFF) or file_manager.is_valid_type(file_path, FileCategory.RAW):
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
        self.handle_path(file_path, FunctionType.PHOTO, self.photo_tab_widget.inputLineEdit)

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
        self.video_tab_widget.inputLineEdit.setText(file_path.as_posix())
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
        self.mapping_tab_widget.tableLineEdit.setText(file_path.as_posix())
        data = prc.open_table(file_path)
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
                                      tab_widget.inputLineEdit,
                                      tab_widget.destinationLineEdit,
                                      *common_line_edits)

        match tab_widget:
            case tab_widget if isinstance(tab_widget, (UiPhotoTabWidget, UiFolderTabWidget)):
                ut.change_widget_state(check_button_state(),
                    tab_widget.cropButton,
                )
            case tab_widget if isinstance(tab_widget, UiMappingTabWidget):
                ut.change_widget_state(
                    check_button_state(tab_widget.tableLineEdit, tab_widget.comboBox_1, tab_widget.comboBox_2),
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
