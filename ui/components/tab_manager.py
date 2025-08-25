from PyQt6.QtCore import QSize, pyqtSignal
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QTabWidget, QWidget

from core.enums import FunctionType
from ui import utils as ut
from ui.enums import GuiIcon
from ui.folder_tab import UiFolderTabWidget
from ui.mapping_tab import UiMappingTabWidget
from ui.photo_tab import UiPhotoTabWidget
from ui.video_tab import UiVideoTabWidget

type TabWidget = UiPhotoTabWidget | UiFolderTabWidget | UiMappingTabWidget | UiVideoTabWidget


class TabManager:
    """Manages tab widgets and their states"""

    # Signals for tab management
    tab_changed = pyqtSignal(int)

    def __init__(self, parent_widget: QWidget, workers: dict):
        self.parent_widget = parent_widget
        self.workers = workers

        # Create tab widget
        self.function_tabWidget = QTabWidget(parent_widget)
        self.function_tabWidget.setObjectName("function_tabWidget")
        self.function_tabWidget.setMovable(True)

        # Tab widgets storage
        self.tab_widgets = {}

        # Create individual tabs
        self._create_tabs()

    def _create_tabs(self):
        """Create all tab widgets"""
        self._create_photo_tab()
        self._create_folder_tab()
        self._create_mapping_tab()
        self._create_video_tab()

    def _create_photo_tab(self):
        """Create photo tab without redundant input fields"""
        icon2 = QIcon()
        icon2.addFile(GuiIcon.PICTURE, QSize(), QIcon.Mode.Normal, QIcon.State.Off)

        self.photo_tab = QWidget()
        self.photo_tab.setObjectName("photo_tab")

        self.verticalLayout_2 = ut.setup_vbox("verticalLayout_2", self.photo_tab)
        self.photo_tab_widget = UiPhotoTabWidget(
            self.workers['photo'], "photo_tab_widget", self.photo_tab
        )

        # Hide the redundant input fields that will be handled by unified address bar
        self.photo_tab_widget.horizontalLayout_2.setParent(None)  # Remove input layout
        self.photo_tab_widget.horizontalLayout_3.setParent(None)  # Remove destination layout

        self.verticalLayout_2.addWidget(self.photo_tab_widget)
        self.function_tabWidget.addTab(self.photo_tab, icon2, "")

        self.tab_widgets[FunctionType.PHOTO] = self.photo_tab_widget

    def _create_folder_tab(self):
        """Create folder tab without redundant input fields"""
        icon3 = QIcon()
        icon3.addFile(GuiIcon.FOLDER, QSize(), QIcon.Mode.Normal, QIcon.State.Off)

        self.folder_tab = QWidget()
        self.folder_tab.setObjectName("folder_tab")

        self.verticalLayout_3 = ut.setup_vbox("verticalLayout_3", self.folder_tab)
        self.folder_tab_widget = UiFolderTabWidget(
            self.workers['folder'], "folder_tab_widget", self.folder_tab
        )

        # Modify the setup to remove redundant input fields
        # Access the verticalLayout_200 in page_1 to remove input and destination layouts
        input_layout = None
        destination_layout = None

        # Find the input and destination layouts
        for i in range(self.folder_tab_widget.verticalLayout_200.count()):
            item = self.folder_tab_widget.verticalLayout_200.itemAt(i)
            if item is None:
                continue
            if hasattr(item, 'layout') and item.layout() is not None:
                layout = item.layout()
                if layout is None:
                    continue
                if hasattr(layout, "objectName"):
                    if layout.objectName() == "horizontalLayout_4":  # Input layout
                        input_layout = item
                    elif layout.objectName() == "horizontalLayout_3":  # Destination layout
                        destination_layout = item

        # Remove input and destination layouts if found
        if input_layout:
            self.folder_tab_widget.verticalLayout_200.removeItem(input_layout)
        if destination_layout:
            self.folder_tab_widget.verticalLayout_200.removeItem(destination_layout)

        self.verticalLayout_3.addWidget(self.folder_tab_widget)
        self.function_tabWidget.addTab(self.folder_tab, icon3, "")

        self.tab_widgets[FunctionType.FOLDER] = self.folder_tab_widget

    def _create_mapping_tab(self):
        """Create mapping tab without redundant input fields"""
        icon1 = QIcon()
        icon1.addFile(GuiIcon.EXCEL, QSize(), QIcon.Mode.Normal, QIcon.State.Off)

        self.mapping_tab = QWidget()
        self.mapping_tab.setObjectName("mapping_tab")

        self.verticalLayout_4 = ut.setup_vbox("verticalLayout_4", self.mapping_tab)
        self.mapping_tab_widget = UiMappingTabWidget(
            self.workers['mapping'], "mapping_tab_widget", self.mapping_tab
        )

        self.verticalLayout_4.addWidget(self.mapping_tab_widget)
        self.function_tabWidget.addTab(self.mapping_tab, icon1, "")

        self.tab_widgets[FunctionType.MAPPING] = self.mapping_tab_widget

    def _create_video_tab(self):
        """Create video tab without redundant input fields"""
        icon4 = QIcon()
        icon4.addFile(GuiIcon.CLAPPERBOARD, QSize(), QIcon.Mode.Normal, QIcon.State.Off)

        self.video_tab = QWidget()
        self.video_tab.setObjectName("video_tab")

        self.verticalLayout_5 = ut.setup_vbox("verticalLayout_5", self.video_tab)
        self.video_tab_widget = UiVideoTabWidget(
            self.workers['video'], "video_tab_widget", self.video_tab
        )

        self.verticalLayout_5.addWidget(self.video_tab_widget)
        self.function_tabWidget.addTab(self.video_tab, icon4, "")

        self.tab_widgets[FunctionType.VIDEO] = self.video_tab_widget

    def get_current_tab_widget(self) -> TabWidget | None:
        """Get the currently active tab widget"""
        current_index = self.function_tabWidget.currentIndex()
        return self.tab_widgets.get(current_index)

    def get_tab_widget(self, function_type: FunctionType) -> TabWidget | None:
        """Get tab widget by function type"""
        return self.tab_widgets.get(function_type)

    def set_current_tab(self, function_type: FunctionType):
        """Set the current tab by function type"""
        self.function_tabWidget.setCurrentIndex(function_type.value)

    def get_current_index(self) -> int:
        """Get current tab index"""
        return self.function_tabWidget.currentIndex()

    def get_tab_count(self) -> int:
        """Get total number of tabs"""
        return self.function_tabWidget.count()

    def retranslate_ui(self):
        """Retranslate tab titles"""
        from PyQt6.QtCore import QCoreApplication

        self.function_tabWidget.setTabText(
            self.function_tabWidget.indexOf(self.photo_tab),
            QCoreApplication.translate("self", "Photo Crop", None)
        )
        self.function_tabWidget.setTabText(
            self.function_tabWidget.indexOf(self.folder_tab),
            QCoreApplication.translate("self", "Folder Crop", None)
        )
        self.function_tabWidget.setTabText(
            self.function_tabWidget.indexOf(self.mapping_tab),
            QCoreApplication.translate("self", "Mapping Crop", None)
        )
        self.function_tabWidget.setTabText(
            self.function_tabWidget.indexOf(self.video_tab),
            QCoreApplication.translate("self", "Video Crop", None)
        )

    def handle_tab_close(self, index: int):
        """Handle tab close button clicks"""
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
