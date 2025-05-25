from pathlib import Path

import numpy as np
import polars as pl
from PyQt6 import QtCore, QtGui, QtWidgets

from core import DataFrameModel
from core import processing as prc
from core.croppers import MappingCropper
from core.enums import FunctionType
from file_types import FileCategory, file_manager
from ui import utils as ut
from ui.image_hover_preview import ImageHoverPreview

from .batch_tab import UiBatchCropWidget


class UiMappingTabWidget(UiBatchCropWidget):
    """Mapping tab widget with enhanced inheritance from batch crop_from_path widget"""

    def __init__(self, crop_worker: MappingCropper, object_name: str, parent: QtWidgets.QWidget) -> None:
        """Initialize the mapping tab widget"""
        super().__init__(crop_worker, object_name, parent)

        # Path storage fields
        self.input_path = ""         # Folder path
        self.table_path = ""         # Table file path
        self.destination_path = ""   # Destination folder

        # Data model
        self.model: DataFrameModel | None = None
        self.data_frame: pl.DataFrame | None = None

        # Create a file model for the tree view (like FolderCropper)
        self.file_model = QtGui.QFileSystemModel(self)
        self.file_model.setFilter(QtCore.QDir.Filter.NoDotAndDotDot | QtCore.QDir.Filter.Files)
        p_types = (
            file_manager.get_extensions(FileCategory.PHOTO) |
            file_manager.get_extensions(FileCategory.TIFF) |
            file_manager.get_extensions(FileCategory.RAW)
        )
        file_filter = np.array([f'*{file}' for file in p_types])
        self.file_model.setNameFilters(file_filter)

        # Create mapping-specific widgets
        self.tableButton = self.create_nav_button("tableButton")
        self.comboBox_1 = QtWidgets.QComboBox()
        self.comboBox_2 = QtWidgets.QComboBox()
        self.comboBox_3 = QtWidgets.QComboBox()
        self.comboBox_4 = QtWidgets.QComboBox()
        self.tableView = QtWidgets.QTableView()

        # Add tree view for folder browsing
        self.treeView = QtWidgets.QTreeView()

        # Create image preview widget (like FolderCropper)
        self.image_preview = ImageHoverPreview(parent=None)
        self.image_preview.hide()

        # Track mouse position for preview
        self._last_mouse_pos = None
        self._hover_timer = QtCore.QTimer()
        self._hover_timer.setSingleShot(True)
        self._hover_timer.timeout.connect(self._show_preview)

        # Set up the main layout structure
        self.setup_layouts()

        # Connect signals
        self.connect_signals()

        # Set initial UI text
        self.retranslateUi()

        # Set initial toolbox page
        self.toolBox.setCurrentIndex(0)

        QtCore.QMetaObject.connectSlotsByName(self)

    def setup_layouts(self) -> None:
        """Set up the main layout structure with pulsing indicator"""
        # ---- Page 1: Crop View ---- (same as before)
        frame, vertical_layout = self.setup_main_crop_frame(self.page_1)

        # Combo boxes, crop and cancel buttons
        button_layout = ut.setup_hbox("horizontalLayout_1")

        # Setup combo boxes
        ut.setup_combobox(self.comboBox_1, button_layout, self.size_policy_expand_fixed, "comboBox_1")
        ut.setup_combobox(self.comboBox_2, button_layout, self.size_policy_expand_fixed, "comboBox_2")

        # Crop and cancel buttons
        self.cropButton.setParent(frame)
        self.cancelButton.setParent(frame)
        button_layout.addWidget(self.cropButton)
        button_layout.addWidget(self.cancelButton)

        vertical_layout.addLayout(button_layout)

        # Pulsing indicator
        self.pulsing_indicator.setParent(frame)
        vertical_layout.addWidget(self.pulsing_indicator)

        self.verticalLayout_200.addWidget(frame)
        self.toolBox.addItem(self.page_1, "Crop View")

        # ---- Page 2: Table View ---- (same as before)
        self.tableView.setObjectName("tableView")
        self.tableView.setParent(self.page_2)
        self.verticalLayout_300.addWidget(self.tableView)

        # Combo boxes for column selection on table view
        combo_layout = ut.setup_hbox("horizontalLayout_4")
        ut.setup_combobox(self.comboBox_3, combo_layout, self.size_policy_expand_fixed, "comboBox_3")
        ut.setup_combobox(self.comboBox_4, combo_layout, self.size_policy_expand_fixed, "comboBox_4")
        self.verticalLayout_300.addLayout(combo_layout)

        self.toolBox.addItem(self.page_2, "Table View")

        # ---- NEW Page 3: Folder View ----
        self.page_3 = QtWidgets.QWidget()
        self.page_3.setObjectName("page_3")
        self.verticalLayout_400 = ut.setup_vbox("verticalLayout_400", self.page_3)

        # Tree view for folder browsing
        self.treeView.setObjectName("treeView")
        self.treeView.setModel(self.file_model)
        self.treeView.setMouseTracking(True)

        # Connect tree view hover events (like FolderCropper)
        self.treeView.entered.connect(self._on_item_entered)
        if viewport := self.treeView.viewport():
            viewport.installEventFilter(self)

        self.verticalLayout_400.addWidget(self.treeView)
        self.toolBox.addItem(self.page_3, "Folder View")

        # Add toolbox to the main layout
        self.verticalLayout_100.addWidget(self.toolBox)

    def eventFilter(self, obj, event):
        """Filter events to handle mouse movement and leaving the tree view"""
        if obj == self.treeView.viewport():
            if event.type() == QtCore.QEvent.Type.MouseMove:
                self._on_mouse_move(event)
            elif event.type() == QtCore.QEvent.Type.Leave:
                self._on_mouse_leave()
        return super().eventFilter(obj, event)

    def _on_item_entered(self, index):
        """Handle when mouse enters a tree view item"""
        file_path = self.file_model.filePath(index)
        file_path = ut.sanitize_path(file_path)

        if file_path and self._is_image_file(file_path):
            self._last_mouse_pos = QtGui.QCursor.pos()
            self._hover_timer.start(200)

    def _on_mouse_move(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse movement in tree view"""
        index = self.treeView.indexAt(event.position().toPoint())

        if index.isValid():
            file_path = self.file_model.filePath(index)
            file_path = ut.sanitize_path(file_path)

            if file_path and self._is_image_file(file_path):
                global_pos = event.globalPosition().toPoint()
                self._last_mouse_pos = global_pos
                if not self._hover_timer.isActive():
                    self._hover_timer.start(200)
            else:
                self._hide_preview()
        else:
            self._hide_preview()

    def _on_mouse_leave(self):
        """Handle mouse leaving the tree view"""
        self._hide_preview()

    def _show_preview(self):
        """Show the image preview"""
        if self._last_mouse_pos is None:
            return

        pos = self.treeView.mapFromGlobal(self._last_mouse_pos)
        index = self.treeView.indexAt(pos)

        if index.isValid():
            file_path = self.file_model.filePath(index)
            file_path = ut.sanitize_path(file_path)

            if file_path and self._is_image_file(file_path):
                job = self.create_job(
                    FunctionType.MAPPING,
                    folder_path=Path(self.input_path) if self.input_path else None,
                    destination=Path(self.destination_path) if self.destination_path else None,
                    table=self.data_frame,
                    column1=self.comboBox_1,
                    column2=self.comboBox_2
                )

                self.image_preview.preview_file(
                    file_path,
                    self._last_mouse_pos,
                    self.crop_worker.face_detection_tools[0],
                    job
                )

    def _hide_preview(self):
        """Hide the image preview"""
        self._hover_timer.stop()
        self.image_preview.hide_preview()

    def _clear_hover_preview_cache(self):
        """Clear the hover preview cache."""
        self.image_preview.clear_cache()

    def _is_image_file(self, file_path: str) -> bool:
        """Check if a file is an image"""
        path = Path(file_path)
        return (file_manager.is_valid_type(path, FileCategory.PHOTO) or
                file_manager.is_valid_type(path, FileCategory.TIFF) or
                file_manager.is_valid_type(path, FileCategory.RAW))

    def load_data(self) -> None:
        """Load data into the tree view from the selected folder"""
        try:
            if not self.input_path:
                return

            path = Path(self.input_path)
            if not path.exists() or not path.is_dir():
                return

            self.file_model.setRootPath(self.input_path)
            self.treeView.setRootIndex(self.file_model.index(self.input_path))

        except (IndexError, FileNotFoundError, ValueError, AttributeError):
            return

    # Clean up when tab is hidden or destroyed
    def hideEvent(self, event):
        """Hide preview when tab is hidden"""
        self._hide_preview()
        super().hideEvent(event)

    def closeEvent(self, event):
        """Clean up when widget is closed"""
        self._hide_preview()
        self.image_preview.clear_cache()
        self.image_preview.close()
        super().closeEvent(event)

    def connect_signals(self) -> None:
        """Connect widget signals to handlers"""
        # Button connections
        self.tableButton.clicked.connect(self.open_table)
        self.cropButton.clicked.connect(self.mapping_process)

        # Clear hover preview cache when settings change
        self.controlWidget.widthLineEdit.textChanged.connect(self._clear_hover_preview_cache)
        self.controlWidget.heightLineEdit.textChanged.connect(self._clear_hover_preview_cache)

        # Combobox synchronization
        self.comboBox_1.currentTextChanged.connect(lambda text: self.comboBox_3.setCurrentText(text))
        self.comboBox_2.currentTextChanged.connect(lambda text: self.comboBox_4.setCurrentText(text))
        self.comboBox_3.currentTextChanged.connect(lambda text: self.comboBox_1.setCurrentText(text))
        self.comboBox_4.currentTextChanged.connect(lambda text: self.comboBox_2.setCurrentText(text))

        # Register button dependencies with the TabStateManager
        ut.register_button_dependencies(
            self.tab_state_manager,
            self.cropButton,
            {
                self.comboBox_1,
                self.comboBox_2,
                self.controlWidget.widthLineEdit,
                self.controlWidget.heightLineEdit
            }
        )

        # Connect all input widgets for validation tracking
        self.tab_state_manager.connect_widgets(
            self.controlWidget.widthLineEdit,
            self.controlWidget.heightLineEdit,
            self.comboBox_1,
            self.comboBox_2,
            self.exposureCheckBox,
            self.mfaceCheckBox,
            self.tiltCheckBox,
            self.controlWidget.sensitivityDial,
            self.controlWidget.fpctDial,
            self.controlWidget.gammaDial,
            self.controlWidget.topDial,
            self.controlWidget.bottomDial,
            self.controlWidget.leftDial,
            self.controlWidget.rightDial
        )

    def retranslateUi(self) -> None:
        """Update UI text elements"""
        super().retranslateUi()
        self.comboBox_1.setPlaceholderText(QtCore.QCoreApplication.translate("self", "Filename column", None))
        self.comboBox_2.setPlaceholderText(QtCore.QCoreApplication.translate("self", "Mapping column", None))
        self.comboBox_3.setPlaceholderText(QtCore.QCoreApplication.translate("self", "Filename column", None))
        self.comboBox_4.setPlaceholderText(QtCore.QCoreApplication.translate("self", "Mapping column", None))
        self.cropButton.setText("")
        self.cancelButton.setText("")
        self.toolBox.setItemText(self.toolBox.indexOf(self.page_1),
                                 QtCore.QCoreApplication.translate("self", "Crop View", None))
        self.toolBox.setItemText(self.toolBox.indexOf(self.page_2),
                                 QtCore.QCoreApplication.translate("self", "Table View", None))
        self.toolBox.setItemText(self.toolBox.indexOf(self.page_3),
                                 QtCore.QCoreApplication.translate("self", "Folder View", None))

    def open_table(self) -> None:
        """Open a table file dialog with the string-based approach"""
        f_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Open File',
            file_manager.get_default_directory(FileCategory.PHOTO).as_posix(),
            file_manager.get_filter_string(FileCategory.TABLE)
        )

        if f_name := ut.sanitize_path(f_name):
            self.table_path = f_name

            main_window = self.parent().parent().parent()
            if main_window.function_tabWidget.currentIndex() == FunctionType.MAPPING:
                main_window.secondary_input.setText(f_name)

            data = prc.load_table(Path(f_name))
            self.process_data(data)

    def process_data(self, data: pl.DataFrame) -> None:
        """Process the loaded data"""
        try:
            self.data_frame = data
            self.model = DataFrameModel(self.data_frame)
            self.tableView.setModel(self.model)

            columns = self.data_frame.columns
            for combo in [self.comboBox_1, self.comboBox_2, self.comboBox_3, self.comboBox_4]:
                combo.clear()
                combo.addItems(columns)

        except Exception as e:
            print(f"Error processing data: {e}")

    def connect_crop_worker(self) -> None:
        """Connect the signals from the crop worker to UI handlers"""
        widget_list = (self.controlWidget.widthLineEdit, self.controlWidget.heightLineEdit,
                    self.controlWidget.sensitivityDial, self.controlWidget.fpctDial, self.controlWidget.gammaDial,
                    self.controlWidget.topDial, self.controlWidget.bottomDial, self.controlWidget.leftDial,
                    self.controlWidget.rightDial, self.tableButton, self.comboBox_1, self.comboBox_2,
                    self.controlWidget.radioButton_none,
                    self.controlWidget.radioButton_bmp, self.controlWidget.radioButton_jpg,
                    self.controlWidget.radioButton_png, self.controlWidget.radioButton_tiff,
                    self.controlWidget.radioButton_webp, self.cropButton, self.exposureCheckBox,
                    self.mfaceCheckBox, self.tiltCheckBox)

        self.connect_crop_worker_signals(*widget_list)

    def mapping_process(self) -> None:
        """Begin the mapping cropping process"""
        self.crop_worker.show_message_box = False

        def execute_crop():
            job = self.create_job(
                FunctionType.MAPPING,
                folder_path=Path(self.input_path),
                destination=Path(self.destination_path),
                table=self.data_frame,
                column1=self.comboBox_1,
                column2=self.comboBox_2
            )
            self.run_batch_process(job,
                                   function=self.crop_worker.crop,
                                   reset_worker_func=lambda: self.crop_worker.reset_task())

        self.check_source_destination_same(
            self.input_path,
            self.destination_path,
            FunctionType.MAPPING,
            execute_crop
        )
