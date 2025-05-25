from pathlib import Path

import polars as pl
from PyQt6 import QtCore, QtGui, QtWidgets

from core import DataFrameModel
from core import processing as prc
from core.croppers import MappingCropper
from core.enums import FunctionType
from file_types import FileCategory, file_manager
from ui import utils as ut

from .batch_tab import UiBatchCropWidget


class UiMappingTabWidget(UiBatchCropWidget):
    """Mapping tab widget - preview works with just input path"""

    def __init__(self, crop_worker: MappingCropper, object_name: str, parent: QtWidgets.QWidget) -> None:
        """Initialize the mapping tab widget"""
        super().__init__(crop_worker, object_name, parent)

        self.table_path = ""

        # Data model
        self.model: DataFrameModel | None = None
        self.data_frame: pl.DataFrame | None = None

        # Create mapping-specific widgets
        self.tableButton = self.create_nav_button("tableButton")
        self.comboBox_1 = QtWidgets.QComboBox()
        self.comboBox_2 = QtWidgets.QComboBox()
        self.comboBox_3 = QtWidgets.QComboBox()
        self.comboBox_4 = QtWidgets.QComboBox()
        self.tableView = QtWidgets.QTableView()

        self.page_3 = QtWidgets.QWidget()
        self.page_3.setObjectName("page_3")

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
        """Set up the main layout structure with working preview"""
        # ---- Page 1: Crop View ----
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

        # ---- Page 2: Table View ----
        self.tableView.setObjectName("tableView")
        self.tableView.setParent(self.page_2)
        self.verticalLayout_300.addWidget(self.tableView)

        # Combo boxes for column selection on table view
        combo_layout = ut.setup_hbox("horizontalLayout_4")
        ut.setup_combobox(self.comboBox_3, combo_layout, self.size_policy_expand_fixed, "comboBox_3")
        ut.setup_combobox(self.comboBox_4, combo_layout, self.size_policy_expand_fixed, "comboBox_4")
        self.verticalLayout_300.addLayout(combo_layout)

        self.toolBox.addItem(self.page_2, "Table View")

        # ---- Page 3: Folder View ----
        self.treeView.setParent(self.page_3)
        self.verticalLayout_400 = ut.setup_vbox("verticalLayout_400", self.page_3)

        # Tree view for folder browsing
        self.treeView.setObjectName("treeView")
        self.treeView.setModel(self.file_model)

        # Setup working preview system
        self._setup_tree_view_events()

        self.verticalLayout_400.addWidget(self.treeView)
        self.toolBox.addItem(self.page_3, "Folder View")

        # Add toolbox to the main layout
        self.verticalLayout_100.addWidget(self.toolBox)

    def _setup_tree_view_events(self):
        """Set up tree view event handling for mapping tab"""
        print("Setting up tree view events for mapping tab")

        # Enable mouse tracking
        self.treeView.setMouseTracking(True)
        self.treeView.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover, True)

        # Setup viewport
        viewport = self.treeView.viewport()
        if viewport:
            viewport.setMouseTracking(True)
            viewport.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover, True)
            viewport.installEventFilter(self)
            print("Mapping tab: Event filter installed on tree view viewport")

        # Connect signals
        self.treeView.entered.connect(self._on_item_entered)

    def _handle_image_hover(self, file_path: str, global_pos: QtCore.QPoint):
        """Handle hovering over an image file in mapping tab - ONLY INPUT PATH REQUIRED"""
        print(f"Mapping DEBUG: input_path='{self.input_path}', destination_path='{self.destination_path}'")

        # Only require input path for preview
        if not self.input_path:
            print("Mapping tab: Cannot show preview - input path not set")
            return

        if not (clean_path := ut.sanitize_path(file_path)):
            return

        print(f"Mapping: Starting preview timer for: {clean_path}")
        self._last_mouse_pos = global_pos
        self._pending_preview_path = clean_path
        self._hover_timer.start(300)

    def _hide_preview(self):
        """Hide the preview for mapping tab"""
        self._hover_timer.stop()
        self.image_preview.hide_preview()

    def eventFilter(self, obj, event):
        """Event filter for mapping tab tree view"""
        if obj == self.treeView.viewport():
            event_type = event.type()

            if event_type == QtCore.QEvent.Type.MouseMove:
                pos = event.position().toPoint()
                index = self.treeView.indexAt(pos)

                if index.isValid():
                    file_path = self.file_model.filePath(index)
                    if file_path and self._is_image_file(file_path):
                        global_pos = obj.mapToGlobal(pos)
                        self._handle_image_hover(file_path, global_pos)
                    else:
                        self._hide_preview()
                else:
                    self._hide_preview()

            elif event_type == QtCore.QEvent.Type.Leave:
                print("Mouse left mapping viewport")
                self._hide_preview()

        return super().eventFilter(obj, event)

    def _on_item_entered(self, index):
        """Handle when mouse enters a tree view item in mapping tab"""
        print(f"Mapping item entered: {index.row()}")

        if not self.input_path:
            print("Mapping: Input path not set, skipping preview")
            return

        file_path = self.file_model.filePath(index)
        if not file_path or len(file_path) <= 3:
            return

        if sanitized_path := ut.sanitize_path(file_path):
            if self._is_image_file(sanitized_path):
                self._last_mouse_pos = QtGui.QCursor.pos()
                self._pending_preview_path = sanitized_path
                print(f"Mapping: Starting preview timer for: {sanitized_path}")
                self._hover_timer.start(300)

    def _show_preview(self):
        """Show preview for mapping tab - ONLY INPUT PATH REQUIRED"""
        print(f"Mapping _show_preview called for: {self._pending_preview_path}")

        if not self._pending_preview_path or not self._last_mouse_pos:
            return

        if not self.input_path:
            print("Mapping: Cannot show preview - input path not set")
            return

        try:
            folder_path = Path(self.input_path)

            if not folder_path.exists():
                print("Mapping: Input path doesn't exist")
                return

            # Create minimal preview job
            preview_job = self.create_preview_job(folder_path)

            print(f"Showing mapping preview for: {self._pending_preview_path}")
            print(f"At position: {self._last_mouse_pos}")

            self.image_preview.preview_file(
                self._pending_preview_path,
                self._last_mouse_pos,
                self.crop_worker.face_detection_tools[0],
                preview_job
            )

        except Exception as e:
            print(f"Error showing mapping preview: {e}")
            import traceback
            traceback.print_exc()

    def _get_function_type(self):
        """Get the function type for mapping tab"""
        return FunctionType.MAPPING

    def load_data(self) -> None:
        """Load data for mapping tab"""
        print(f"Loading mapping data: {self.input_path}")

        try:
            if not self.input_path:
                self.file_model.setRootPath("")
                self.treeView.setRootIndex(self.file_model.index(""))
                return

            path = Path(self.input_path)
            if not path.exists() or not path.is_dir():
                self.file_model.setRootPath("")
                self.treeView.setRootIndex(self.file_model.index(""))
                return

            self.file_model.setRootPath(self.input_path)
            root_index = self.file_model.index(self.input_path)
            self.treeView.setRootIndex(root_index)

            # Re-setup events after loading
            QtCore.QTimer.singleShot(200, self._setup_tree_view_events)

        except Exception as e:
            print(f"Error loading mapping data: {e}")

    def connect_signals(self) -> None:
        """Connect signals for mapping tab"""
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

        # Register button dependencies
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

        # Connect widgets for validation
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
        """Update UI text elements for mapping tab"""
        super().retranslateUi()
        self.comboBox_1.setPlaceholderText(QtCore.QCoreApplication.translate("self", "Filename column", None))
        self.comboBox_2.setPlaceholderText(QtCore.QCoreApplication.translate("self", "Mapping column", None))
        self.comboBox_3.setPlaceholderText(QtCore.QCoreApplication.translate("self", "Filename column", None))
        self.comboBox_4.setPlaceholderText(QtCore.QCoreApplication.translate("self", "Mapping column", None))
        self.cropButton.setText("")
        self.cancelButton.setText("")
        self.toolBox.setItemText(self.toolBox.indexOf(self.page_1), "Crop View")
        self.toolBox.setItemText(self.toolBox.indexOf(self.page_2), "Table View")
        self.toolBox.setItemText(self.toolBox.indexOf(self.page_3), "Folder View")

    def open_table(self) -> None:
        """Open table file dialog"""
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
        """Connect crop worker signals"""
        widget_list = (self.controlWidget.widthLineEdit, self.controlWidget.heightLineEdit,
                    self.controlWidget.sensitivityDial, self.controlWidget.fpctDial, self.controlWidget.gammaDial,
                    self.controlWidget.topDial, self.controlWidget.bottomDial, self.controlWidget.leftDial,
                    self.controlWidget.rightDial, self.tableButton, self.comboBox_1, self.comboBox_2,
                    self.controlWidget.radioButton_none,
                    self.controlWidget.radioButton_bmp, self.controlWidget.radioButton_jpg,
                    self.controlWidget.radioButton_png, self.controlWidget.radioButton_tiff,
                    self.controlWidget.radioButton_webp, self.cropButton, self.exposureCheckBox, self.mfaceCheckBox,
                    self.tiltCheckBox)

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
