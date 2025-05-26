import contextlib
from pathlib import Path

from PyQt6 import QtCore, QtGui, QtWidgets

from core.config import logger
from core.croppers import FolderCropper
from core.enums import FunctionType
from file_types import FileCategory, file_manager
from ui import utils as ut

from .batch_tab import UiBatchCropWidget


class UiFolderTabWidget(UiBatchCropWidget):
    """Folder tab widget - preview works with just input path"""

    def __init__(self, crop_worker: FolderCropper, object_name: str, parent: QtWidgets.QWidget) -> None:
        """Initialize the folder tab widget"""
        super().__init__(crop_worker, object_name, parent)

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
        """Set up the main layout structure with working preview system"""
        # ---- Page 1: Crop View ----
        frame, vertical_layout = self.setup_main_crop_frame(self.page_1)

        # Crop and cancel buttons
        button_layout = ut.setup_hbox("horizontalLayout_2")
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

        # ---- Page 2: Folder View ----
        self.treeView.setParent(self.page_2)
        self.treeView.setObjectName("treeView")
        self.treeView.setModel(self.file_model)

        # Setup mouse tracking and events
        self._setup_tree_view_events()

        # Add tree view
        self.verticalLayout_300.addWidget(self.treeView)
        self.toolBox.addItem(self.page_2, "Folder View")

        # Add toolbox to the main layout
        self.verticalLayout_100.addWidget(self.toolBox)

    def _setup_tree_view_events(self):
        """Set up tree view event handling"""
        logger.debug("Setting up tree view events for folder tab")

        # Enable mouse tracking on tree view
        self.treeView.setMouseTracking(True)
        self.treeView.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover, True)

        # Get viewport and set it up
        viewport = self.treeView.viewport()
        if viewport is not None:
            viewport.setMouseTracking(True)
            viewport.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover, True)
            viewport.installEventFilter(self)
            logger.debug("Event filter installed on tree view viewport")

        # Connect the entered signal
        self.treeView.entered.connect(self._on_item_entered)

    def _handle_image_hover(self, file_path: str, global_pos: QtCore.QPoint):
        """Handle hovering over an image file - ONLY INPUT PATH REQUIRED"""
        logger.debug(f"DEBUG: input_path='{self.input_path}', destination_path='{self.destination_path}'")

        # Only require input path for preview
        if not self.input_path:
            logger.warning("Cannot show preview - input path not set")
            return

        # Sanitize the file path
        if not (clean_path := ut.sanitize_path(file_path)):
            return

        logger.debug(f"Starting preview timer for: {clean_path}")
        # Start/update hover timer
        self._last_mouse_pos = global_pos
        self._pending_preview_path = clean_path
        self._hover_timer.start(300)

    def _hide_preview(self):
        """Hide the preview"""
        self._hover_timer.stop()
        self.image_preview.hide_preview()

    def eventFilter(self, obj, event):
        """Enhanced event filter for tree view viewport"""
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
                logger.debug("Mouse left viewport")
                self._hide_preview()

        return super().eventFilter(obj, event)

    def _on_item_entered(self, index):
        """Handle when mouse enters a tree view item - backup method"""
        logger.debug(f"Item entered signal: {index.row()}")

        if not self.input_path:
            logger.warning("Cannot show preview - input path not set")
            return

        file_path = self.file_model.filePath(index)
        logger.debug(f"File path from entered signal: {file_path}")

        if not file_path or len(file_path) <= 3:
            logger.warning("Invalid file path or length <= 3")
            return

        if sanitized_path := ut.sanitize_path(file_path):
            if self._is_image_file(sanitized_path):
                self._last_mouse_pos = QtGui.QCursor.pos()
                self._pending_preview_path = sanitized_path
                logger.debug(f"Starting preview timer for: {sanitized_path}")
                self._hover_timer.start(300)

    def _show_preview(self):
        """Show the preview image - ONLY INPUT PATH REQUIRED"""
        logger.debug(f"_show_preview called for: {self._pending_preview_path}")

        if not self._pending_preview_path or not self._last_mouse_pos:
            logger.debug("No pending preview or mouse position")
            return

        if not self.input_path:
            logger.warning("Cannot show preview - input path not set")
            return

        try:
            # Create paths
            folder_path = Path(self.input_path)

            if not folder_path.exists():
                logger.warning("Input path doesn't exist")
                return

            # Create minimal preview job
            preview_job = self.create_preview_job(folder_path)

            logger.debug(f"Showing preview for: {self._pending_preview_path}")
            logger.debug(f"At position: {self._last_mouse_pos}")

            # Show the preview
            self.image_preview.preview_file(
                self._pending_preview_path,
                self._last_mouse_pos,
                self.crop_worker.face_detection_tools[0],
                preview_job
            )

        except Exception as e:
            logger.exception(f"Error showing preview: {e}")
            import traceback
            traceback.print_exc()

    def load_data(self) -> None:
        """Load data into the tree view from the selected folder"""
        logger.debug(f"Loading folder data: {self.input_path}")

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

            # Set the root path and index
            self.file_model.setRootPath(self.input_path)
            root_index = self.file_model.index(self.input_path)
            self.treeView.setRootIndex(root_index)

            logger.debug(f"Loaded {self.file_model.rowCount(root_index)} items")

            # Re-setup events after loading data
            QtCore.QTimer.singleShot(200, self._setup_tree_view_events)

        except Exception as e:
            logger.exception(f"Error loading data: {e}")
            with contextlib.suppress(Exception):
                self.file_model.setRootPath("")
                self.treeView.setRootIndex(self.file_model.index(""))

    def _get_function_type(self):
        """Get the function type for this widget"""
        return FunctionType.FOLDER

    def connect_signals(self) -> None:
        """Connect widget signals to handlers"""
        # Button connections
        self.cropButton.clicked.connect(self.folder_process)

        # Clear hover preview cache when width or height changes
        self.controlWidget.widthLineEdit.textChanged.connect(self._clear_hover_preview_cache)
        self.controlWidget.heightLineEdit.textChanged.connect(self._clear_hover_preview_cache)

        # Register button dependencies with the TabStateManager
        ut.register_button_dependencies(
            self.tab_state_manager,
            self.cropButton,
            {
                self.controlWidget.widthLineEdit,
                self.controlWidget.heightLineEdit
            }
        )

        # Connect all input widgets for validation tracking
        self.tab_state_manager.connect_widgets(
            self.controlWidget.widthLineEdit,
            self.controlWidget.heightLineEdit,
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
        self.cropButton.setText("")
        self.cancelButton.setText("")
        self.toolBox.setItemText(self.toolBox.indexOf(self.page_1),
                                 QtCore.QCoreApplication.translate("self", "Crop View", None))
        self.toolBox.setItemText(self.toolBox.indexOf(self.page_2),
                                 QtCore.QCoreApplication.translate("self", "Folder View", None))

    def open_path(self, line_edit_type: str) -> None:
        """Open the file/folder selection dialog"""
        if line_edit_type == "destination":
            f_name = QtWidgets.QFileDialog.getExistingDirectory(
                self,
                'Select Directory',
                file_manager.get_default_directory(FileCategory.PHOTO).as_posix()
            )
            self.destination_path = f_name

            # Update main window's destination input if this is the active tab
            main_window = self.parent().parent().parent()
            if main_window.function_tabWidget.currentIndex() == FunctionType.FOLDER:
                main_window.destination_input.setText(f_name)

        elif line_edit_type == "input":
            f_name = QtWidgets.QFileDialog.getExistingDirectory(
                self,
                'Select Directory',
                file_manager.get_default_directory(FileCategory.PHOTO).as_posix()
            )
            self.input_path = f_name

            # Update main window's unified address bar if this is the active tab
            main_window = self.parent().parent().parent()
            if main_window.function_tabWidget.currentIndex() == FunctionType.FOLDER:
                main_window.unified_address_bar.setText(f_name)

            # Load the data into the tree view
            self.load_data()

    def connect_crop_worker(self) -> None:
        """Connect the signals from the crop worker to UI handlers"""
        widget_list = (self.controlWidget.widthLineEdit, self.controlWidget.heightLineEdit,
                    self.controlWidget.sensitivityDial, self.controlWidget.fpctDial, self.controlWidget.gammaDial,
                    self.controlWidget.topDial, self.controlWidget.bottomDial, self.controlWidget.leftDial,
                    self.controlWidget.rightDial, self.controlWidget.radioButton_none,
                    self.controlWidget.radioButton_bmp, self.controlWidget.radioButton_jpg,
                    self.controlWidget.radioButton_png, self.controlWidget.radioButton_tiff,
                    self.controlWidget.radioButton_webp, self.cropButton, self.exposureCheckBox, self.mfaceCheckBox,
                    self.tiltCheckBox)

        self.connect_crop_worker_signals(*widget_list)

    def folder_process(self) -> None:
        """Begin the folder cropping process"""
        self.crop_worker.show_message_box = False

        def execute_crop():
            self.cropButton.setEnabled(False)
            self.cropButton.repaint()

            job = self.create_job(
                FunctionType.FOLDER,
                folder_path=Path(self.input_path) if self.input_path else None,
                destination=Path(self.destination_path) if self.destination_path else None
            )
            self.run_batch_process(job,
                                function=self.crop_worker.crop,
                                reset_worker_func=lambda: self.crop_worker.reset_task())

        # Check if source and destination are the same and warn if needed
        if self.input_path and self.destination_path:
            self.check_source_destination_same(
                self.input_path,
                self.destination_path,
                FunctionType.FOLDER,
                execute_crop
            )
