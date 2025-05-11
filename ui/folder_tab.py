from pathlib import Path

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from core.croppers import FolderCropper
from core.enums import FunctionType
from file_types import FileCategory, file_manager
from ui import utils as ut

from .batch_tab import UiBatchCropWidget
from .image_hover_preview import ImageHoverPreview


class UiFolderTabWidget(UiBatchCropWidget):
    """Folder tab widget with enhanced inheritance from the batch crop_from_path widget"""

    def __init__(self, crop_worker: FolderCropper, object_name: str, parent: QtWidgets.QWidget) -> None:
        """Initialize the folder tab widget"""
        super().__init__(crop_worker, object_name, parent)

        # Path storage fields
        self.input_path = ""
        self.destination_path = ""

        # Create a file model for the tree view
        self.file_model = QtGui.QFileSystemModel(self)
        self.file_model.setFilter(QtCore.QDir.Filter.NoDotAndDotDot | QtCore.QDir.Filter.Files)
        p_types = (
            file_manager.get_extensions(FileCategory.PHOTO) |
            file_manager.get_extensions(FileCategory.TIFF) |
            file_manager.get_extensions(FileCategory.RAW)
        )
        file_filter = np.array([f'*{file}' for file in p_types])
        self.file_model.setNameFilters(file_filter)

        self.treeView = QtWidgets.QTreeView(self.page_2)

        # Create image preview widget
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
        # ---- Page 1: Crop View ---- (unchanged)
        # Main frame with image and controls
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

        # Add page to toolbox
        self.toolBox.addItem(self.page_1, "Crop View")

        # ---- Page 2: Folder View ----
        self.treeView.setObjectName("treeView")
        self.treeView.setModel(self.file_model)

        # Enable mouse tracking for hover detection
        self.treeView.setMouseTracking(True)

        # Connect tree view hover events
        self.treeView.entered.connect(self._on_item_entered)
        if viewport := self.treeView.viewport():
            viewport.installEventFilter(self)

        self.verticalLayout_300.addWidget(self.treeView)

        # Add page to toolbox
        self.toolBox.addItem(self.page_2, "Folder View")

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
        # Get file path from model index
        file_path = self.file_model.filePath(index)

        # Check if it's an image file
        if file_path and self._is_image_file(file_path):
            self._last_mouse_pos = QtGui.QCursor.pos()
            self._hover_timer.start(200)  # Delay before showing preview

    def _on_mouse_move(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse movement in tree view"""
        index = self.treeView.indexAt(event.position().toPoint())

        if index.isValid():
            file_path = self.file_model.filePath(index)
            if file_path and self._is_image_file(file_path):
                # Use globalPosition() for PyQt6 instead of globalPos()
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

        # Get current index under mouse
        pos = self.treeView.mapFromGlobal(self._last_mouse_pos)
        index = self.treeView.indexAt(pos)

        if index.isValid():
            file_path = self.file_model.filePath(index)
            if file_path and self._is_image_file(file_path):
                # Create job for preview
                job = self.create_job(
                    FunctionType.FOLDER,
                    folder_path=Path(self.input_path) if self.input_path else None,
                    destination=Path(self.destination_path) if self.destination_path else None
                )

                # Show preview
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

    def _is_image_file(self, file_path: str) -> bool:
        """Check if a file is an image"""
        path = Path(file_path)
        return (file_manager.is_valid_type(path, FileCategory.PHOTO) or
                file_manager.is_valid_type(path, FileCategory.TIFF) or
                file_manager.is_valid_type(path, FileCategory.RAW))

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
        self.cropButton.clicked.connect(self.folder_process)

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
        """Open the file /folder selection dialog with updated string-based approach"""
        if line_edit_type == "destination":
            f_name = QtWidgets.QFileDialog.getExistingDirectory(
                self,
                'Select Directory',
                file_manager.get_default_directory(FileCategory.PHOTO).as_posix()
            )
            # Validate the file exists and is accessible
            if f_name := ut.sanitize_path(f_name):
                # Update the destination path
                self.destination_path = f_name

                # Also update the main window's destination input if this is the active tab
                main_window = self.parent().parent().parent()
                if main_window.function_tabWidget.currentIndex() == FunctionType.FOLDER:
                    main_window.destination_input.setText(f_name)

        elif line_edit_type == "input":
            f_name = QtWidgets.QFileDialog.getExistingDirectory(
                self,
                'Select Directory',
                file_manager.get_default_directory(FileCategory.PHOTO).as_posix()
            )
            # Validate the file exists and is accessible
            if f_name := ut.sanitize_path(f_name):
                # Update the input path
                self.input_path = f_name

                # Also update the main window's unified address bar if this is the active tab
                main_window = self.parent().parent().parent()
                if main_window.function_tabWidget.currentIndex() == FunctionType.FOLDER:
                    main_window.unified_address_bar.setText(f_name)

                # Load the data into the tree view
                self.load_data()

    def load_data(self) -> None:
        """Load data into the tree view from the selected folder"""
        try:
            # Use the stored input_path instead of inputLineEdit.text()
            if not self.input_path:
                return

            # Verify the path exists and is a directory
            path = Path(self.input_path)
            if not path.exists():
                return

            if not path.is_dir():
                return

            self.file_model.setRootPath(self.input_path)
            self.treeView.setRootIndex(self.file_model.index(self.input_path))

        except (IndexError, FileNotFoundError, ValueError, AttributeError):
            return

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
            # Manually disable the crop_from_path button right away
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
