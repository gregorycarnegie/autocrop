import contextlib
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import QApplication, QFrame, QMessageBox, QPushButton, QToolBox, QVBoxLayout, QWidget

from core import Job
from core.croppers import BatchCropper
from file_types import FileCategory, file_manager
from ui import utils as ut
from ui.image_hover_preview import ImageHoverPreview
from ui.pulsing_indicator import PulsingProgressIndicator

from .crop_widget import UiCropWidget
from .enums import GuiIcon


class UiBatchCropWidget(UiCropWidget):
    """
    Intermediate base widget class for batch cropping functionality.
    Provides common UI components and behaviors for folder and mapping tabs.
    """

    PROGRESSBAR_STEPS: int = 1_000

    def __init__(self, crop_worker: BatchCropper, object_name: str, parent: QWidget) -> None:
        """Initialize the batch crop widget with pulsing progress indicator"""
        super().__init__(parent, object_name)
        self.crop_worker = crop_worker
        self._pending_preview_path = ''
        self.input_path = ""
        self.destination_path = ""

        # Create pulsing progress indicator
        self.pulsing_indicator = PulsingProgressIndicator(self)
        self.pulsing_indicator.setObjectName("pulsingIndicator")

        # Create common UI elements for batch operations
        self.toolBox = QToolBox(self)
        self.toolBox.setObjectName("toolBox")

        # Create a file model for the tree view (like FolderCropper)
        self.file_model = QtGui.QFileSystemModel(self)
        self.file_model.setFilter(QtCore.QDir.Filter.NoDotAndDotDot | QtCore.QDir.Filter.Files)

        # IMPORTANT: Don't set any root path initially to prevent drive enumeration
        # The model will remain empty until a valid path is explicitly set

        p_types = (
            file_manager.get_extensions(FileCategory.PHOTO) |
            file_manager.get_extensions(FileCategory.TIFF) |
            file_manager.get_extensions(FileCategory.RAW)
        )
        file_filter = np.array([f'*{file}' for file in p_types])
        self.file_model.setNameFilters(file_filter)

        # Create image preview widget (like FolderCropper)
        self.image_preview = ImageHoverPreview(parent=None)
        self.image_preview.hide()

        # Track mouse position for preview
        self._last_mouse_pos = None
        self._hover_timer = QtCore.QTimer()
        self._hover_timer.setSingleShot(True)
        self._hover_timer.timeout.connect(self._show_preview)

        # Create pages for the toolbox
        self.page_1 = QWidget()
        self.page_1.setObjectName("page_1")
        self.page_2 = QWidget()
        self.page_2.setObjectName("page_2")

        self.treeView = QtWidgets.QTreeView()

        # Set up page layouts
        self.verticalLayout_200 = ut.setup_vbox("verticalLayout_200", self.page_1)
        self.verticalLayout_300 = ut.setup_vbox("verticalLayout_300", self.page_2)

        # Buttons that all batch processors need
        self.cropButton, self.cancelButton = self.create_main_action_buttons()
        self.cancelButton.clicked.connect(lambda: self.handle_cancel_button_click())

        self.connect_crop_worker_signals(self.cropButton)

    def connect_signals(self) -> None:
        """Connect widget signals to handlers"""
        # Connect cancel button to terminate
        self.cancelButton.clicked.connect(self.crop_worker.terminate)
        self.cancelButton.clicked.connect(self.handle_cancel_button_click)

        # Connect crop worker signals for proper state management
        self.connect_crop_worker()

    def handle_cancel_button_click(self) -> None:
        """Handle cancel button click to properly update button states"""
        try:
            self.crop_worker.terminate()
        except Exception as e:
            print(f"Error terminating worker: {e}")

        # Force button states immediately
        self.cancelButton.setEnabled(False)
        self.cropButton.setEnabled(True)

        # Reset pulsing indicator to idle state
        self.pulsing_indicator.reset()

        # Force UI update
        QApplication.processEvents()

    def enable_cancel_button(self) -> None:
        """Enable the cancel button immediately"""
        self.cancelButton.setEnabled(True)
        self.cancelButton.repaint()
        QApplication.processEvents()

    def connect_crop_worker(self) -> None:
        raise NotImplementedError("function must be implemented in subclasses.")

    def create_main_action_buttons(self, parent_frame: QFrame | None=None) -> tuple[QPushButton, QPushButton]:
        """Create crop_from_path and cancel buttons with consistent styling"""
        # Crop button
        crop_button = self.create_main_button("cropButton", GuiIcon.CROP)
        crop_button.setParent(parent_frame)
        crop_button.setDisabled(True)

        # Cancel button
        cancel_button = self.create_main_button("cancelButton", GuiIcon.CANCEL)
        cancel_button.setParent(parent_frame)
        cancel_button.setDisabled(True)

        return crop_button, cancel_button

    def setup_main_crop_frame(self, parent_widget: QWidget) -> tuple[QFrame, QVBoxLayout]:
        """Create and set up the main crop_from_path frame with checkboxes and image widget"""
        frame = self.create_main_frame("frame")
        frame.setParent(parent_widget)
        vertical_layout = ut.setup_vbox("verticalLayout", frame)

        # Checkbox section
        self.toggleCheckBox.setParent(frame)
        self.mfaceCheckBox.setParent(frame)
        self.tiltCheckBox.setParent(frame)
        self.exposureCheckBox.setParent(frame)

        checkbox_layout = ut.setup_hbox("horizontalLayout_1")
        self.setup_checkboxes_frame(checkbox_layout)
        vertical_layout.addLayout(checkbox_layout)

        # Image widget
        self.imageWidget.setParent(frame)
        vertical_layout.addWidget(self.imageWidget)

        return frame, vertical_layout

    @staticmethod
    def cancel_button_operation(cancel_button: QPushButton, *crop_buttons: QPushButton) -> None:
        """Handle cancel button operations"""
        cancel_button.setDisabled(True)
        for crop_button in crop_buttons:
            crop_button.setEnabled(True)

    def connect_crop_worker_signals(self, *widget_list: QWidget) -> None:
        """Connect the signals from the crop worker to UI handlers"""
        with contextlib.suppress(TypeError, RuntimeError):
            # Disconnect existing connections to avoid duplicates
            self.crop_worker.started.disconnect()
            self.crop_worker.finished.disconnect()
            # No need to disconnect progress signal anymore

        # Batch start connection - disable all controls and start pulsing
        self.crop_worker.started.connect(lambda: ut.disable_widget(*widget_list))
        self.crop_worker.started.connect(lambda: ut.enable_widget(self.cancelButton))
        self.crop_worker.started.connect(self.pulsing_indicator.start_processing)

        # Batch end connection - re-enable controls and show complete state
        self.crop_worker.finished.connect(lambda: ut.enable_widget(*widget_list))
        self.crop_worker.finished.connect(lambda: ut.disable_widget(self.cancelButton))
        self.crop_worker.finished.connect(self.pulsing_indicator.finish_processing)
        self.crop_worker.finished.connect(lambda: ut.show_message_box(self.destination))

    def run_batch_process(self, job: Job, *,
                          function: Callable[..., Any],
                          reset_worker_func: Callable[..., Any]) -> None:
        """Run a batch processing operation using threading instead of multiprocessing"""
        reset_worker_func()

        # Reset pulsing indicator
        self.pulsing_indicator.reset()

        # Disable crop button and enable cancel button manually
        self.pulsing_indicator.start_processing()
        self.cropButton.setEnabled(False)
        self.cropButton.repaint()
        self.enable_cancel_button()

        # Process UI events to ensure indicator is shown
        QApplication.processEvents()

        # Use Thread instead of Process to avoid pickling issues
        thread = threading.Thread(target=function, args=(job,), daemon=True)
        thread.start()

    @staticmethod
    def check_source_destination_same(
        source_path: str,
        dest_path: str,
        function_type,
        process_func: Callable
    ) -> None:
        """Check if source and destination are the same and warn if needed"""
        if Path(source_path) == Path(dest_path):
            match ut.show_warning(function_type):
                case QMessageBox.StandardButton.Yes:
                    process_func()
                case _:
                    return
        else:
            process_func()

    def _on_item_entered(self, index):
        """Handle when mouse enters a tree view item"""
        # Early return if no valid paths are set to avoid processing drive letters
        if not self.input_path or not self.destination_path:
            return

        file_path = self.file_model.filePath(index)

        # Skip if the file path is empty or appears to be a drive letter
        if not file_path or len(file_path) <= 3:  # Skip "C:", "C:\", etc.
            return

        # Only sanitize and process if we have a reasonable file path
        if file_path := ut.sanitize_path(file_path):
            if self._is_image_file(file_path):
                self._last_mouse_pos = QtGui.QCursor.pos()
                # Store the file path that should be previewed when timer fires
                self._pending_preview_path = file_path
                self._hover_timer.start(200)

    def eventFilter(self, obj, event):
        """Filter events to handle mouse movement and leaving the tree view"""
        if obj == self.treeView.viewport():
            if event.type() == QtCore.QEvent.Type.MouseMove:
                self._on_mouse_move(event)
            elif event.type() == QtCore.QEvent.Type.Leave:
                self._on_mouse_leave()
        return super().eventFilter(obj, event)

    def _on_mouse_move(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse movement in tree view"""
        # Early return if no valid paths are set to avoid processing drive letters
        if not self.input_path or not self.destination_path:
            self._hide_preview()
            return

        index = self.treeView.indexAt(event.position().toPoint())

        if index.isValid():
            file_path = self.file_model.filePath(index)

            # Skip if the file path is empty or appears to be a drive letter
            if not file_path or len(file_path) <= 3:  # Skip "C:", "C:\", etc.
                self._hide_preview()
                return

            # Only sanitize and process if we have a reasonable file path
            if file_path := ut.sanitize_path(file_path):
                if self._is_image_file(file_path):
                    global_pos = event.globalPosition().toPoint()
                    self._last_mouse_pos = global_pos
                    # Store the file path that should be previewed when timer fires
                    self._pending_preview_path = file_path
                    if not self._hover_timer.isActive():
                        self._hover_timer.start(200)
                else:
                    self._hide_preview()
            else:
                self._hide_preview()
        else:
            self._hide_preview()

    def _on_mouse_leave(self):
        """Handle mouse leaving the tree view"""
        self._hide_preview()

    def _show_preview(self):
        """Show the preview image in the preview widget"""
        # Early return if no valid paths are set
        if not self.input_path or not self.destination_path:
            return

        if self._last_mouse_pos is None or not self._pending_preview_path:
            return

        # Use the stored file path instead of recalculating from mouse position
        file_path = self._pending_preview_path

        if file_path and self._is_image_file(file_path):
            try:
                # Create job for preview with proper path validation
                folder_path = Path(self.input_path) if self.input_path and Path(self.input_path).exists() else None
                destination_path = Path(self.destination_path) if self.destination_path and Path(self.destination_path).exists() else None

                # Skip preview if required paths are not valid
                if not folder_path or not destination_path:
                    return

                job = self.create_job(
                    self._get_function_type(),  # This should be implemented by subclasses
                    folder_path=folder_path,
                    destination=destination_path
                )

                # Show preview
                self.image_preview.preview_file(
                    file_path,
                    self._last_mouse_pos,
                    self.crop_worker.face_detection_tools[0],
                    job
                )
            except Exception as e:
                # Silently handle preview errors to avoid disrupting user experience
                print(f"Preview error: {e}")
                return

    def _get_function_type(self):
        """Get the function type for this widget - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _get_function_type method")

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
                # Clear the tree view when no path is set to prevent showing all drives
                self.file_model.setRootPath("")
                self.treeView.setRootIndex(self.file_model.index(""))
                return

            path = Path(self.input_path)
            if not path.exists() or not path.is_dir():
                # Clear the tree view for invalid paths
                self.file_model.setRootPath("")
                self.treeView.setRootIndex(self.file_model.index(""))
                return

            self.file_model.setRootPath(self.input_path)
            self.treeView.setRootIndex(self.file_model.index(self.input_path))

        except (IndexError, FileNotFoundError, ValueError, AttributeError):
            # Clear the tree view on any error
            with contextlib.suppress(Exception):
                self.file_model.setRootPath("")
                self.treeView.setRootIndex(self.file_model.index(""))
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
