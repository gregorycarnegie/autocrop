import contextlib
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PyQt6.QtCore import QDir, QEvent, QPoint, Qt, QTimer
from PyQt6.QtGui import QCursor, QFileSystemModel
from PyQt6.QtWidgets import QApplication, QFrame, QMessageBox, QPushButton, QToolBox, QTreeView, QVBoxLayout, QWidget

from core import Job
from core.config import logger
from core.croppers import BatchCropper
from file_types import FileCategory, file_manager
from ui import utils as ut
from ui.image_hover_preview import ImageHoverPreview
from ui.pulsing_indicator import PulsingProgressIndicator

from .crop_widget import UiCropWidget
from .enums import GuiIcon


class UiBatchCropWidget(UiCropWidget):
    """
    Enhanced batch crop widget with improved preview system
    """

    PROGRESSBAR_STEPS: int = 1_000

    def __init__(self, crop_worker: BatchCropper, object_name: str, parent: QWidget) -> None:
        """Initialize the batch crop widget with debugging"""
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

        # Create a file model for the tree view
        self.file_model = QFileSystemModel(self)
        self.file_model.setFilter(QDir.Filter.NoDotAndDotDot | QDir.Filter.Files)

        # Set up file filters
        p_types = (
            file_manager.get_extensions(FileCategory.PHOTO) |
            file_manager.get_extensions(FileCategory.TIFF) |
            file_manager.get_extensions(FileCategory.RAW)
        )
        file_filter = np.array([f'*{file}' for file in p_types])
        self.file_model.setNameFilters(file_filter)
        self.file_model.setNameFilterDisables(False)

        # Create image preview widget with no parent initially
        self.image_preview = ImageHoverPreview(parent=None)
        self.image_preview.hide()

        # Enhanced hover tracking
        self._last_mouse_pos = None
        self._hover_timer = QTimer()
        self._hover_timer.setSingleShot(True)
        self._hover_timer.timeout.connect(self._show_preview)
        self._current_hovered_file = None

        # Create pages for the toolbox
        self.page_1 = QWidget()
        self.page_1.setObjectName("page_1")
        self.page_2 = QWidget()
        self.page_2.setObjectName("page_2")

        self.treeView = QTreeView()
        self._setup_tree_view()

        # Set up page layouts
        self.verticalLayout_200 = ut.setup_vbox("verticalLayout_200", self.page_1)
        self.verticalLayout_300 = ut.setup_vbox("verticalLayout_300", self.page_2)

        # Buttons that all batch processors need
        self.cropButton, self.cancelButton = self.create_main_action_buttons()
        self.cancelButton.clicked.connect(lambda: self.handle_cancel_button_click())

        self.connect_crop_worker_signals(self.cropButton)
        logger.debug(f"UiBatchCropWidget initialized: {object_name}")
    def _setup_tree_view(self):
        """Set up tree view with proper event handling"""
        logger.debug("Setting up tree view with enhanced mouse tracking")
        # Configure tree view
        self.treeView.setObjectName("treeView")
        self.treeView.setModel(self.file_model)
        self.treeView.setMouseTracking(True)

        # Get viewport and configure it
        viewport = self.treeView.viewport()
        if viewport is not None:
            viewport.setMouseTracking(True)
            viewport.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
            viewport.installEventFilter(self)
            logger.debug("Viewport configured with mouse tracking and event filter")
        # Enable hover for tree view itself
        self.treeView.setAttribute(Qt.WidgetAttribute.WA_Hover, True)

        # Connect signals - use lambda to ensure proper connection
        self.treeView.entered.connect(lambda index: self._on_item_entered(index))

        # Additional connection for mouse events
        self.treeView.mouseMoveEvent = self._tree_mouse_move_event

    def _tree_mouse_move_event(self, event):
        """Custom mouse move event handler for tree view"""
        logger.debug(f"Tree view mouse move at: {event.position()}")
        # Get the index at the mouse position
        index = self.treeView.indexAt(event.position().toPoint())

        if index.isValid():
            file_path = self.file_model.filePath(index)
            logger.debug(f"Mouse over file: {file_path}")
            if self._is_image_file(file_path):
                self._handle_hover_start(file_path, event.globalPosition().toPoint())
        else:
            self._handle_hover_end()

        # Call the original mouse move event
        QTreeView.mouseMoveEvent(self.treeView, event)

    def _handle_hover_start(self, file_path: str, global_pos: QPoint):
        """Handle start of hover over an image file"""
        logger.debug(f"Hover start: {file_path}")
        # Skip if paths aren't set up
        if not self.input_path or not self.destination_path:
            logger.debug("Skipping hover - paths not set")
            return

        if file_path == self._current_hovered_file:
            return  # Already hovering over this file

        self._current_hovered_file = file_path
        self._last_mouse_pos = global_pos
        self._pending_preview_path = file_path

        # Start timer for delayed preview
        self._hover_timer.start(500)  # 500ms delay

    def _handle_hover_end(self):
        """Handle end of hover"""
        logger.debug("Hover end")
        self._hover_timer.stop()
        self._current_hovered_file = None
        self.image_preview.hide_preview()

    def eventFilter(self, obj, event):
        """Enhanced event filter with comprehensive logging"""
        if obj == self.treeView.viewport():
            event_type = event.type()
            logger.debug(f"Event filter - Type: {event_type}")
            if event_type == QEvent.Type.MouseMove:
                logger.debug(f"Mouse move event in viewport: {event.position()}")
                self._handle_viewport_mouse_move(event)
                return False
            elif event_type == QEvent.Type.Leave:
                logger.debug("Mouse leave viewport")
                self._handle_hover_end()
                return False
            elif event_type == QEvent.Type.Enter:
                logger.debug("Mouse enter viewport")
                return False

        return super().eventFilter(obj, event)

    def _handle_viewport_mouse_move(self, event):
        """Handle mouse move in viewport"""
        # Get index at mouse position
        pos = event.position().toPoint()
        index = self.treeView.indexAt(pos)

        if index.isValid():
            file_path = self.file_model.filePath(index)
            if file_path and self._is_image_file(file_path):
                global_pos = self.treeView.viewport().mapToGlobal(pos)
                self._handle_hover_start(file_path, global_pos)
                return

        self._handle_hover_end()

    def _on_item_entered(self, index):
        """Handle when mouse enters a tree view item"""
        logger.debug(f"Item entered: {index.row()}")
        if not self.input_path or not self.destination_path:
            logger.debug("Paths not set, skipping preview")
            return

        file_path = self.file_model.filePath(index)
        logger.debug(f"File path from index: {file_path}")
        if not file_path or len(file_path) <= 3:
            return

        if sanitized_path := ut.sanitize_path(file_path):
            if self._is_image_file(sanitized_path):
                self._last_mouse_pos = QCursor.pos()
                self._pending_preview_path = sanitized_path
                logger.debug(f"Starting preview timer for: {sanitized_path}")
                self._hover_timer.start(300)

    def create_preview_job(self, folder_path: Path) -> Job:
        """Create a minimal job just for preview purposes"""
        def str_to_int(txt: str) -> int:
            return int(txt) if txt.isdigit() else 200

        return Job(
            width=str_to_int(self.controlWidget.widthLineEdit.text()),
            height=str_to_int(self.controlWidget.heightLineEdit.text()),
            fix_exposure_job=self.exposureCheckBox.isChecked(),
            multi_face_job=self.mfaceCheckBox.isChecked(),
            auto_tilt_job=self.tiltCheckBox.isChecked(),
            sensitivity=self.controlWidget.sensitivityDial.value(),
            face_percent=self.controlWidget.fpctDial.value(),
            gamma=self.controlWidget.gammaDial.value(),
            top=self.controlWidget.topDial.value(),
            bottom=self.controlWidget.bottomDial.value(),
            left=self.controlWidget.leftDial.value(),
            right=self.controlWidget.rightDial.value(),
            radio_buttons=self.controlWidget.radio_tuple,
            folder_path=folder_path,
            destination=None  # No destination needed for preview
        )


    def _show_preview(self):
        """Show the preview image"""
        logger.debug(f"_show_preview called. Pending path: {self._pending_preview_path}")
        if not self._pending_preview_path or not bool(self._last_mouse_pos):
            logger.debug("No pending preview or mouse position")
            return

        if not self.input_path or not self.destination_path:
            logger.debug("Required paths not set")
            return

        file_path = self._pending_preview_path

        if not self._is_image_file(file_path):
            logger.debug(f"Not an image file: {file_path}")
            return

        try:
            # Create job for preview
            folder_path = Path(self.input_path) if self.input_path and Path(self.input_path).exists() else None
            destination_path = (
                Path(self.destination_path) if self.destination_path and Path(self.destination_path).exists() else None
            )

            if not folder_path or not destination_path:
                logger.debug("Invalid folder or destination path")
                return

            job = self.create_job(
                self._get_function_type(),
                folder_path=folder_path,
                destination=destination_path
            )
            logger.debug(f"Created job, showing preview for: {file_path}")
            logger.debug(f"Mouse position: {self._last_mouse_pos}")
            # Show preview - this should work now
            self.image_preview.preview_file(
                file_path,
                self._last_mouse_pos,
                self.crop_worker.face_detection_tools[0],
                job
            )

        except Exception as e:
            logger.debug(f"Error in _show_preview: {e}")
            logger.exception("Exception occurred", exc_info=True)

    def _get_function_type(self):
        """Get the function type for this widget - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _get_function_type method")

    def _is_image_file(self, file_path: str) -> bool:
        """Check if a file is an image with enhanced logging"""
        path = Path(file_path)
        is_photo = file_manager.is_valid_type(path, FileCategory.PHOTO)
        is_tiff = file_manager.is_valid_type(path, FileCategory.TIFF)
        is_raw = file_manager.is_valid_type(path, FileCategory.RAW)
        result = is_photo or is_tiff or is_raw
        logger.debug(f"Is image file '{file_path}': {result} (photo: {is_photo}, tiff: {is_tiff}, raw: {is_raw})")
        return result

    def load_data(self) -> None:
        """Load data into the tree view from the selected folder"""
        logger.debug(f"Loading data for path: {self.input_path}")
        try:
            if not self.input_path:
                self.file_model.setRootPath("")
                self.treeView.setRootIndex(self.file_model.index(""))
                logger.debug("No input path, cleared tree view")
                return

            path = Path(self.input_path)
            if not path.exists() or not path.is_dir():
                self.file_model.setRootPath("")
                self.treeView.setRootIndex(self.file_model.index(""))
                logger.debug(f"Invalid path: {self.input_path}")
                return

            self.file_model.setRootPath(self.input_path)
            root_index = self.file_model.index(self.input_path)
            self.treeView.setRootIndex(root_index)
            logger.debug(f"Tree view loaded with {self.file_model.rowCount(root_index)} items")
        except Exception as e:
            logger.debug(f"Error loading data: {e}")
            with contextlib.suppress(Exception):
                self.file_model.setRootPath("")
                self.treeView.setRootIndex(self.file_model.index(""))

    # Rest of the methods remain the same...
    def connect_signals(self) -> None:
        """Connect widget signals to handlers"""
        self.cancelButton.clicked.connect(self.crop_worker.terminate)
        self.cancelButton.clicked.connect(self.handle_cancel_button_click)
        self.connect_crop_worker()

    def handle_cancel_button_click(self) -> None:
        """Handle cancel button click to properly update button states"""
        try:
            self.crop_worker.terminate()
        except Exception as e:
            logger.debug(f"Error terminating worker: {e}")
            self.cancelButton.setEnabled(False)
            self.cropButton.setEnabled(True)
            self.pulsing_indicator.reset()
            QApplication.processEvents()

    def enable_cancel_button(self) -> None:
        """Enable the cancel button immediately"""
        self.cancelButton.setEnabled(True)
        self.cancelButton.repaint()
        QApplication.processEvents()

    def connect_crop_worker(self) -> None:
        raise NotImplementedError("function must be implemented in subclasses.")

    def create_main_action_buttons(self, parent_frame: QFrame | None=None) -> tuple[QPushButton, QPushButton]:
        """Create crop and cancel buttons with consistent styling"""
        crop_button = self.create_main_button("cropButton", GuiIcon.CROP)
        crop_button.setParent(parent_frame)
        crop_button.setDisabled(True)

        cancel_button = self.create_main_button("cancelButton", GuiIcon.CANCEL)
        cancel_button.setParent(parent_frame)
        cancel_button.setDisabled(True)

        return crop_button, cancel_button

    def setup_main_crop_frame(self, parent_widget: QWidget) -> tuple[QFrame, QVBoxLayout]:
        """Create and set up the main crop frame with checkboxes and image widget"""
        frame = self.create_main_frame("frame")
        frame.setParent(parent_widget)
        vertical_layout = ut.setup_vbox("verticalLayout", frame)

        self.toggleCheckBox.setParent(frame)
        self.mfaceCheckBox.setParent(frame)
        self.tiltCheckBox.setParent(frame)
        self.exposureCheckBox.setParent(frame)

        checkbox_layout = ut.setup_hbox("horizontalLayout_1")
        self.setup_checkboxes_frame(checkbox_layout)
        vertical_layout.addLayout(checkbox_layout)

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
            self.crop_worker.started.disconnect()
            self.crop_worker.finished.disconnect()

        self.crop_worker.started.connect(lambda: ut.disable_widget(*widget_list))
        self.crop_worker.started.connect(lambda: ut.enable_widget(self.cancelButton))
        self.crop_worker.started.connect(self.pulsing_indicator.start_processing)

        self.crop_worker.finished.connect(lambda: ut.enable_widget(*widget_list))
        self.crop_worker.finished.connect(lambda: ut.disable_widget(self.cancelButton))
        self.crop_worker.finished.connect(self.pulsing_indicator.finish_processing)
        self.crop_worker.finished.connect(lambda: ut.show_message_box(self.destination))

    def run_batch_process(self, job: Job, *,
                          function: Callable[..., Any],
                          reset_worker_func: Callable[..., Any]) -> None:
        """Run a batch processing operation using threading instead of multiprocessing"""
        reset_worker_func()
        self.pulsing_indicator.reset()
        self.pulsing_indicator.start_processing()
        self.cropButton.setEnabled(False)
        self.cropButton.repaint()
        self.enable_cancel_button()
        QApplication.processEvents()

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

    def _clear_hover_preview_cache(self):
        """Clear the hover preview cache."""
        self.image_preview.clear_cache()

    # Clean up when tab is hidden or destroyed
    def hideEvent(self, event):
        """Hide preview when tab is hidden"""
        self._handle_hover_end()
        super().hideEvent(event)

    def closeEvent(self, event):
        """Clean up when widget is closed"""
        self._handle_hover_end()
        self.image_preview.clear_cache()
        self.image_preview.close()
        super().closeEvent(event)
