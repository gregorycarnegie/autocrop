from functools import partial
from multiprocessing import Process
from pathlib import Path
from typing import Optional, Callable, Any

from PyQt6 import QtCore, QtWidgets

from core import Job
from core.croppers import BatchCropper
from ui import utils as ut
from .crop_widget import UiCropWidget
from .enums import GuiIcon


class UiBatchCropWidget(UiCropWidget):
    """
    Intermediate base widget class for batch cropping functionality.
    Provides common UI components and behaviors for folder and mapping tabs.
    """

    PROGRESSBAR_STEPS: int = 1_000

    def __init__(self, crop_worker: BatchCropper, object_name: str, parent: QtWidgets.QWidget) -> None:
        """Initialize the batch crop widget with common components"""
        super().__init__(parent, object_name)
        self.crop_worker = crop_worker

        # Create common UI elements for batch operations
        self.progressBar = self.create_progress_bar("progressBar")
        self.toolBox = QtWidgets.QToolBox(self)
        self.toolBox.setObjectName("toolBox")

        # Create pages for the toolbox
        self.page_1 = QtWidgets.QWidget()
        self.page_1.setObjectName("page_1")
        self.page_2 = QtWidgets.QWidget()
        self.page_2.setObjectName("page_2")

        # Set up page layouts
        self.verticalLayout_200 = ut.setup_vbox("verticalLayout_200", self.page_1)
        self.verticalLayout_300 = ut.setup_vbox("verticalLayout_300", self.page_2)

        # Buttons that all batch processors need
        self.cropButton , self.cancelButton = self.create_main_action_buttons()

    def connect_signals(self) -> None:
        """Connect widget signals to handlers"""
        # Button connections
        self.cropButton.clicked.connect(partial(self.cancelButton.setDisabled, True))
        self.cancelButton.clicked.connect(self.crop_worker.terminate)
        self.cancelButton.clicked.connect(partial(self.cancel_button_operation, self.cancelButton, self.cropButton))
        self.connect_crop_worker()

    def connect_crop_worker(self) -> None:
        raise NotImplementedError("function must be implemented in subclasses.")

    def worker(self, *args: Any) -> None:
        """Worker function to be overridden in subclasses"""
        raise NotImplementedError("Worker function must be implemented in subclasses.")

    def create_progress_bar(self, name: str, parent: Optional[QtWidgets.QWidget] = None) -> QtWidgets.QProgressBar:
        """Create a progress bar with consistent styling"""
        progress_bar = QtWidgets.QProgressBar() if parent is None else QtWidgets.QProgressBar(parent)
        progress_bar.setObjectName(name)
        progress_bar.setMinimumSize(QtCore.QSize(0, 15))
        progress_bar.setMaximumSize(QtCore.QSize(16_777_215, 15))
        progress_bar.setRange(0, self.PROGRESSBAR_STEPS)
        progress_bar.setValue(0)
    
        # Apply styling
        progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #cccccc;
                border-radius: 3px;
                background-color: #f0f0f0;
                text-align: center;
                color: #505050;
            }
            
            QProgressBar::chunk {
                background-color: #4285f4;
                width: 10px;
                margin: 0.5px;
            }
        """)

        progress_bar.setTextVisible(False)
        return progress_bar

    def create_main_action_buttons(self, parent_frame: Optional[QtWidgets.QFrame]=None) -> tuple[QtWidgets.QPushButton, QtWidgets.QPushButton]:
        """Create crop and cancel buttons with consistent styling"""
        # Crop button
        crop_button = self.create_main_button("cropButton", GuiIcon.CROP)
        crop_button.setParent(parent_frame)
        crop_button.setDisabled(True)

        # Cancel button
        cancel_button = self.create_main_button("cancelButton", GuiIcon.CANCEL)
        cancel_button.setParent(parent_frame)
        cancel_button.setDisabled(True)

        return crop_button, cancel_button

    def setup_main_crop_frame(self, parent_widget: QtWidgets.QWidget) -> tuple[QtWidgets.QFrame, QtWidgets.QVBoxLayout]:
        """Create and set up the main crop frame with checkboxes and image widget"""
        frame = self.create_main_frame("frame")
        frame.setParent(parent_widget)
        verticalLayout = ut.setup_vbox("verticalLayout", frame)

        # Checkbox section
        self.toggleCheckBox.setParent(frame)
        self.mfaceCheckBox.setParent(frame)
        self.tiltCheckBox.setParent(frame)
        self.exposureCheckBox.setParent(frame)

        checkboxLayout = ut.setup_hbox("horizontalLayout_1")
        self.setup_checkboxes_frame(checkboxLayout)
        verticalLayout.addLayout(checkboxLayout)

        # Image widget
        self.imageWidget.setParent(frame)
        verticalLayout.addWidget(self.imageWidget)

        return frame, verticalLayout

    def update_progress(self, x: int, y: int) -> None:
        """Update the progress bar based on crop worker progress"""
        self.progressBar.setValue(int(self.PROGRESSBAR_STEPS * x / y))
        QtWidgets.QApplication.processEvents()

    @staticmethod
    def cancel_button_operation(cancel_button: QtWidgets.QPushButton, *crop_buttons: QtWidgets.QPushButton) -> None:
        """Handle cancel button operations"""
        cancel_button.setDisabled(True)
        for crop_button in crop_buttons:
            crop_button.setEnabled(True)

    def connect_crop_worker_signals(self, widget_list: tuple) -> None:
        """Connect the signals from the crop worker to UI handlers"""
        # Batch start connection
        self.crop_worker.started.connect(lambda: ut.disable_widget(*widget_list))
        self.crop_worker.started.connect(lambda: ut.enable_widget(self.cancelButton))

        # Batch end connection
        self.crop_worker.finished.connect(lambda: ut.enable_widget(*widget_list))
        self.crop_worker.finished.connect(lambda: ut.disable_widget(self.cancelButton))
        self.crop_worker.finished.connect(lambda: ut.show_message_box(self.destination))
        self.crop_worker.progress.connect(self.update_progress)

    @staticmethod
    def run_batch_process(job: Job, *,
                          function: Callable[..., Any],
                          reset_worker_func: Callable[..., Any]) -> None:
        """Run a batch processing operation"""
        reset_worker_func()
        process = Process(target=function, daemon=True, args=(job,))
        process.run()

    @staticmethod
    def check_source_destination_same(source_path: str, dest_path: str,
                                      function_type, process_func: Callable) -> None:
        """Check if source and destination are the same and warn if needed"""
        if Path(source_path) == Path(dest_path):
            match ut.show_warning(function_type):
                case QtWidgets.QMessageBox.StandardButton.Yes:
                    process_func()
                case _:
                    return
        else:
            process_func()
