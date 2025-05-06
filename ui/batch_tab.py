
import contextlib
import threading
from pathlib import Path
from typing import Optional, Callable, Any

from PyQt6 import QtWidgets

from core import Job
from core.croppers import BatchCropper
from ui import utils as ut
from ui.pulsing_indicator import PulsingProgressIndicator
from .crop_widget import UiCropWidget
from .enums import GuiIcon


class UiBatchCropWidget(UiCropWidget):
    """
    Intermediate base widget class for batch cropping functionality.
    Provides common UI components and behaviors for folder and mapping tabs.
    """

    PROGRESSBAR_STEPS: int = 1_000

    def __init__(self, crop_worker: BatchCropper, object_name: str, parent: QtWidgets.QWidget) -> None:
        """Initialize the batch crop widget with pulsing progress indicator"""
        super().__init__(parent, object_name)
        self.crop_worker = crop_worker
        
        # Create pulsing progress indicator
        self.pulsing_indicator = PulsingProgressIndicator(self)
        self.pulsing_indicator.setObjectName("pulsingIndicator")
        
        # Create common UI elements for batch operations
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
        QtWidgets.QApplication.processEvents()

    def enable_cancel_button(self) -> None:
        """Enable the cancel button immediately"""
        self.cancelButton.setEnabled(True)
        self.cancelButton.repaint()
        QtWidgets.QApplication.processEvents()

    def connect_crop_worker(self) -> None:
        raise NotImplementedError("function must be implemented in subclasses.")

    def create_main_action_buttons(self, parent_frame: Optional[QtWidgets.QFrame]=None) -> tuple[QtWidgets.QPushButton, QtWidgets.QPushButton]:
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

    def setup_main_crop_frame(self, parent_widget: QtWidgets.QWidget) -> tuple[QtWidgets.QFrame, QtWidgets.QVBoxLayout]:
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
    def cancel_button_operation(cancel_button: QtWidgets.QPushButton, *crop_buttons: QtWidgets.QPushButton) -> None:
        """Handle cancel button operations"""
        cancel_button.setDisabled(True)
        for crop_button in crop_buttons:
            crop_button.setEnabled(True)

    def connect_crop_worker_signals(self, *widget_list: QtWidgets.QWidget) -> None:
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
        QtWidgets.QApplication.processEvents()
        
        # Use Thread instead of Process to avoid pickling issues
        thread = threading.Thread(target=function, args=(job,), daemon=True)
        thread.start()

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
