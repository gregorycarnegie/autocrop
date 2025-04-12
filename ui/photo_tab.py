from pathlib import Path

from PyQt6 import QtCore, QtWidgets

from core.croppers import PhotoCropper
from core.enums import FunctionType
from file_types import file_manager, FileCategory
from ui import utils as ut
from .crop_widget import UiCropWidget
from .enums import GuiIcon


class UiPhotoTabWidget(UiCropWidget):
    """Photo tab widget with enhanced inheritance from the base crop widget"""
    
    def __init__(self, crop_worker: PhotoCropper, object_name: str, parent: QtWidgets.QWidget) -> None:
        """Initialize the photo tab widget"""
        super().__init__(parent, object_name)
        self.crop_worker = crop_worker
        self.selection_state = self.SELECTED
    
        # Path storage fields
        self.input_path = ""
        self.destination_path = ""
        
        # Set up the main layout structure
        self.setup_layouts()
        
        # Connect signals
        self.connect_signals()
        
        # Set initial UI text
        self.retranslateUi()
        
        QtCore.QMetaObject.connectSlotsByName(self)

    def setup_layouts(self) -> None:
        """Set up the main layout structure"""
        # Input file selection
        self.horizontalLayout_2.setStretch(0, 1)
        self.verticalLayout_100.addLayout(self.horizontalLayout_2)
        
        # Main frame with image and controls
        frame = self.create_main_frame("frame")
        verticalLayout = ut.setup_vbox("verticalLayout_4", frame)
        
        # Checkbox section
        self.toggleCheckBox.setParent(frame)
        self.mfaceCheckBox.setParent(frame)
        self.tiltCheckBox.setParent(frame)
        self.exposureCheckBox.setParent(frame)
        self.setup_checkboxes_frame(self.horizontalLayout_1)
        verticalLayout.addLayout(self.horizontalLayout_1)
        
        # Image widget
        self.imageWidget.setParent(frame)
        verticalLayout.addWidget(self.imageWidget)
        
        # Crop button
        self.cropButton = self.create_main_button("cropButton", GuiIcon.CROP)
        self.cropButton.setParent(frame)
        self.cropButton.setDisabled(True)
        verticalLayout.addWidget(self.cropButton)
        
        # Add frame to main layout
        self.verticalLayout_100.addWidget(frame)
        
        # Destination selection
        self.verticalLayout_100.addLayout(self.horizontalLayout_3)

    def connect_signals(self) -> None:
        """Connect widget signals to handlers"""
        # Button connections
        self.cropButton.clicked.connect(self.crop_photo)
        
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
            # self.inputLineEdit,
            self.controlWidget.widthLineEdit,
            self.controlWidget.heightLineEdit, 
            # self.destinationLineEdit,
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

    def open_path(self, line_edit_type: str) -> None:
        """
        Open file/folder selection dialog
        Since we removed the line edits, this now takes a string identifier
        """
        if line_edit_type == "input":
            f_name, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, 'Open File',
                file_manager.get_default_directory(FileCategory.PHOTO).as_posix(),
                file_manager.get_filter_string(FileCategory.PHOTO)
            )
            # Validate the file exists and is accessible
            if f_name := ut.sanitize_path(f_name):
                # Update the input path
                self.input_path = f_name

                # Also update the main window's unified address bar if this is the active tab
                main_window = self.parent().parent().parent()
                if main_window.function_tabWidget.currentIndex() == FunctionType.PHOTO:
                    main_window.unified_address_bar.setText(f_name)

        elif line_edit_type == "destination":
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
                if main_window.function_tabWidget.currentIndex() == FunctionType.PHOTO:
                    main_window.destination_input.setText(f_name)

    def crop_photo(self) -> None:
        """Execute the photo cropping operation"""
        def execute_crop():
            job = self.create_job(
                FunctionType.PHOTO,
                photo_path=Path(self.input_path) if self.input_path else None,
                destination=Path(self.destination_path) if self.destination_path else None
            )
            self.crop_worker.crop(Path(self.input_path), job)

        # Check if source and destination are the same and warn if needed
        if self.input_path and self.destination_path:
            if Path(self.input_path).parent == Path(self.destination_path):
                match ut.show_warning(FunctionType.PHOTO):
                    case QtWidgets.QMessageBox.StandardButton.Yes:
                        execute_crop()
                    case _:
                        return
            else:
                execute_crop()
