from pathlib import Path

from PyQt6 import QtCore, QtWidgets

from core.croppers import PhotoCropper
from core.enums import FunctionType
from file_types import registry
from line_edits import PathLineEdit, PathType
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
        
        # Override the input line edit to use the correct path type
        self.inputLineEdit = self.create_str_line_edit("inputLineEdit", PathType.IMAGE)
        
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
        self.inputLineEdit.setParent(self)
        self.inputButton.setParent(self)
        self.horizontalLayout_2.addWidget(self.inputLineEdit)
        self.horizontalLayout_2.addWidget(self.inputButton)
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
        self.destinationLineEdit.setParent(self)
        self.destinationButton.setParent(self)
        self.setup_destination_layout(self.horizontalLayout_3)
        self.verticalLayout_100.addLayout(self.horizontalLayout_3)

    def connect_signals(self) -> None:
        """Connect widget signals to handlers"""
        # Button connections
        self.inputButton.clicked.connect(lambda: self.open_path(self.inputLineEdit))
        self.destinationButton.clicked.connect(lambda: self.open_path(self.destinationLineEdit))
        self.cropButton.clicked.connect(lambda: self.crop_photo())
        
        # Connect input widgets for validation
        self.connect_input_widgets(
            self.inputLineEdit, 
            self.controlWidget.widthLineEdit,
            self.controlWidget.heightLineEdit, 
            self.destinationLineEdit, 
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
        self.inputLineEdit.setPlaceholderText(
            QtCore.QCoreApplication.translate("self", "Choose the image you want to crop", None)
        )
        self.inputButton.setText(QtCore.QCoreApplication.translate("self", "Open Image", None))
        self.destinationLineEdit.setPlaceholderText(
            QtCore.QCoreApplication.translate("self", "Choose where you want to save the cropped image", None)
        )
        self.cropButton.setText("")

    def open_path(self, line_edit: PathLineEdit) -> None:
        """Open file/folder selection dialog based on line edit type"""
        if line_edit is self.inputLineEdit:
            f_name, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, 'Open File', 
                registry.get_default_dir("photo").as_posix(), 
                registry.get_filter_string("photo")
            )
            line_edit.setText(f_name)
        elif line_edit is self.destinationLineEdit:
            super().open_path(line_edit)

    def disable_buttons(self) -> None:
        """Enable/disable buttons based on input validation"""
        ut.change_widget_state(
            ut.all_filled(
                self.inputLineEdit, 
                self.destinationLineEdit, 
                self.controlWidget.widthLineEdit,
                self.controlWidget.heightLineEdit
            ),
            self.cropButton
        )

    def crop_photo(self) -> None:
        """Execute the photo cropping operation"""
        def execute_crop():
            job = self.create_job(
                FunctionType.PHOTO,
                photo_path=Path(self.inputLineEdit.text()),
                destination=Path(self.destinationLineEdit.text())
            )
            self.crop_worker.crop(Path(self.inputLineEdit.text()), job)

        # Check if source and destination are the same and warn if needed
        if Path(self.inputLineEdit.text()).parent == Path(self.destinationLineEdit.text()):
            match ut.show_warning(FunctionType.PHOTO):
                case QtWidgets.QMessageBox.StandardButton.Yes:
                    execute_crop()
                case _:
                    return
        else:
            execute_crop()
