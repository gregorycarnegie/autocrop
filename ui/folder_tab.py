from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from core.croppers import FolderCropper
from core.enums import FunctionType
from file_types import registry
from ui import utils as ut
from .batch_tab import UiBatchCropWidget


class UiFolderTabWidget(UiBatchCropWidget):
    """Folder tab widget with enhanced inheritance from the batch crop widget"""

    def __init__(self, crop_worker: FolderCropper, object_name: str, parent: QtWidgets.QWidget) -> None:
        """Initialize the folder tab widget"""
        super().__init__(crop_worker, object_name, parent)

        # Create file model for the tree view
        self.file_model = QtGui.QFileSystemModel(self)
        self.file_model.setFilter(QtCore.QDir.Filter.NoDotAndDotDot | QtCore.QDir.Filter.Files)
        p_types = registry.get_extensions('photo') | registry.get_extensions('tiff') | registry.get_extensions('raw')
        file_filter = np.array([f'*{file}' for file in p_types])
        self.file_model.setNameFilters(file_filter)

        # Set up the main layout structure
        self.setup_layouts()
        self.treeView: Optional[QtWidgets.QTreeView] = None

        # Connect signals
        self.connect_signals()

        # Set initial UI text
        self.retranslateUi()

        # Set initial toolbox page
        self.toolBox.setCurrentIndex(0)

        QtCore.QMetaObject.connectSlotsByName(self)

    def setup_layouts(self) -> None:
        """Set up the main layout structure"""
        # ---- Page 1: Crop View ----
        # Input file selection
        self.inputLineEdit.setParent(self.page_1)
        self.inputButton.setParent(self.page_1)

        input_layout = ut.setup_hbox("horizontalLayout_4")
        input_layout.addWidget(self.inputLineEdit)
        input_layout.addWidget(self.inputButton)
        input_layout.setStretch(0, 1)

        self.verticalLayout_200.addLayout(input_layout)

        # Main frame with image and controls
        frame, verticalLayout = self.setup_main_crop_frame(self.page_1)

        # Crop and cancel buttons
        buttonLayout = ut.setup_hbox("horizontalLayout_2")

        self.cropButton, self.cancelButton = self.create_main_action_buttons(frame)
        buttonLayout.addWidget(self.cropButton)
        buttonLayout.addWidget(self.cancelButton)

        verticalLayout.addLayout(buttonLayout)

        # Progress bar
        self.progressBar.setParent(frame)
        verticalLayout.addWidget(self.progressBar)

        self.verticalLayout_200.addWidget(frame)

        # Destination selection
        self.destinationLineEdit.setParent(self.page_1)
        self.destinationButton.setParent(self.page_1)

        destLayout = ut.setup_hbox("horizontalLayout_3")
        destLayout.addWidget(self.destinationLineEdit)
        destLayout.addWidget(self.destinationButton)
        destLayout.setStretch(0, 1)

        self.verticalLayout_200.addLayout(destLayout)

        # Add page to toolbox
        self.toolBox.addItem(self.page_1, "Crop View")

        # ---- Page 2: Folder View ----
        self.treeView = QtWidgets.QTreeView(self.page_2)
        self.treeView.setObjectName("treeView")
        self.treeView.setModel(self.file_model)

        self.verticalLayout_300.addWidget(self.treeView)

        # Add page to toolbox
        self.toolBox.addItem(self.page_2, "Folder View")

        # Add toolbox to main layout
        self.verticalLayout_100.addWidget(self.toolBox)

    def connect_signals(self) -> None:
        """Connect widget signals to handlers"""
        # Button connections
        self.inputButton.clicked.connect(lambda: self.open_path(self.inputLineEdit))
        self.destinationButton.clicked.connect(lambda: self.open_path(self.destinationLineEdit))
        self.cropButton.clicked.connect(lambda: self.folder_process())
        self.cancelButton.clicked.connect(lambda: self.crop_worker.terminate())
        self.cancelButton.clicked.connect(lambda: self.cancel_button_operation(self.cancelButton, self.cropButton))

        # Input widgets for validation
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

        # Input line edit also updates the tree view
        self.inputLineEdit.textChanged.connect(lambda: self.load_data())

        # Connect crop worker signals
        self.connect_crop_worker()

    def retranslateUi(self) -> None:
        """Update UI text elements"""
        super().retranslateUi()
        self.inputLineEdit.setPlaceholderText(
            QtCore.QCoreApplication.translate("self", "Choose the folder you want to crop", None)
        )
        self.inputButton.setText(QtCore.QCoreApplication.translate("self", "Select Folder", None))
        self.destinationLineEdit.setPlaceholderText(
            QtCore.QCoreApplication.translate("self", "Choose where you want to save the cropped images", None)
        )
        self.destinationButton.setText(QtCore.QCoreApplication.translate("self", "Destination Folder", None))
        self.cropButton.setText("")
        self.cancelButton.setText("")
        self.toolBox.setItemText(self.toolBox.indexOf(self.page_1),
                                 QtCore.QCoreApplication.translate("self", "Crop View", None))
        self.toolBox.setItemText(self.toolBox.indexOf(self.page_2),
                                 QtCore.QCoreApplication.translate("self", "Folder View", None))

    def open_path(self, line_edit) -> None:
        """Open file/folder selection dialog for a path"""
        super().open_path(line_edit)
        if line_edit is self.inputLineEdit:
            self.load_data()

    def load_data(self) -> None:
        """Load data into the tree view from the selected folder"""
        try:
            f_name = self.inputLineEdit.text()
            self.file_model.setRootPath(f_name)
            self.treeView.setRootIndex(self.file_model.index(f_name))
        except (IndexError, FileNotFoundError, ValueError, AttributeError):
            return

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

    def connect_crop_worker(self) -> None:
        """Connect the signals from the crop worker to UI handlers"""
        widget_list = (self.controlWidget.widthLineEdit, self.controlWidget.heightLineEdit,
                       self.controlWidget.sensitivityDial, self.controlWidget.fpctDial, self.controlWidget.gammaDial,
                       self.controlWidget.topDial, self.controlWidget.bottomDial, self.controlWidget.leftDial,
                       self.controlWidget.rightDial, self.inputLineEdit, self.destinationLineEdit,
                       self.destinationButton, self.inputButton, self.controlWidget.radioButton_none,
                       self.controlWidget.radioButton_bmp, self.controlWidget.radioButton_jpg,
                       self.controlWidget.radioButton_png, self.controlWidget.radioButton_tiff,
                       self.controlWidget.radioButton_webp, self.cropButton, self.exposureCheckBox, self.mfaceCheckBox,
                       self.tiltCheckBox)

        self.connect_crop_worker_signals(widget_list)

    def folder_process(self) -> None:
        """Begin the folder cropping process"""
        self.crop_worker.show_message_box = False

        def execute_crop():
            job = self.create_job(
                FunctionType.FOLDER,
                folder_path=Path(self.inputLineEdit.text()),
                destination=Path(self.destinationLineEdit.text())
            )
            self.run_batch_process(job,
                                   function=self.crop_worker.crop,
                                   reset_worker_func=lambda: self.crop_worker.reset_task())

        # Check if source and destination are the same and warn if needed
        self.check_source_destination_same(
            self.inputLineEdit.text(),
            self.destinationLineEdit.text(),
            FunctionType.FOLDER,
            execute_crop
        )
