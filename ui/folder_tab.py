from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from core.croppers import FolderCropper
from core.enums import FunctionType
from file_types import file_manager, FileCategory
from ui import utils as ut
from .batch_tab import UiBatchCropWidget


class UiFolderTabWidget(UiBatchCropWidget):
    """Folder tab widget with enhanced inheritance from the batch crop_from_path widget"""

    def __init__(self, crop_worker: FolderCropper, object_name: str, parent: QtWidgets.QWidget) -> None:
        """Initialize the folder tab widget"""
        super().__init__(crop_worker, object_name, parent)
        
        # Path storage fields
        self.input_path = ""
        self.destination_path = ""

        # Create file model for the tree view
        self.file_model = QtGui.QFileSystemModel(self)
        self.file_model.setFilter(QtCore.QDir.Filter.NoDotAndDotDot | QtCore.QDir.Filter.Files)
        p_types = file_manager.get_extensions(FileCategory.PHOTO) | file_manager.get_extensions(FileCategory.TIFF) | file_manager.get_extensions(FileCategory.RAW)
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

        input_layout = ut.setup_hbox("horizontalLayout_4")
        input_layout.setStretch(0, 1)

        self.verticalLayout_200.addLayout(input_layout)

        # Main frame with image and controls
        frame, verticalLayout = self.setup_main_crop_frame(self.page_1)

        # Crop and cancel buttons
        buttonLayout = ut.setup_hbox("horizontalLayout_2")

        # self.cropButton, self.cancelButton = self.create_main_action_buttons(frame)
        self.cropButton.setParent(frame)
        self.cancelButton.setParent(frame)
        buttonLayout.addWidget(self.cropButton)
        buttonLayout.addWidget(self.cancelButton)

        verticalLayout.addLayout(buttonLayout)

        # Progress bar
        self.progressBar.setParent(frame)
        verticalLayout.addWidget(self.progressBar)

        self.verticalLayout_200.addWidget(frame)

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
        """Open file/folder selection dialog with updated string-based approach"""
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

            self.file_model.setRootPath(self.input_path)
            self.treeView.setRootIndex(self.file_model.index(self.input_path))
        except (IndexError, FileNotFoundError, ValueError, AttributeError):
            return

    def connect_crop_worker(self) -> None:
        """Connect the signals from the crop_from_path worker to UI handlers"""
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
            # Manually disable crop_from_path button right away
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
