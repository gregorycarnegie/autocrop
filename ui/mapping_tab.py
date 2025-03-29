from pathlib import Path
from typing import Any, Optional

import polars as pl
from PyQt6 import QtCore, QtWidgets

from core import DataFrameModel
from core import processing as prc
from core.croppers import MappingCropper
from core.enums import FunctionType
from file_types import registry
from line_edits import LineEditState, PathType
from ui import utils as ut
from .batch_tab import UiBatchCropWidget
from .enums import GuiIcon


class UiMappingTabWidget(UiBatchCropWidget):
    """Mapping tab widget with enhanced inheritance from batch crop widget"""

    def __init__(self, crop_worker: MappingCropper, object_name: str, parent: QtWidgets.QWidget) -> None:
        """Initialize the mapping tab widget"""
        super().__init__(crop_worker, object_name, parent)

        # Data model
        self.model: Optional[DataFrameModel] = None
        self.data_frame: Optional[pl.DataFrame] = None

        # Create mapping-specific widgets
        self.tableLineEdit = self.create_str_line_edit("tableLineEdit", PathType.TABLE)
        self.tableButton = self.create_nav_button("tableButton")
        self.comboBox_1 = QtWidgets.QComboBox()
        self.comboBox_2 = QtWidgets.QComboBox()
        self.comboBox_3 = QtWidgets.QComboBox()
        self.comboBox_4 = QtWidgets.QComboBox()
        self.tableView = QtWidgets.QTableView()

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
        """Set up the main layout structure"""
        # ---- Page 1: Crop View ----
        # Grid layout for input fields
        gridLayout = QtWidgets.QGridLayout()
        gridLayout.setObjectName("gridLayout")

        # Input folder selection
        self.inputLineEdit.setParent(self.page_1)
        self.inputButton.setParent(self.page_1)
        self.inputButton.setIcon(self.folder_icon)

        gridLayout.addWidget(self.inputLineEdit, 0, 0, 1, 1)
        gridLayout.addWidget(self.inputButton, 0, 1, 1, 1)

        # Table file selection
        self.tableLineEdit.setParent(self.page_1)
        self.tableButton.setParent(self.page_1)
        icon = ut.create_button_icon(GuiIcon.EXCEL)
        self.tableButton.setIcon(icon)

        gridLayout.addWidget(self.tableLineEdit, 1, 0, 1, 1)
        gridLayout.addWidget(self.tableButton, 1, 1, 1, 1)

        # Set column properties
        gridLayout.setColumnStretch(0, 20)
        gridLayout.setColumnStretch(1, 1)
        gridLayout.setColumnMinimumWidth(0, 20)
        gridLayout.setColumnMinimumWidth(1, 1)

        self.verticalLayout_200.addLayout(gridLayout)

        # Main frame with image and controls
        frame, verticalLayout = self.setup_main_crop_frame(self.page_1)

        # Comboboxes, crop and cancel buttons
        buttonLayout = ut.setup_hbox("horizontalLayout_1")

        # Setup comboboxes
        ut.setip_combobox(self.comboBox_1, buttonLayout, self.size_policy_expand_fixed, "comboBox_1")
        ut.setip_combobox(self.comboBox_2, buttonLayout, self.size_policy_expand_fixed, "comboBox_2")

        # Crop and cancel buttons
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
        self.setup_destination_layout(self.horizontalLayout_3)
        self.verticalLayout_200.addLayout(self.horizontalLayout_3)

        # Add page to toolbox
        self.toolBox.addItem(self.page_1, "Crop View")

        # ---- Page 2: Table View ----
        # Table view
        self.tableView.setObjectName("tableView")
        self.tableView.setParent(self.page_2)
        self.verticalLayout_300.addWidget(self.tableView)

        # Comboboxes for column selection on table view
        comboLayout = ut.setup_hbox("horizontalLayout_4")

        ut.setup_combobox(self.comboBox_3, comboLayout, self.size_policy_expand_fixed, "comboBox_3")
        ut.setup_combobox(self.comboBox_4, comboLayout, self.size_policy_expand_fixed, "comboBox_4")

        self.verticalLayout_300.addLayout(comboLayout)

        # Add page to toolbox
        self.toolBox.addItem(self.page_2, "Table View")

        # Add toolbox to main layout
        self.verticalLayout_100.addWidget(self.toolBox)

    def connect_signals(self) -> None:
        """Connect widget signals to handlers"""
        # Button connections
        self.inputButton.clicked.connect(lambda: self.open_path(self.inputLineEdit))
        self.tableButton.clicked.connect(lambda: self.open_table())
        # self.destinationButton.clicked.connect(lambda: self.open_path(self.destinationLineEdit))
        self.cropButton.clicked.connect(lambda: self.mapping_process())
        self.cancelButton.clicked.connect(lambda: self.crop_worker.terminate())
        self.cancelButton.clicked.connect(lambda: self.cancel_button_operation(self.cancelButton, self.cropButton))

        # Combobox synchronization
        self.comboBox_1.currentTextChanged.connect(lambda text: self.comboBox_3.setCurrentText(text))
        self.comboBox_2.currentTextChanged.connect(lambda text: self.comboBox_4.setCurrentText(text))
        self.comboBox_3.currentTextChanged.connect(lambda text: self.comboBox_1.setCurrentText(text))
        self.comboBox_4.currentTextChanged.connect(lambda text: self.comboBox_2.setCurrentText(text))

        # Register button dependencies with the TabStateManager
        self.tab_state_manager.register_button_dependencies(
            self.cropButton,
            {
                self.inputLineEdit,
                self.tableLineEdit,
                self.destinationLineEdit,
                self.comboBox_1,
                self.comboBox_2,
                self.controlWidget.widthLineEdit,
                self.controlWidget.heightLineEdit
            }
        )
        
        # Connect all input widgets for validation tracking
        self.tab_state_manager.connect_widgets(
            self.inputLineEdit,
            self.controlWidget.widthLineEdit,
            self.controlWidget.heightLineEdit, 
            self.destinationLineEdit,
            self.tableLineEdit,
            self.comboBox_1,
            self.comboBox_2,
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

        # Connect crop worker signals
        self.connect_crop_worker()

    def retranslateUi(self) -> None:
        """Update UI text elements"""
        super().retranslateUi()
        self.inputLineEdit.setPlaceholderText(
            QtCore.QCoreApplication.translate("self", "Choose the folder you want to crop", None)
        )
        self.inputButton.setText(QtCore.QCoreApplication.translate("self", "Select Folder", None))
        self.tableLineEdit.setPlaceholderText(
            QtCore.QCoreApplication.translate("self", "Choose the Excel or CSV file with the mapping", None)
        )
        self.tableButton.setText(QtCore.QCoreApplication.translate("self", "Open File", None))
        self.destinationLineEdit.setPlaceholderText(
            QtCore.QCoreApplication.translate("self", "Choose where you want to save the cropped images", None)
        )
        self.destinationButton.setText(QtCore.QCoreApplication.translate("self", "Destination Folder", None))
        self.comboBox_1.setPlaceholderText(QtCore.QCoreApplication.translate("self", "Filename column", None))
        self.comboBox_2.setPlaceholderText(QtCore.QCoreApplication.translate("self", "Mapping column", None))
        self.comboBox_3.setPlaceholderText(QtCore.QCoreApplication.translate("self", "Filename column", None))
        self.comboBox_4.setPlaceholderText(QtCore.QCoreApplication.translate("self", "Mapping column", None))
        self.cropButton.setText("")
        self.cancelButton.setText("")
        self.toolBox.setItemText(self.toolBox.indexOf(self.page_1),
                                 QtCore.QCoreApplication.translate("self", "Crop View", None))
        self.toolBox.setItemText(self.toolBox.indexOf(self.page_2),
                                 QtCore.QCoreApplication.translate("self", "Table View", None))

    def open_table(self) -> None:
        """Open a table file dialog"""
        f_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Open File',
            registry.get_default_dir("photo").as_posix(),
            registry.get_filter_string("table")
        )
        self.tableLineEdit.setText(f_name)

        # Load the table if valid
        if self.tableLineEdit.state is LineEditState.INVALID_INPUT:
            return

        data = prc.open_table(Path(f_name))
        self.process_data(data)

    def process_data(self, data: Any) -> None:
        """Process the loaded data"""
        try:
            # Ensure data is a polars DataFrame
            if not isinstance(data, pl.DataFrame):
                return

            self.data_frame = data
            self.model = DataFrameModel(self.data_frame)
            self.tableView.setModel(self.model)

            # Update combo boxes with column names
            columns = self.data_frame.columns
            for combo in [self.comboBox_1, self.comboBox_2, self.comboBox_3, self.comboBox_4]:
                combo.clear()
                combo.addItems(columns)

        except Exception as e:
            print(f"Error processing data: {e}")

    def connect_crop_worker(self) -> None:
        """Connect the signals from the crop worker to UI handlers"""
        widget_list = (self.controlWidget.widthLineEdit, self.controlWidget.heightLineEdit,
                       self.controlWidget.sensitivityDial, self.controlWidget.fpctDial, self.controlWidget.gammaDial,
                       self.controlWidget.topDial, self.controlWidget.bottomDial, self.controlWidget.leftDial,
                       self.controlWidget.rightDial, self.inputLineEdit, self.destinationLineEdit,
                       self.tableButton, self.destinationButton, self.inputButton, self.tableLineEdit,
                       self.comboBox_1, self.comboBox_2, self.controlWidget.radioButton_none,
                       self.controlWidget.radioButton_bmp, self.controlWidget.radioButton_jpg,
                       self.controlWidget.radioButton_png, self.controlWidget.radioButton_tiff,
                       self.controlWidget.radioButton_webp, self.cropButton, self.exposureCheckBox,
                       self.mfaceCheckBox, self.tiltCheckBox)

        self.connect_crop_worker_signals(widget_list)

    def mapping_process(self) -> None:
        """Begin the mapping cropping process"""
        self.crop_worker.show_message_box = False

        def execute_crop():
            job = self.create_job(
                FunctionType.MAPPING,
                folder_path=Path(self.inputLineEdit.text()),
                destination=Path(self.destinationLineEdit.text()),
                table=self.data_frame,
                column1=self.comboBox_1,
                column2=self.comboBox_2
            )
            self.run_batch_process(job,
                                   function=self.crop_worker.crop,
                                   reset_worker_func=lambda: self.crop_worker.reset_task())

        # Check if source and destination are the same and warn if needed
        self.check_source_destination_same(
            self.inputLineEdit.text(),
            self.destinationLineEdit.text(),
            FunctionType.MAPPING,
            execute_crop
        )
