from pathlib import Path
from typing import Optional

import polars as pl
from PyQt6 import QtCore, QtWidgets

from core import DataFrameModel
from core import processing as prc
from core.croppers import MappingCropper
from core.enums import FunctionType
from file_types import file_manager, FileCategory
from ui import utils as ut
from .batch_tab import UiBatchCropWidget


class UiMappingTabWidget(UiBatchCropWidget):
    """Mapping tab widget with enhanced inheritance from batch crop_from_path widget"""

    def __init__(self, crop_worker: MappingCropper, object_name: str, parent: QtWidgets.QWidget) -> None:
        """Initialize the mapping tab widget"""
        super().__init__(crop_worker, object_name, parent)
        
        # Path storage fields
        self.input_path = ""         # Folder path 
        self.table_path = ""         # Table file path
        self.destination_path = ""   # Destination folder

        # Data model
        self.model: Optional[DataFrameModel] = None
        self.data_frame: Optional[pl.DataFrame] = None

        # Create mapping-specific widgets
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
        """Set up the main layout structure with pulsing indicator"""
        # ---- Page 1: Crop View ----
        # Main frame with image and controls
        frame, vertical_layout = self.setup_main_crop_frame(self.page_1)

        # Combo boxes, crop and cancel buttons
        button_layout = ut.setup_hbox("horizontalLayout_1")

        # Setup combo boxes
        ut.setup_combobox(self.comboBox_1, button_layout, self.size_policy_expand_fixed, "comboBox_1")
        ut.setup_combobox(self.comboBox_2, button_layout, self.size_policy_expand_fixed, "comboBox_2")

        # Crop and cancel buttons
        self.cropButton.setParent(frame)
        self.cancelButton.setParent(frame)
        button_layout.addWidget(self.cropButton)
        button_layout.addWidget(self.cancelButton)

        vertical_layout.addLayout(button_layout)

        # Pulsing indicator (replaces progress bar)
        self.pulsing_indicator.setParent(frame)
        vertical_layout.addWidget(self.pulsing_indicator)

        self.verticalLayout_200.addWidget(frame)

        # Add page to toolbox
        self.toolBox.addItem(self.page_1, "Crop View")

        # ---- Page 2: Table View ----
        # Table view
        self.tableView.setObjectName("tableView")
        self.tableView.setParent(self.page_2)
        self.verticalLayout_300.addWidget(self.tableView)

        # Combo boxes for column selection on table view
        combo_layout = ut.setup_hbox("horizontalLayout_4")

        ut.setup_combobox(self.comboBox_3, combo_layout, self.size_policy_expand_fixed, "comboBox_3")
        ut.setup_combobox(self.comboBox_4, combo_layout, self.size_policy_expand_fixed, "comboBox_4")

        self.verticalLayout_300.addLayout(combo_layout)

        # Add page to toolbox
        self.toolBox.addItem(self.page_2, "Table View")

        # Add toolbox to the main layout
        self.verticalLayout_100.addWidget(self.toolBox)

    def connect_signals(self) -> None:
        """Connect widget signals to handlers"""
        # Button connections
        self.tableButton.clicked.connect(self.open_table)
        self.cropButton.clicked.connect(self.mapping_process)

        # Combobox synchronization
        self.comboBox_1.currentTextChanged.connect(lambda text: self.comboBox_3.setCurrentText(text))
        self.comboBox_2.currentTextChanged.connect(lambda text: self.comboBox_4.setCurrentText(text))
        self.comboBox_3.currentTextChanged.connect(lambda text: self.comboBox_1.setCurrentText(text))
        self.comboBox_4.currentTextChanged.connect(lambda text: self.comboBox_2.setCurrentText(text))

        # Register button dependencies with the TabStateManager
        ut.register_button_dependencies(
            self.tab_state_manager,
            self.cropButton,
            {
                self.comboBox_1,
                self.comboBox_2,
                self.controlWidget.widthLineEdit,
                self.controlWidget.heightLineEdit
            }
        )
        
        # Connect all input widgets for validation tracking
        self.tab_state_manager.connect_widgets(
            self.controlWidget.widthLineEdit,
            self.controlWidget.heightLineEdit,
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

    def retranslateUi(self) -> None:
        """Update UI text elements"""
        super().retranslateUi()
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
        """Open a table file dialog with the string-based approach"""
        f_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Open File',
            file_manager.get_default_directory(FileCategory.PHOTO).as_posix(),
            file_manager.get_filter_string(FileCategory.TABLE)
        )

        # Validate the file exists and is accessible
        if f_name := ut.sanitize_path(f_name):
            # Update the table path
            self.table_path = f_name

            # Also update the main window's secondary input if this is the active tab
            main_window = self.parent().parent().parent()
            if main_window.function_tabWidget.currentIndex() == FunctionType.MAPPING:
                main_window.secondary_input.setText(f_name)

            # Load and process the table data
            data = prc.load_table(Path(f_name))
            self.process_data(data)

    def process_data(self, data: pl.DataFrame) -> None:
        """Process the loaded data"""
        try:
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
                    self.controlWidget.rightDial, self.tableButton, self.comboBox_1, self.comboBox_2,
                    self.controlWidget.radioButton_none,
                    self.controlWidget.radioButton_bmp, self.controlWidget.radioButton_jpg,
                    self.controlWidget.radioButton_png, self.controlWidget.radioButton_tiff,
                    self.controlWidget.radioButton_webp, self.cropButton, self.exposureCheckBox,
                    self.mfaceCheckBox, self.tiltCheckBox)

        self.connect_crop_worker_signals(*widget_list)

    def mapping_process(self) -> None:
        """Begin the mapping cropping process"""
        self.crop_worker.show_message_box = False

        def execute_crop():
            job = self.create_job(
                FunctionType.MAPPING,
                folder_path=Path(self.input_path),
                destination=Path(self.destination_path ),
                table=self.data_frame,
                column1=self.comboBox_1,
                column2=self.comboBox_2
            )
            self.run_batch_process(job,
                                   function=self.crop_worker.crop,
                                   reset_worker_func=lambda: self.crop_worker.reset_task())

        # Check if source and destination are the same and warn if needed
        self.check_source_destination_same(
            self.input_path,
            self.destination_path ,
            FunctionType.MAPPING,
            execute_crop
        )
