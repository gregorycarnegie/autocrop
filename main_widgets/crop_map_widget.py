from pathlib import Path
from typing import Optional, Any, Union

import pandas as pd
from PyQt6 import QtCore, QtWidgets

from core import CustomDialWidget, DataFrameModel, ExtWidget, FunctionTabSelectionState, FunctionType, ImageWidget, \
    utils, window_functions
from file_types import Photo, Table
from line_edits import PathLineEdit, PathType, NumberLineEdit, LineEditState
from core.cropper import Cropper
from .crop_batch_widget import CropBatchWidget
from .enums import ButtonType


class CropMapWidget(CropBatchWidget):
    def __init__(self, crop_worker: Cropper,
                 width_line_edit: NumberLineEdit,
                 height_line_edit: NumberLineEdit,
                 ext_widget: ExtWidget,
                 sensitivity_dial_area: CustomDialWidget,
                 face_dial_area: CustomDialWidget,
                 gamma_dial_area: CustomDialWidget,
                 top_dial_area: CustomDialWidget,
                 bottom_dial_area: CustomDialWidget,
                 left_dial_area: CustomDialWidget,
                 right_dial_area: CustomDialWidget,
                 parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(crop_worker, width_line_edit, height_line_edit, ext_widget, sensitivity_dial_area,
                         face_dial_area, gamma_dial_area, top_dial_area, bottom_dial_area, left_dial_area,
                         right_dial_area, parent)
        self.folderButton = self.setup_process_button('folderButton', 'folder', ButtonType.NAVIGATION_BUTTON)
        self.model: Optional[DataFrameModel] = None
        self.data_frame: Optional[pd.DataFrame] = None
        self.setObjectName('Form')
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self)
        self.verticalLayout_3.setObjectName('verticalLayout_3')
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName('gridLayout')
        self.folderLineEdit = self.setup_path_line_edit('folderLineEdit')
        self.gridLayout.addWidget(self.folderLineEdit, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.folderButton, 0, 1, 1, 1)
        self.tableLineEdit = self.setup_path_line_edit('tableLineEdit', PathType.TABLE)
        self.gridLayout.addWidget(self.tableLineEdit, 1, 0, 1, 1)
        self.tableButton = self.setup_process_button('tableButton', 'excel', ButtonType.NAVIGATION_BUTTON)
        self.gridLayout.addWidget(self.tableButton, 1, 1, 1, 1)
        self.verticalLayout_3.addLayout(self.gridLayout)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding,
                                           QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem)
        self.horizontalLayout_3.addWidget(self.mfaceCheckBox)
        self.horizontalLayout_3.addWidget(self.tiltCheckBox)
        self.horizontalLayout_3.addWidget(self.exposureCheckBox)
        self.verticalLayout_1.addLayout(self.horizontalLayout_3)
        self.imageWidget = ImageWidget(parent=self.frame)
        self.imageWidget.setStyleSheet('')
        self.imageWidget.setObjectName('imageWidget')
        self.verticalLayout_1.addWidget(self.imageWidget)
        self.horizontalLayout_2.addWidget(self.cropButton)
        self.horizontalLayout_2.addWidget(self.cancelButton)
        self.verticalLayout_1.addLayout(self.horizontalLayout_2)
        self.verticalLayout_1.addWidget(self.progressBar)
        self.verticalLayout_1.setStretch(0, 1)
        self.verticalLayout_1.setStretch(1, 10)
        self.verticalLayout_1.setStretch(2, 1)
        self.verticalLayout_1.setStretch(3, 1)
        self.horizontalLayout_1.addWidget(self.frame)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName('verticalLayout_2')
        self.tableView = QtWidgets.QTableView(parent=self)
        self.tableView.setObjectName('tableView')
        self.verticalLayout_2.addWidget(self.tableView)
        self.comboBox_1 = QtWidgets.QComboBox(parent=self)
        self.comboBox_1.setMinimumSize(QtCore.QSize(0, 22))
        self.comboBox_1.setMaximumSize(QtCore.QSize(16777215, 22))
        self.comboBox_1.setObjectName('comboBox_1')
        self.horizontalLayout_4.addWidget(self.comboBox_1)
        self.comboBox_2 = QtWidgets.QComboBox(parent=self)
        self.comboBox_2.setMinimumSize(QtCore.QSize(0, 22))
        self.comboBox_2.setMaximumSize(QtCore.QSize(16777215, 22))
        self.comboBox_2.setObjectName('comboBox_2')
        self.horizontalLayout_4.addWidget(self.comboBox_2)
        self.verticalLayout_2.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_1.addLayout(self.verticalLayout_2)
        self.horizontalLayout_1.setStretch(0, 1)
        self.horizontalLayout_1.setStretch(1, 2)
        self.verticalLayout_3.addLayout(self.horizontalLayout_1)
        self.horizontalLayout_5.addWidget(self.destinationLineEdit)
        self.horizontalLayout_5.addWidget(self.destinationButton)
        self.verticalLayout_3.addLayout(self.horizontalLayout_5)

        # Connections
        self.folderButton.clicked.connect(lambda: self.open_folder(self.folderLineEdit))
        self.destinationButton.clicked.connect(lambda: self.open_folder(self.destinationLineEdit))
        self.tableButton.clicked.connect(lambda: self.open_table())
        self.cropButton.clicked.connect(lambda: self.mapping_process())
        self.cancelButton.clicked.connect(lambda: self.crop_worker.terminate(FunctionType.MAPPING))
        self.cancelButton.clicked.connect(
            lambda: self.cancel_button_operation(self.cancelButton, self.cropButton))

        self.connect_input_widgets(self.folderLineEdit, self.widthLineEdit, self.heightLineEdit,
                                   self.destinationLineEdit, self.comboBox_1, self.comboBox_2, self.exposureCheckBox,
                                   self.mfaceCheckBox, self.tiltCheckBox, self.sensitivity_dialArea.dial,
                                   self.face_dialArea.dial, self.gamma_dialArea.dial, self.top_dialArea.dial,
                                   self.bottom_dialArea.dial, self.left_dialArea.dial, self.right_dialArea.dial)
        # Connect crop worker
        self.connect_crop_worker()

        self.retranslateUi()
        self.disable_buttons()
        window_functions.change_widget_state(False, self.cropButton, self.cancelButton)
        QtCore.QMetaObject.connectSlotsByName(self)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate('Form', 'Form'))
        self.folderLineEdit.setPlaceholderText(_translate('Form', 'Choose the folder you want to crop'))
        self.folderButton.setText(_translate('Form', 'Select Folder'))
        self.tableLineEdit.setPlaceholderText(_translate('Form', 'Choose the Excel or CSV file with the mapping'))
        self.tableButton.setText(_translate('Form', 'Open File'))
        self.mfaceCheckBox.setText(_translate('Form', 'Multi-Face'))
        self.tiltCheckBox.setText(_translate('Form', 'Autotilt'))
        self.exposureCheckBox.setText(_translate('Form', 'Autocorrect'))
        self.comboBox_1.setPlaceholderText(_translate('Form', 'Filename column'))
        self.comboBox_2.setPlaceholderText(_translate('Form', 'Mapping column'))
        self.destinationLineEdit.setPlaceholderText(
            _translate('Form', 'Choose where you want to save the cropped images'))
        self.destinationButton.setText(_translate('Form', "Destination Folder"))

    def connect_crop_worker(self) -> None:
        widget_list = (self.widthLineEdit, self.heightLineEdit, self.sensitivity_dialArea.dial, self.face_dialArea.dial,
                       self.gamma_dialArea.dial, self.top_dialArea.dial, self.bottom_dialArea.dial,
                       self.left_dialArea.dial, self.right_dialArea.dial, self.folderLineEdit, self.destinationLineEdit,
                       self.destinationButton, self.folderButton, self.tableLineEdit, self.comboBox_1, self.comboBox_2,
                       self.extWidget.radioButton_1, self.extWidget.radioButton_2, self.extWidget.radioButton_3,
                       self.extWidget.radioButton_4, self.extWidget.radioButton_5, self.extWidget.radioButton_6,
                       self.cropButton, self.exposureCheckBox, self.mfaceCheckBox, self.tiltCheckBox)
        # Mapping start connection
        self.crop_worker.mapping_started.connect(lambda: window_functions.disable_widget(*widget_list))
        self.crop_worker.mapping_started.connect(lambda: window_functions.enable_widget(self.cancelButton))
        # Mapping end connection
        self.crop_worker.mapping_finished.connect(lambda: window_functions.enable_widget(*widget_list))
        self.crop_worker.mapping_finished.connect(lambda: window_functions.disable_widget(self.cancelButton))
        self.crop_worker.mapping_finished.connect(lambda: window_functions.show_message_box(self.destinationLineEdit))
        self.crop_worker.mapping_progress.connect(lambda: self.update_progress(self.crop_worker.bar_value_m))
    
    def display_crop(self) -> None:
        job = self.create_job(self.exposureCheckBox, self.mfaceCheckBox, self.tiltCheckBox)
        self.crop_worker.display_crop(job, self.folderLineEdit, self.imageWidget)

    def load_data(self) -> None:
        try:
            self.display_crop()
        except (IndexError, FileNotFoundError, ValueError, AttributeError):
            return None

    def reload_widgets(self) -> None:
        def callback(input_path: Path) -> None:
            if not input_path.as_posix():
                return None
            self.display_crop()

        if not self.widthLineEdit.text() or not self.heightLineEdit.text():
            return None
        if self.selection_state == FunctionTabSelectionState.SELECTED:
            f_name = Path(self.folderLineEdit.text())
            callback(f_name)

    def connect_input_widgets(self, *input_widgets: QtWidgets.QWidget) -> None:
        for input_widget in input_widgets:
            match input_widget:
                case NumberLineEdit() | PathLineEdit():
                    input_widget.textChanged.connect(lambda: self.reload_widgets())
                    input_widget.textChanged.connect(lambda: self.disable_buttons())
                case QtWidgets.QDial():
                    input_widget.valueChanged.connect(lambda: self.reload_widgets())
                case QtWidgets.QComboBox():
                    input_widget.currentTextChanged.connect(lambda: self.disable_buttons())
                case QtWidgets.QCheckBox():
                    self.connect_checkboxs(input_widget)
                case _: pass

    def disable_buttons(self) -> None:
        def all_filled(*line_edits: Union[PathLineEdit, NumberLineEdit, QtWidgets.QComboBox]) -> bool:
            x = all(edit.state == LineEditState.VALID_INPUT
                    for edit in line_edits if isinstance(edit, (PathLineEdit, NumberLineEdit)))
            y = all(edit.currentText() for edit in line_edits if isinstance(edit, QtWidgets.QComboBox))
            return x and y

        def update_widget_state(condition: bool, *widgets: QtWidgets.QWidget) -> None:
            for widget in widgets:
                window_functions.change_widget_state(condition, widget)

        # Mapping logic
        update_widget_state(
            all_filled(self.folderLineEdit, self.tableLineEdit, self.destinationLineEdit, self.comboBox_1,
                       self.comboBox_2, self.widthLineEdit, self.heightLineEdit),
            self.cropButton)

    def open_table(self) -> None:
        f_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Open File', Photo().default_directory, Table().type_string)
        self.tableLineEdit.setText(f_name)
        if self.tableLineEdit.state is LineEditState.INVALID_INPUT:
            return None
        data = utils.open_table((Path(f_name)))
        self.validate_pandas_file(data)

    def mapping_process(self) -> None:
        job = self.create_job(self.exposureCheckBox, 
                              self.mfaceCheckBox, 
                              self.tiltCheckBox,
                              folder_path=self.folderLineEdit, 
                              destination=self.destinationLineEdit,
                              table=self.data_frame, 
                              column1=self.comboBox_1, 
                              column2=self.comboBox_2)
        self.run_batch_process(self.crop_worker.mapping_crop, self.crop_worker.reset_m_task, job)

    def validate_pandas_file(self, data: Any) -> None:
        try:
            assert isinstance(data, pd.DataFrame)
        except AssertionError:
            return None
        self.process_data(data)

    def process_data(self, data: pd.DataFrame) -> None:
        self.data_frame = data
        self.model = DataFrameModel(self.data_frame)
        self.tableView.setModel(self.model)
        self.comboBox_1.addItems(self.data_frame.columns.to_numpy())
        self.comboBox_2.addItems(self.data_frame.columns.to_numpy())
