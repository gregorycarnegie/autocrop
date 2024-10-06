from pathlib import Path
from typing import Any, Optional

import pandas as pd
from PyQt6 import QtCore, QtWidgets

from core import DataFrameModel
from core import utils as ut
from core import window_functions as wf
from core.croppers import MappingCropper
from core.enums import FunctionType, GuiIcon
from core.operation_types import FaceToolPair
from file_types import Photo, Table
from line_edits import LineEditState, NumberLineEdit, PathLineEdit, PathType
from .ui_crop_batch_widget import UiCropBatchWidget


class UiMappingTabWidget(UiCropBatchWidget):
    def __init__(self, crop_worker: MappingCropper, object_name: str, parent: QtWidgets.QWidget, face_tool_list: list[FaceToolPair]):
        super().__init__(object_name, parent, face_tool_list)
        self.crop_worker = crop_worker
        
        self.model: Optional[DataFrameModel] = None
        self.data_frame: Optional[pd.DataFrame] = None

        self.FolderTab = QtWidgets.QToolBox(self)
        self.FolderTab.setObjectName(u"FolderTab")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.inputLineEdit.setParent(self.page_1)

        self.gridLayout.addWidget(self.inputLineEdit, 0, 0, 1, 1)

        self.inputButton.setParent(self.page_1)
        self.inputButton.setIcon(self.folder_icon)

        self.gridLayout.addWidget(self.inputButton, 0, 1, 1, 1)

        self.tableLineEdit = self.create_str_line_edit(u"tableLineEdit", PathType.TABLE)
        self.tableLineEdit.setParent(self.page_1)

        self.gridLayout.addWidget(self.tableLineEdit, 1, 0, 1, 1)

        self.tableButton = self.create_nav_button(u"tableButton")
        self.tableButton.setParent(self.page_1)
        icon = wf.create_button_icon(GuiIcon.EXCEL)
        self.tableButton.setIcon(icon)

        self.gridLayout.addWidget(self.tableButton, 1, 1, 1, 1)
        self.gridLayout.setColumnStretch(0, 20)
        self.gridLayout.setColumnStretch(1, 1)
        self.gridLayout.setColumnMinimumWidth(0, 20)
        self.gridLayout.setColumnMinimumWidth(1, 1)

        self.verticalLayout_200.addLayout(self.gridLayout)

        self.frame = wf.create_frame(u"frame", self.page_1, self.size_policy2)
        self.verticalLayout = wf.setup_vbox(u"verticalLayout", self.frame)
        self.toggleCheckBox.setParent(self.frame)

        self.horizontalLayout_2.addWidget(self.toggleCheckBox)

        self.horizontalSpacer = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding,
                                                      QtWidgets.QSizePolicy.Policy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer)

        self.mfaceCheckBox.setParent(self.frame)
        self.horizontalLayout_2.addWidget(self.mfaceCheckBox)

        self.tiltCheckBox.setParent(self.frame)
        self.horizontalLayout_2.addWidget(self.tiltCheckBox)

        self.exposureCheckBox.setParent(self.frame)
        self.horizontalLayout_2.addWidget(self.exposureCheckBox)
        self.horizontalLayout_2.setStretch(1, 1)

        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.imageWidget.setParent(self.frame)

        self.verticalLayout.addWidget(self.imageWidget)

        self.comboBox_1 = QtWidgets.QComboBox(self.frame)
        self.comboBox_1.setObjectName(u"comboBox_1")
        self.size_policy1.setHeightForWidth(self.comboBox_1.sizePolicy().hasHeightForWidth())
        self.comboBox_1.setSizePolicy(self.size_policy1)
        self.comboBox_1.setMinimumSize(QtCore.QSize(0, 40))
        self.comboBox_1.setMaximumSize(QtCore.QSize(16_777_215, 40))

        self.horizontalLayout_1.addWidget(self.comboBox_1)

        self.comboBox_2 = QtWidgets.QComboBox(self.frame)
        self.comboBox_2.setObjectName(u"comboBox_2")
        self.size_policy1.setHeightForWidth(self.comboBox_2.sizePolicy().hasHeightForWidth())
        self.comboBox_2.setSizePolicy(self.size_policy1)
        self.comboBox_2.setMinimumSize(QtCore.QSize(0, 40))
        self.comboBox_2.setMaximumSize(QtCore.QSize(16_777_215, 40))

        self.horizontalLayout_1.addWidget(self.comboBox_2)

        self.cropButton = wf.create_main_button(u"cropButton", self.size_policy2, GuiIcon.CROP, self.frame)
        self.cropButton.setDisabled(True)

        self.horizontalLayout_1.addWidget(self.cropButton)

        self.cancelButton = wf.create_main_button(u"cancelButton", self.size_policy2, GuiIcon.CANCEL, self.frame)
        self.cancelButton.setDisabled(True)

        self.horizontalLayout_1.addWidget(self.cancelButton)

        self.verticalLayout.addLayout(self.horizontalLayout_1)

        self.progressBar.setParent(self.frame)

        self.verticalLayout.addWidget(self.progressBar)

        self.verticalLayout_200.addWidget(self.frame)

        self.destinationLineEdit.setParent(self.page_1)

        self.horizontalLayout_3.addWidget(self.destinationLineEdit)

        self.destinationButton.setParent(self.page_1)
        self.destinationButton.setIcon(self.folder_icon)

        self.horizontalLayout_3.addWidget(self.destinationButton)
        self.horizontalLayout_3.setStretch(0, 20)
        self.horizontalLayout_3.setStretch(1, 1)

        self.verticalLayout_200.addLayout(self.horizontalLayout_3)

        self.FolderTab.addItem(self.page_1, u"Crop View")
        self.tableView = QtWidgets.QTableView(self.page_2)
        self.tableView.setObjectName(u"tableView")

        self.verticalLayout_300.addWidget(self.tableView)

        self.comboBox_3 = QtWidgets.QComboBox(self.page_2)
        self.comboBox_3.setObjectName(u"comboBox_3")
        self.size_policy1.setHeightForWidth(self.comboBox_3.sizePolicy().hasHeightForWidth())
        self.comboBox_3.setSizePolicy(self.size_policy1)
        self.comboBox_3.setMinimumSize(QtCore.QSize(0, 30))
        self.comboBox_3.setMaximumSize(QtCore.QSize(16_777_215, 30))

        self.horizontalLayout_4.addWidget(self.comboBox_3)

        self.comboBox_4 = QtWidgets.QComboBox(self.page_2)
        self.comboBox_4.setObjectName(u"comboBox_4")
        self.size_policy1.setHeightForWidth(self.comboBox_4.sizePolicy().hasHeightForWidth())
        self.comboBox_4.setSizePolicy(self.size_policy1)
        self.comboBox_4.setMinimumSize(QtCore.QSize(0, 30))
        self.comboBox_4.setMaximumSize(QtCore.QSize(16_777_215, 30))

        self.horizontalLayout_4.addWidget(self.comboBox_4)

        self.verticalLayout_300.addLayout(self.horizontalLayout_4)

        self.FolderTab.addItem(self.page_2, u"Table View")

        self.verticalLayout_100.addWidget(self.FolderTab)

        # Connect Widgets
        self.inputButton.clicked.connect(lambda: self.open_path(self.inputLineEdit))
        self.tableButton.clicked.connect(lambda: self.open_table())
        self.destinationButton.clicked.connect(lambda: self.open_path(self.destinationLineEdit))
        
        self.cropButton.clicked.connect(lambda: self.mapping_process())
        self.cancelButton.clicked.connect(lambda: self.crop_worker.terminate())
        self.cancelButton.clicked.connect(lambda: self.cancel_button_operation(self.cancelButton, self.cropButton))

        self.comboBox_1.currentTextChanged.connect(lambda text: self.comboBox_3.setCurrentText(text))
        self.comboBox_2.currentTextChanged.connect(lambda text: self.comboBox_4.setCurrentText(text))
        self.comboBox_3.currentTextChanged.connect(lambda text: self.comboBox_1.setCurrentText(text))
        self.comboBox_4.currentTextChanged.connect(lambda text: self.comboBox_2.setCurrentText(text))

        self.connect_input_widgets(self.inputLineEdit, self.controlWidget.widthLineEdit,
                                   self.controlWidget.heightLineEdit, self.destinationLineEdit, self.exposureCheckBox,
                                   self.tableLineEdit, self.comboBox_1, self.comboBox_2,
                                   self.mfaceCheckBox, self.tiltCheckBox, self.controlWidget.sensitivityDial,
                                   self.controlWidget.fpctDial, self.controlWidget.gammaDial,
                                   self.controlWidget.topDial, self.controlWidget.bottomDial,
                                   self.controlWidget.leftDial, self.controlWidget.rightDial)

        self.toggleCheckBox.toggled.connect(self.controlWidget.setVisible)

        # Connect crop worker
        self.connect_crop_worker()

        self.retranslateUi()

        self.FolderTab.setCurrentIndex(0)

        QtCore.QMetaObject.connectSlotsByName(self)

    # setupUi

    def retranslateUi(self):
        self.setWindowTitle(QtCore.QCoreApplication.translate("self", u"Form", None))
        self.inputLineEdit.setPlaceholderText(
            QtCore.QCoreApplication.translate("self", u"Choose the folder you want to crop", None))
        self.inputButton.setText(QtCore.QCoreApplication.translate("self", u"Select Folder", None))
        self.tableLineEdit.setPlaceholderText(
            QtCore.QCoreApplication.translate("self", u"Choose the Excel or CSV file with the mapping", None))
        self.tableButton.setText(QtCore.QCoreApplication.translate("self", u"Open File", None))
        self.toggleCheckBox.setText(QtCore.QCoreApplication.translate("self", u"Toggle Settings", None))
        self.mfaceCheckBox.setText(QtCore.QCoreApplication.translate("self", u"Multi-Face", None))
        self.tiltCheckBox.setText(QtCore.QCoreApplication.translate("self", u"Autotilt", None))
        self.exposureCheckBox.setText(QtCore.QCoreApplication.translate("self", u"Autocorrect", None))
        self.comboBox_1.setPlaceholderText(QtCore.QCoreApplication.translate("self", u"Filename column", None))
        self.comboBox_2.setPlaceholderText(QtCore.QCoreApplication.translate("self", u"Mapping column", None))
        self.cropButton.setText("")
        self.cancelButton.setText("")
        self.destinationLineEdit.setPlaceholderText(
            QtCore.QCoreApplication.translate("self", u"Choose where you want to save the cropped images", None))
        self.destinationButton.setText(QtCore.QCoreApplication.translate("self", u"Destination Folder", None))
        self.FolderTab.setItemText(self.FolderTab.indexOf(self.page_1),
                                   QtCore.QCoreApplication.translate("self", u"Crop View", None))
        self.comboBox_3.setPlaceholderText(QtCore.QCoreApplication.translate("self", u"Filename column", None))
        self.comboBox_4.setPlaceholderText(QtCore.QCoreApplication.translate("self", u"Mapping column", None))
        self.FolderTab.setItemText(self.FolderTab.indexOf(self.page_2),
                                   QtCore.QCoreApplication.translate("self", u"Table View", None))

    # retranslateUi

    def connect_crop_worker(self) -> None:
        widget_list = (self.controlWidget.widthLineEdit, self.controlWidget.heightLineEdit,
                       self.controlWidget.sensitivityDial, self.controlWidget.fpctDial, self.controlWidget.gammaDial,
                       self.controlWidget.topDial, self.controlWidget.bottomDial, self.controlWidget.leftDial,
                       self.controlWidget.rightDial, self.inputLineEdit, self.destinationLineEdit,
                       self.tableButton,
                       self.destinationButton, self.inputButton, self.tableLineEdit, self.comboBox_1, self.comboBox_2,
                       self.controlWidget.radioButton_none, self.controlWidget.radioButton_bmp,
                       self.controlWidget.radioButton_jpg, self.controlWidget.radioButton_png,
                       self.controlWidget.radioButton_tiff, self.controlWidget.radioButton_webp, self.cropButton,
                       self.exposureCheckBox, self.mfaceCheckBox, self.tiltCheckBox)
        # Mapping start connection
        self.crop_worker.started.connect(lambda: wf.disable_widget(*widget_list))
        self.crop_worker.started.connect(lambda: wf.enable_widget(self.cancelButton))
        # Mapping end connection
        self.crop_worker.finished.connect(lambda: wf.enable_widget(*widget_list))
        self.crop_worker.finished.connect(lambda: wf.disable_widget(self.cancelButton))
        self.crop_worker.finished.connect(lambda: wf.show_message_box(self.destination))
        self.crop_worker.progress.connect(self.update_progress)

    def connect_input_widgets(self, *input_widgets: QtWidgets.QWidget) -> None:
        for input_widget in input_widgets:
            match input_widget:
                case NumberLineEdit() | PathLineEdit():
                    input_widget.textChanged.connect(lambda: self.disable_buttons())
                case QtWidgets.QComboBox():
                    input_widget.currentTextChanged.connect(lambda: self.disable_buttons())
                case QtWidgets.QCheckBox():
                    self.connect_checkbox(input_widget)
                case _:
                    pass

    def disable_buttons(self) -> None:
        wf.change_widget_state(
            wf.all_filled(self.inputLineEdit, self.tableLineEdit, self.destinationLineEdit, self.comboBox_1,
                          self.comboBox_2, self.controlWidget.widthLineEdit, self.controlWidget.heightLineEdit),
            self.cropButton)

    def open_table(self) -> None:
        f_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Open File', Photo.default_directory, Table.type_string())
        self.tableLineEdit.setText(f_name)
        if self.tableLineEdit.state is LineEditState.INVALID_INPUT:
            return
        data = ut.open_table((Path(f_name)))
        self.validate_pandas_file(data)

    def validate_pandas_file(self, data: Any) -> None:
        try:
            assert isinstance(data, pd.DataFrame)
        except AssertionError:
            return
        self.process_data(data)

    def process_data(self, data: pd.DataFrame) -> None:
        self.data_frame = data
        self.model = DataFrameModel(self.data_frame)
        self.tableView.setModel(self.model)
        self.comboBox_1.addItems(self.data_frame.columns.to_numpy())
        self.comboBox_2.addItems(self.data_frame.columns.to_numpy())
        self.comboBox_3.addItems(self.data_frame.columns.to_numpy())
        self.comboBox_4.addItems(self.data_frame.columns.to_numpy())

    def mapping_process(self) -> None:
        self.crop_worker.message_box = False
        def callback():
            job = self.create_job(FunctionType.MAPPING,
                                  folder_path=Path(self.inputLineEdit.text()),
                                  destination=Path(self.destinationLineEdit.text()),
                                  table=self.data_frame,
                                  column1=self.comboBox_1,
                                  column2=self.comboBox_2)
            self.run_batch_process(job, function=self.crop_worker.crop,
                                   reset_worker_func=lambda: self.crop_worker.reset_task())

        if Path(self.inputLineEdit.text()) == Path(self.destinationLineEdit.text()):
            match wf.show_warning(FunctionType.MAPPING):
                case QtWidgets.QMessageBox.StandardButton.Yes:
                    callback()
                case _:
                    return
        callback()
