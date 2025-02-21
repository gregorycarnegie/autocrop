from pathlib import Path

from PyQt6 import QtCore, QtGui, QtWidgets

from core import window_functions as wf
from core.croppers import FolderCropper
from core.enums import FunctionType, GuiIcon
from file_types import Photo
from line_edits import PathLineEdit
from .ui_crop_batch_widget import UiCropBatchWidget


class UiFolderTabWidget(UiCropBatchWidget):
    def __init__(self, crop_worker: FolderCropper,
                 object_name: str,
                 parent: QtWidgets.QWidget) -> None:
        super().__init__(object_name, parent)
        self.crop_worker = crop_worker

        self.file_model = QtGui.QFileSystemModel(self)
        self.file_model.setFilter(QtCore.QDir.Filter.NoDotAndDotDot | QtCore.QDir.Filter.Files)
        self.file_model.setNameFilters(Photo.file_filter())

        self.toolBox = QtWidgets.QToolBox(self)
        self.toolBox.setObjectName(u"toolBox")
        self.inputLineEdit.setParent(self.page_1)

        self.horizontalLayout_4.addWidget(self.inputLineEdit)

        self.inputButton.setParent(self.page_1)
        self.inputButton.setIcon(self.folder_icon)

        self.horizontalLayout_4.addWidget(self.inputButton)
        self.horizontalLayout_4.setStretch(0, 1)

        self.verticalLayout_200.addLayout(self.horizontalLayout_4)

        self.frame = wf.create_frame(u"frame", self.page_1, self.size_policy2)
        self.verticalLayout = wf.setup_vbox(u"verticalLayout", self.frame)

        self.toggleCheckBox.setParent(self.frame)

        self.horizontalLayout_1.addWidget(self.toggleCheckBox)

        self.horizontalSpacer = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding,
                                                      QtWidgets.QSizePolicy.Policy.Minimum)

        self.horizontalLayout_1.addItem(self.horizontalSpacer)

        self.mfaceCheckBox.setParent(self.frame)
        self.horizontalLayout_1.addWidget(self.mfaceCheckBox)

        self.tiltCheckBox.setParent(self.frame)
        self.horizontalLayout_1.addWidget(self.tiltCheckBox)

        self.exposureCheckBox.setParent(self.frame)

        self.horizontalLayout_1.addWidget(self.exposureCheckBox)
        self.horizontalLayout_1.setStretch(1, 20)

        self.verticalLayout.addLayout(self.horizontalLayout_1)

        self.imageWidget.setParent(self.frame)

        self.verticalLayout.addWidget(self.imageWidget)

        self.cropButton = wf.create_main_button(u"cropButton", self.size_policy2, GuiIcon.CROP, self.frame)
        self.cropButton.setDisabled(True)

        self.horizontalLayout_2.addWidget(self.cropButton)

        self.cancelButton = wf.create_main_button(u"cancelButton", self.size_policy2, GuiIcon.CANCEL, self.frame)
        self.cancelButton.setDisabled(True)

        self.horizontalLayout_2.addWidget(self.cancelButton)

        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.progressBar.setParent(self.frame)

        self.verticalLayout.addWidget(self.progressBar)

        self.verticalLayout_200.addWidget(self.frame)

        self.destinationLineEdit.setParent(self.page_1)

        self.horizontalLayout_3.addWidget(self.destinationLineEdit)

        self.destinationButton.setParent(self.page_1)
        self.destinationButton.setIcon(self.folder_icon)

        self.horizontalLayout_3.addWidget(self.destinationButton)
        self.horizontalLayout_3.setStretch(0, 1)

        self.verticalLayout_200.addLayout(self.horizontalLayout_3)

        self.toolBox.addItem(self.page_1, u"Crop View")
        self.treeView = QtWidgets.QTreeView(self.page_2)
        self.treeView.setObjectName(u"treeView")
        self.treeView.setModel(self.file_model)

        self.verticalLayout_300.addWidget(self.treeView)

        self.toolBox.addItem(self.page_2, u"Folder View")

        self.verticalLayout_100.addWidget(self.toolBox)

        # Connect Widgets
        self.inputButton.clicked.connect(lambda: self.open_path(self.inputLineEdit))
        self.destinationButton.clicked.connect(lambda: self.open_path(self.destinationLineEdit))
        self.inputLineEdit.textChanged.connect(lambda: self.load_data())
        self.cropButton.clicked.connect(lambda: self.folder_process())
        self.cancelButton.clicked.connect(lambda: self.crop_worker.terminate())
        self.cancelButton.clicked.connect(lambda: self.cancel_button_operation(self.cancelButton, self.cropButton))

        self.connect_input_widgets(self.inputLineEdit, self.controlWidget.widthLineEdit,
                                   self.controlWidget.heightLineEdit, self.destinationLineEdit, self.exposureCheckBox,
                                   self.mfaceCheckBox, self.tiltCheckBox, self.controlWidget.sensitivityDial,
                                   self.controlWidget.fpctDial, self.controlWidget.gammaDial,
                                   self.controlWidget.topDial, self.controlWidget.bottomDial,
                                   self.controlWidget.leftDial, self.controlWidget.rightDial)

        self.toggleCheckBox.toggled.connect(self.controlWidget.setVisible)

        # Connect crop worker
        self.connect_crop_worker()

        self.retranslateUi()

        self.toolBox.setCurrentIndex(0)

        QtCore.QMetaObject.connectSlotsByName(self)

    # setupUi

    def retranslateUi(self):
        self.setWindowTitle(QtCore.QCoreApplication.translate("self", u"Form", None))
        self.inputLineEdit.setPlaceholderText(
            QtCore.QCoreApplication.translate("self", u"Choose the folder you want to crop", None))
        self.inputButton.setText(QtCore.QCoreApplication.translate("self", u"Select Folder", None))
        self.toggleCheckBox.setText(QtCore.QCoreApplication.translate("self", u"Toggle Settings", None))
        self.mfaceCheckBox.setText(QtCore.QCoreApplication.translate("self", u"Multi-Face", None))
        self.tiltCheckBox.setText(QtCore.QCoreApplication.translate("self", u"Autotilt", None))
        self.exposureCheckBox.setText(QtCore.QCoreApplication.translate("self", u"Autocorrect", None))
        self.cropButton.setText("")
        self.cancelButton.setText("")
        self.destinationLineEdit.setPlaceholderText(
            QtCore.QCoreApplication.translate("self", u"Choose where you want to save the cropped images", None))
        self.destinationButton.setText(QtCore.QCoreApplication.translate("self", u"Destination Folder", None))
        self.toolBox.setItemText(self.toolBox.indexOf(self.page_1),
                                 QtCore.QCoreApplication.translate("self", u"Crop View", None))
        self.toolBox.setItemText(self.toolBox.indexOf(self.page_2),
                                 QtCore.QCoreApplication.translate("self", u"Folder View", None))

    # retranslateUi

    def connect_crop_worker(self) -> None:
        widget_list = (self.controlWidget.widthLineEdit, self.controlWidget.heightLineEdit,
                       self.controlWidget.sensitivityDial, self.controlWidget.fpctDial, self.controlWidget.gammaDial,
                       self.controlWidget.topDial, self.controlWidget.bottomDial, self.controlWidget.leftDial,
                       self.controlWidget.rightDial, self.inputLineEdit, self.destinationLineEdit,
                       self.destinationButton, self.inputButton, self.controlWidget.radioButton_none,
                       self.controlWidget.radioButton_bmp, self.controlWidget.radioButton_jpg,
                       self.controlWidget.radioButton_png, self.controlWidget.radioButton_tiff,
                       self.controlWidget.radioButton_webp, self.cropButton, self.exposureCheckBox, self.mfaceCheckBox,
                       self.tiltCheckBox)

        # Folder start connection
        self.crop_worker.started.connect(lambda: wf.disable_widget(*widget_list))
        self.crop_worker.started.connect(lambda: wf.enable_widget(self.cancelButton))

        # Folder end connection
        self.crop_worker.finished.connect(lambda: wf.enable_widget(*widget_list))
        self.crop_worker.finished.connect(lambda: wf.disable_widget(self.cancelButton))
        self.crop_worker.finished.connect(lambda: wf.show_message_box(self.destination))
        self.crop_worker.progress.connect(self.update_progress)

    def open_path(self, line_edit: PathLineEdit) -> None:
        """Only subclasses of the CustomCropWidget class should implement this method"""
        super().open_path(line_edit)
        if line_edit is self.inputLineEdit:
            self.load_data()

    def load_data(self) -> None:
        try:
            f_name = self.inputLineEdit.text()
            self.file_model.setRootPath(f_name)
            self.treeView.setRootIndex(self.file_model.index(f_name))
        except (IndexError, FileNotFoundError, ValueError, AttributeError):
            return

    def disable_buttons(self) -> None:
        wf.change_widget_state(
            wf.all_filled(self.inputLineEdit, self.destinationLineEdit, self.controlWidget.widthLineEdit,
                          self.controlWidget.heightLineEdit),
            self.cropButton)

    def folder_process(self) -> None:
        self.crop_worker.show_message_box = False
        def callback():
            job = self.create_job(FunctionType.FOLDER,
                                  folder_path=Path(self.inputLineEdit.text()),
                                  destination=Path(self.destinationLineEdit.text()))
            self.run_batch_process(job, function=self.crop_worker.crop,
                                   reset_worker_func=lambda: self.crop_worker.reset_task())

        if Path(self.inputLineEdit.text()) == Path(self.destinationLineEdit.text()):
            match wf.show_warning(FunctionType.FOLDER):
                case QtWidgets.QMessageBox.StandardButton.Yes:
                    callback()
                case _:
                    return
        callback()
