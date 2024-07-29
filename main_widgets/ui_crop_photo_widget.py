from pathlib import Path

from PyQt6 import QtCore, QtWidgets

from core import Cropper
from core import window_functions as wf
from core.enums import FunctionType, GuiIcon
from file_types import Photo
from line_edits import LineEditState, PathLineEdit, PathType
from .ui_crop_widget import UiCropWidget


class UiPhotoTabWidget(UiCropWidget):
    def __init__(self, crop_worker: Cropper, object_name: str, parent: QtWidgets.QWidget):
        super().__init__(crop_worker, parent)
        self.setObjectName(object_name)
        self.selection_state = self.SELECTED

        self.inputLineEdit = self.create_str_line_edit(u"inputLineEdit", PathType.IMAGE)

        self.inputLineEdit.setParent(self)

        self.horizontalLayout_2.addWidget(self.inputLineEdit)

        self.inputButton.setParent(self)
        icon = wf.create_button_icon(GuiIcon.PICTURE)
        self.inputButton.setIcon(icon)

        self.horizontalLayout_2.addWidget(self.inputButton)

        self.horizontalLayout_2.setStretch(0, 1)

        self.verticalLayout_100.addLayout(self.horizontalLayout_2)

        self.frame = wf.create_frame(u"frame", self, self.size_policy2)
        self.verticalLayout_4 = wf.setup_vbox(u"verticalLayout_4", self.frame)
        self.toggleCheckBox.setParent(self.frame)
        self.toggleCheckBox.setChecked(True)

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

        self.verticalLayout_4.addLayout(self.horizontalLayout_1)

        self.imageWidget.setParent(self.frame)

        self.verticalLayout_4.addWidget(self.imageWidget)

        self.cropButton = wf.create_main_button(u"cropButton", self.size_policy1, GuiIcon.CROP, self.frame)
        self.cropButton.setDisabled(True)

        self.verticalLayout_4.addWidget(self.cropButton)

        self.verticalLayout_100.addWidget(self.frame)

        self.destinationLineEdit.setParent(self)

        self.horizontalLayout_3.addWidget(self.destinationLineEdit)

        self.destinationButton.setParent(self)
        self.destinationButton.setIcon(self.folder_icon)

        self.horizontalLayout_3.addWidget(self.destinationButton)
        self.horizontalLayout_3.setStretch(0, 1)

        self.verticalLayout_100.addLayout(self.horizontalLayout_3)

        # Connections
        self.inputButton.clicked.connect(lambda: self.open_folder(self.inputLineEdit))
        self.destinationButton.clicked.connect(lambda: self.open_folder(self.destinationLineEdit))
        self.cropButton.clicked.connect(lambda: self.crop_photo())
        self.inputLineEdit.textChanged.connect(lambda: self.reload_widgets())
        self.connect_input_widgets(self.inputLineEdit, self.controlWidget.widthLineEdit,
                                   self.controlWidget.heightLineEdit, self.destinationLineEdit, self.exposureCheckBox,
                                   self.mfaceCheckBox, self.tiltCheckBox, self.controlWidget.sensitivityDial,
                                   self.controlWidget.fpctDial, self.controlWidget.gammaDial,
                                   self.controlWidget.topDial, self.controlWidget.bottomDial,
                                   self.controlWidget.leftDial, self.controlWidget.rightDial)

        self.toggleCheckBox.toggled.connect(self.controlWidget.setVisible)

        self.controlWidget.widthLineEdit.textChanged.connect(lambda: self.reload_widgets())
        self.controlWidget.heightLineEdit.textChanged.connect(lambda: self.reload_widgets())

        self.retranslateUi()

        QtCore.QMetaObject.connectSlotsByName(self)

    # setupUi

    def retranslateUi(self):
        self.setWindowTitle(QtCore.QCoreApplication.translate("self", u"Form", None))
        self.inputLineEdit.setPlaceholderText(
            QtCore.QCoreApplication.translate("self", u"Choose the image you want to crop", None))
        self.inputButton.setText(QtCore.QCoreApplication.translate("self", u"Open Image", None))
        self.toggleCheckBox.setText(QtCore.QCoreApplication.translate("self", u"Toggle Settings", None))
        self.mfaceCheckBox.setText(QtCore.QCoreApplication.translate("self", u"Multi-Face", None))
        self.tiltCheckBox.setText(QtCore.QCoreApplication.translate("self", u"Autotilt", None))
        self.exposureCheckBox.setText(QtCore.QCoreApplication.translate("self", u"Autocorrect", None))
        self.cropButton.setText("")
        self.destinationLineEdit.setPlaceholderText(
            QtCore.QCoreApplication.translate("self", u"Choose where you want to save the cropped image", None))
        self.destinationButton.setText(QtCore.QCoreApplication.translate("self", u"Destination Folder", None))

    # retranslateUi

    def display_crop(self) -> None:
        job = self.create_job()
        self.crop_worker.display_crop(job, self.inputLineEdit, self.imageWidget)

    def open_folder(self, line_edit: PathLineEdit) -> None:
        if line_edit is self.inputLineEdit:
            f_name, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, 'Open File', Photo.default_directory, Photo.type_string())
            line_edit.setText(f_name)
            if self.inputLineEdit.state is LineEditState.INVALID_INPUT:
                return
            self.load_data()
        elif line_edit is self.destinationLineEdit:
            f_name = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Directory', Photo.default_directory)
            line_edit.setText(f_name)

    def load_data(self) -> None:
        try:
            self.display_crop()
        except (IndexError, FileNotFoundError, ValueError, AttributeError):
            return

    def reload_widgets(self) -> None:
        def callback(input_path: Path) -> None:
            if not input_path.as_posix():
                return
            self.display_crop()

        if not self.controlWidget.widthLineEdit.text() or not self.controlWidget.heightLineEdit.text():
            return
        if self.selection_state == self.SELECTED:
            f_name = Path(self.inputLineEdit.text())
            callback(f_name)

    def disable_buttons(self) -> None:
        wf.change_widget_state(
            wf.all_filled(self.inputLineEdit, self.destinationLineEdit, self.controlWidget.widthLineEdit,
                          self.controlWidget.heightLineEdit),
            self.cropButton)

    def crop_photo(self) -> None:
        def callback():
            job = self.create_job(FunctionType.PHOTO,
                                  photo_path=Path(self.inputLineEdit.text()),
                                  destination=Path(self.destinationLineEdit.text()))
            self.crop_worker.photo_crop(Path(self.inputLineEdit.text()), job)

        if Path(self.inputLineEdit.text()).parent == Path(self.destinationLineEdit.text()):
            match wf.show_warning(FunctionType.PHOTO):
                case QtWidgets.QMessageBox.StandardButton.Yes:
                    callback()
                case _:
                    return
        callback()
