from pathlib import Path
from typing import Optional

from PyQt6 import QtCore, QtWidgets

from core import Cropper, CustomDialWidget, ExtWidget, FunctionType
from core import window_functions as wf
from file_types import Photo
from line_edits import LineEditState, NumberLineEdit, PathLineEdit, PathType
from .custom_crop_widget import CustomCropWidget
from .enums import ButtonType


class CropPhotoWidget(CustomCropWidget):
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
        self.selection_state = self.SELECTED
        self.setObjectName('Form')
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self)
        self.verticalLayout_2.setObjectName('verticalLayout_2')
        self.photoLineEdit = self.setup_path_line_edit('photoLineEdit', PathType.IMAGE)
        self.horizontalLayout_2.addWidget(self.photoLineEdit)
        self.photoButton = self.setup_process_button('photoButton', 'picture', ButtonType.NAVIGATION_BUTTON)
        self.horizontalLayout_2.addWidget(self.photoButton)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.verticalLayout_1.setParent(self.frame)
        self.horizontalLayout_1.addItem(self.spacerItem)
        wf.add_widgets(self.horizontalLayout_1, self.mfaceCheckBox, self.tiltCheckBox, self.exposureCheckBox)
        self.verticalLayout_1.addLayout(self.horizontalLayout_1)
        self.imageWidget = self.setup_image_widget(parent=self.frame)
        wf.add_widgets(self.verticalLayout_1, self.imageWidget, self.cropButton)
        self.verticalLayout_1.setStretch(1, 1)
        self.verticalLayout_1.setStretch(2, 1)
        self.verticalLayout_2.addWidget(self.frame)
        wf.add_widgets(self.horizontalLayout_3, self.destinationLineEdit, self.destinationButton)
        self.verticalLayout_2.addLayout(self.horizontalLayout_3)

        # Connections
        self.photoButton.clicked.connect(lambda: self.open_folder(self.photoLineEdit))
        self.destinationButton.clicked.connect(lambda: self.open_folder(self.destinationLineEdit))
        self.cropButton.clicked.connect(lambda: self.crop_photo())
        self.photoLineEdit.textChanged.connect(lambda: self.reload_widgets())
        self.connect_input_widgets(self.photoLineEdit, self.widthLineEdit, self.heightLineEdit,
                                   self.destinationLineEdit, self.exposureCheckBox, self.mfaceCheckBox,
                                   self.tiltCheckBox, self.sensitivity_dialArea.dial, self.face_dialArea.dial,
                                   self.gamma_dialArea.dial, self.top_dialArea.dial, self.bottom_dialArea.dial,
                                   self.left_dialArea.dial, self.right_dialArea.dial)
        self.retranslateUi()
        self.disable_buttons()
        wf.change_widget_state(False, self.cropButton)
        QtCore.QMetaObject.connectSlotsByName(self)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate('Form', 'Form'))
        self.photoLineEdit.setPlaceholderText(_translate('Form', 'Choose the image you want to crop'))
        self.photoButton.setText(_translate('Form', 'Open Image'))
        self.mfaceCheckBox.setText(_translate('Form', 'Multi-Face'))
        self.tiltCheckBox.setText(_translate('Form', 'Autotilt'))
        self.exposureCheckBox.setText(_translate('Form', 'Autocorrect'))
        self.destinationLineEdit.setPlaceholderText(
            _translate('Form', 'Choose where you want to save the cropped image'))
        self.destinationButton.setText(_translate('Form', 'Destination Folder'))

    def display_crop(self) -> None:
        job = self.create_job(self.exposureCheckBox, self.mfaceCheckBox, self.tiltCheckBox)
        self.crop_worker.display_crop(job, self.photoLineEdit, self.imageWidget)

    def open_folder(self, line_edit: PathLineEdit) -> None:
        if line_edit is self.photoLineEdit:
            f_name, _ = QtWidgets.QFileDialog.getOpenFileName(
                    self, 'Open File', Photo.default_directory, Photo.type_string())
            line_edit.setText(f_name)
            if self.photoLineEdit.state is LineEditState.INVALID_INPUT:
                return None
            self.load_data()
        elif line_edit is self.destinationLineEdit:
            f_name = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Directory', Photo.default_directory)
            line_edit.setText(f_name)

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
        if self.selection_state == self.SELECTED:
            f_name = Path(self.photoLineEdit.text())
            callback(f_name)

    def disable_buttons(self) -> None:
        wf.update_widget_state(
            self.all_filled(self.photoLineEdit, self.destinationLineEdit, self.widthLineEdit, self.heightLineEdit),
            self.cropButton)

    def crop_photo(self) -> None:
        def callback():
            job = self.create_job(self.exposureCheckBox,
                                  self.mfaceCheckBox,
                                  self.tiltCheckBox,
                                  FunctionType.PHOTO,
                                  photo_path=Path(self.photoLineEdit.text()),
                                  destination=Path(self.destinationLineEdit.text()))
            self.crop_worker.photo_crop(Path(self.photoLineEdit.text()), job, self.crop_worker.face_workers[0])

        if Path(self.photoLineEdit.text()).parent == Path(self.destinationLineEdit.text()):
            match wf.show_warning(FunctionType.PHOTO):
                case QtWidgets.QMessageBox.StandardButton.Yes:
                    callback()
                case _: return
        callback()
