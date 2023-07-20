from pathlib import Path
from typing import Optional, Union

from PyQt6 import QtCore, QtGui, QtWidgets

from .cropper import Cropper
from .custom_crop_widget import CustomCropWidget
from .custom_dial_widget import CustomDialWidget
from .enums import FunctionTabSelectionState
from .ext_widget import ExtWidget
from .f_type_photo import Photo
from .image_widget import ImageWidget
from .window_functions import change_widget_state, uncheck_boxes
from ..line_edits import PathLineEdit, PathType, NumberLineEdit, LineEditState


class CropPhotoWidget(CustomCropWidget):
    def __init__(self, crop_worker: Cropper,
                 widthLineEdit: NumberLineEdit,
                 heightLineEdit: NumberLineEdit,
                 extWidget: ExtWidget,
                 sensitivity_dialArea: CustomDialWidget,
                 face_dialArea: CustomDialWidget,
                 gamma_dialArea: CustomDialWidget,
                 top_dialArea: CustomDialWidget,
                 bottom_dialArea: CustomDialWidget,
                 left_dialArea: CustomDialWidget,
                 right_dialArea: CustomDialWidget,
                 parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(crop_worker, widthLineEdit, heightLineEdit, extWidget,
                         sensitivity_dialArea, face_dialArea, gamma_dialArea, top_dialArea,
                         bottom_dialArea, left_dialArea, right_dialArea, parent)

        self.selection_state = FunctionTabSelectionState.SELECTED

        self.setObjectName("Form")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.photoLineEdit = PathLineEdit(path_type=PathType.IMAGE, parent=self)
        self.photoLineEdit.setMinimumSize(QtCore.QSize(0, 24))
        self.photoLineEdit.setMaximumSize(QtCore.QSize(16777215, 24))
        self.photoLineEdit.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhUrlCharactersOnly)
        self.photoLineEdit.setObjectName("photoLineEdit")
        self.horizontalLayout_2.addWidget(self.photoLineEdit)
        self.photoButton = QtWidgets.QPushButton(parent=self)
        self.photoButton.setMinimumSize(QtCore.QSize(124, 24))
        self.photoButton.setMaximumSize(QtCore.QSize(16777215, 24))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("resources/icons/picture.svg"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.photoButton.setIcon(icon)
        self.photoButton.setObjectName("photoButton")
        self.horizontalLayout_2.addWidget(self.photoButton)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.frame = QtWidgets.QFrame(parent=self)
        self.frame.setStyleSheet("background: #1f2c33")
        self.frame.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.frame.setObjectName("frame")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.frame)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_1 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_1.setObjectName("horizontalLayout_1")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding,
                                           QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout_1.addItem(spacerItem)
        self.mfaceCheckBox = QtWidgets.QCheckBox(parent=self.frame)
        self.mfaceCheckBox.setStyleSheet(self.CHECKBOX_STYLESHEET)
        self.mfaceCheckBox.setObjectName("mfaceCheckBox")
        self.horizontalLayout_1.addWidget(self.mfaceCheckBox)
        self.tiltCheckBox = QtWidgets.QCheckBox(parent=self.frame)
        self.tiltCheckBox.setStyleSheet(self.CHECKBOX_STYLESHEET)
        self.tiltCheckBox.setObjectName("tiltCheckBox")
        self.horizontalLayout_1.addWidget(self.tiltCheckBox)
        self.exposureCheckBox = QtWidgets.QCheckBox(parent=self.frame)
        self.exposureCheckBox.setStyleSheet(self.CHECKBOX_STYLESHEET)
        self.exposureCheckBox.setObjectName("exposureCheckBox")
        self.horizontalLayout_1.addWidget(self.exposureCheckBox)
        self.verticalLayout.addLayout(self.horizontalLayout_1)
        self.imageWidget = ImageWidget(parent=self.frame)
        self.imageWidget.setStyleSheet("")
        self.imageWidget.setObjectName("imageWidget")
        self.verticalLayout.addWidget(self.imageWidget)
        self.cropButton = QtWidgets.QPushButton(parent=self.frame)
        self.cropButton.setMinimumSize(QtCore.QSize(0, 24))
        self.cropButton.setMaximumSize(QtCore.QSize(16777215, 24))
        self.cropButton.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("resources/icons/crop.svg"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.cropButton.setIcon(icon1)
        self.cropButton.setObjectName("cropButton")
        self.verticalLayout.addWidget(self.cropButton)
        self.verticalLayout.setStretch(1, 1)
        self.verticalLayout.setStretch(2, 1)
        self.verticalLayout_2.addWidget(self.frame)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.destinationLineEdit = PathLineEdit(parent=self)
        self.destinationLineEdit.setMinimumSize(QtCore.QSize(0, 24))
        self.destinationLineEdit.setMaximumSize(QtCore.QSize(16777215, 24))
        self.destinationLineEdit.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhUrlCharactersOnly)
        self.destinationLineEdit.setObjectName("destinationLineEdit")
        self.horizontalLayout_3.addWidget(self.destinationLineEdit)
        self.destinationButton = QtWidgets.QPushButton(parent=self)
        self.destinationButton.setMinimumSize(QtCore.QSize(124, 24))
        self.destinationButton.setMaximumSize(QtCore.QSize(16777215, 24))
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("resources/icons/folder.svg"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.destinationButton.setIcon(icon2)
        self.destinationButton.setObjectName("destinationButton")
        self.horizontalLayout_3.addWidget(self.destinationButton)
        self.verticalLayout_2.addLayout(self.horizontalLayout_3)

        # Connections
        self.photoButton.clicked.connect(lambda: self.open_folder(self.photoLineEdit))

        self.destinationButton.clicked.connect(lambda: self.open_folder(self.destinationLineEdit))

        self.cropButton.clicked.connect(lambda: self.crop_photo())

        self.connect_input_widgets(self.widthLineEdit, self.heightLineEdit, self.destinationLineEdit,
                                   self.exposureCheckBox, self.mfaceCheckBox, self.tiltCheckBox,
                                   self.sensitivity_dialArea.dial, self.face_dialArea.dial, self.gamma_dialArea.dial,
                                   self.top_dialArea.dial, self.bottom_dialArea.dial,
                                   self.left_dialArea.dial, self.right_dialArea.dial)

        self.retranslateUi()
        self.disable_buttons()
        change_widget_state(False, self.cropButton)
        QtCore.QMetaObject.connectSlotsByName(self)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("Form", "Form"))
        self.photoLineEdit.setPlaceholderText(_translate("Form", "Choose the image you want to crop"))
        self.photoButton.setText(_translate("Form", "PushButton"))
        self.mfaceCheckBox.setText(_translate("Form", "Multi-Face"))
        self.tiltCheckBox.setText(_translate("Form", "Autotilt"))
        self.exposureCheckBox.setText(_translate("Form", "Autocorrect"))
        self.destinationLineEdit.setPlaceholderText(
            _translate("Form", "Choose where you want to save the cropped image"))
        self.destinationButton.setText(_translate("Form", "Destination Folder"))

    def display_crop(self) -> None:
        job = self.create_job(self.exposureCheckBox, self.mfaceCheckBox, self.tiltCheckBox)
        self.crop_worker.display_crop(job, self.photoLineEdit, self.imageWidget)

    def open_folder(self, line_edit: PathLineEdit) -> None:
        if line_edit is self.photoLineEdit:
                f_name, _ = QtWidgets.QFileDialog.getOpenFileName(
                    self, 'Open File', Photo().default_directory, Photo().type_string())
                line_edit.setText(f_name)
                self.load_data()
        elif line_edit is self.destinationLineEdit:
             f_name = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Directory', Photo().default_directory)
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
        if self.selection_state == FunctionTabSelectionState.SELECTED:
            f_name = Path(self.photoLineEdit.text())
            callback(f_name)
           
    def connect_input_widgets(self, *input_widgets: QtWidgets.QWidget) -> None:
        for input_widget in input_widgets:
            if isinstance(input_widget, (NumberLineEdit, PathLineEdit)):
                if input_widget is self.photoLineEdit:
                    input_widget.textChanged.connect(lambda: self.reload_widgets())
                input_widget.textChanged.connect(lambda: self.disable_buttons())
            elif isinstance(input_widget, QtWidgets.QDial):
                input_widget.valueChanged.connect(lambda: self.reload_widgets())
            elif isinstance(input_widget, QtWidgets.QCheckBox):
                input_widget.stateChanged.connect(lambda: self.reload_widgets())
                if input_widget is self.mfaceCheckBox:
                    input_widget.clicked.connect(lambda: uncheck_boxes(self.exposureCheckBox, self.tiltCheckBox))
                else:
                    input_widget.clicked.connect(lambda: uncheck_boxes(self.mfaceCheckBox))
 
    def disable_buttons(self) -> None:
        def all_filled(*line_edits: Union[PathLineEdit, NumberLineEdit, QtWidgets.QComboBox]) -> bool:
            x = all(edit.state == LineEditState.VALID_INPUT
                    for edit in line_edits if isinstance(edit, (PathLineEdit, NumberLineEdit)))
            y = all(edit.currentText() for edit in line_edits if isinstance(edit, QtWidgets.QComboBox))
            return x and y

        def update_widget_state(condition: bool, *widgets: QtWidgets.QWidget) -> None:
            for widget in widgets:
                change_widget_state(condition, widget)

        # Photo logic
        update_widget_state(
            all_filled(self.photoLineEdit, self.destinationLineEdit, self.widthLineEdit, self.heightLineEdit),
            self.cropButton)

    def crop_photo(self) -> None:
        job = self.create_job(self.exposureCheckBox, 
                              self.mfaceCheckBox, 
                              self.tiltCheckBox,
                              photo_path=self.photoLineEdit, 
                              destination=self.destinationLineEdit)
        self.crop_worker.crop(Path(self.photoLineEdit.text()), 
                              job, 
                              self.crop_worker.face_workers[0])
