from pathlib import Path
from typing import Optional, Union

from PyQt6 import QtCore, QtGui, QtWidgets

from core import Cropper, CustomDialWidget, ExtWidget, FunctionTabSelectionState, FunctionType, ImageWidget, \
    window_functions
from file_types import Photo
from line_edits import PathLineEdit, NumberLineEdit, LineEditState
from .crop_batch_widget import CropBatchWidget


class CropFolderWidget(CropBatchWidget):
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
        self.file_model = QtGui.QFileSystemModel(self)
        self.file_model.setFilter(QtCore.QDir.Filter.NoDotAndDotDot | QtCore.QDir.Filter.Files)
        self.file_model.setNameFilters(Photo().file_filter)
        self.setObjectName('Form')
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self)
        self.verticalLayout_2.setObjectName('verticalLayout_2')
        self.horizontalLayout_1 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_1.setObjectName('horizontalLayout_1')
        self.folderLineEdit.setParent(self)
        self.horizontalLayout_1.addWidget(self.folderLineEdit)
        self.folderButton.setParent(self)
        self.horizontalLayout_1.addWidget(self.folderButton)
        self.verticalLayout_2.addLayout(self.horizontalLayout_1)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName('horizontalLayout_3')
        self.verticalLayout = QtWidgets.QVBoxLayout(self.frame)
        self.verticalLayout.setObjectName('verticalLayout')
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName('horizontalLayout_4')
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding,
                                           QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem)
        self.mfaceCheckBox.setParent(self.frame)
        self.horizontalLayout_4.addWidget(self.mfaceCheckBox)
        self.tiltCheckBox.setParent(self.frame)
        self.horizontalLayout_4.addWidget(self.tiltCheckBox)
        self.exposureCheckBox.setParent(self.frame)
        self.horizontalLayout_4.addWidget(self.exposureCheckBox)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        self.imageWidget = ImageWidget(parent=self.frame)
        self.imageWidget.setStyleSheet('')
        self.imageWidget.setObjectName('imageWidget')
        self.verticalLayout.addWidget(self.imageWidget)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName('horizontalLayout_5')
        self.cropButton.setParent(self.frame)
        self.horizontalLayout_5.addWidget(self.cropButton)
        self.cancelButton.setParent(self.frame)
        self.horizontalLayout_5.addWidget(self.cancelButton)
        self.verticalLayout.addLayout(self.horizontalLayout_5)
        self.progressBar.setParent(self.frame)
        self.verticalLayout.addWidget(self.progressBar)
        self.verticalLayout.setStretch(0, 1)
        self.verticalLayout.setStretch(1, 10)
        self.verticalLayout.setStretch(2, 1)
        self.verticalLayout.setStretch(3, 1)
        self.horizontalLayout_3.addWidget(self.frame)
        self.treeView = QtWidgets.QTreeView(parent=self)
        self.treeView.setObjectName('treeView')
        self.treeView.setModel(self.file_model)
        self.horizontalLayout_3.addWidget(self.treeView)
        self.horizontalLayout_3.setStretch(0, 4)
        self.horizontalLayout_3.setStretch(1, 3)
        self.verticalLayout_2.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName('horizontalLayout_2')
        self.destinationLineEdit.setParent(self)
        self.horizontalLayout_2.addWidget(self.destinationLineEdit)
        self.destinationButton.setParent(self)
        self.horizontalLayout_2.addWidget(self.destinationButton)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)

        # Connections
        self.folderButton.clicked.connect(lambda: self.open_folder(self.folderLineEdit))
        self.destinationButton.clicked.connect(lambda: self.open_folder(self.destinationLineEdit))
        self.treeView.selectionModel().selectionChanged.connect(lambda: self.reload_widgets())
        self.cropButton.clicked.connect(lambda: self.folder_process())
        self.cancelButton.clicked.connect(lambda: self.crop_worker.terminate(FunctionType.FOLDER))
        self.cancelButton.clicked.connect(lambda: self.cancel_button_operation(self.cancelButton, self.cropButton))
                                          
        self.connect_input_widgets(self.folderLineEdit, self.widthLineEdit, self.heightLineEdit,
                                   self.destinationLineEdit, self.exposureCheckBox, self.mfaceCheckBox,
                                   self.tiltCheckBox, self.sensitivity_dialArea.dial, self.face_dialArea.dial,
                                   self.gamma_dialArea.dial, self.top_dialArea.dial, self.bottom_dialArea.dial,
                                   self.left_dialArea.dial, self.right_dialArea.dial)
        # Folder start connection
        self.crop_worker.folder_started.connect(
            lambda: window_functions.disable_widget(self.widthLineEdit, self.heightLineEdit, self.sensitivity_dialArea.dial,
                                   self.face_dialArea.dial, self.gamma_dialArea.dial, self.top_dialArea.dial,
                                   self.bottom_dialArea.dial, self.left_dialArea.dial, self.right_dialArea.dial,
                                   self.folderLineEdit, self.destinationLineEdit, self.destinationButton,
                                   self.folderButton, self.extWidget.radioButton_1, self.extWidget.radioButton_2,
                                   self.extWidget.radioButton_3, self.extWidget.radioButton_4,
                                   self.extWidget.radioButton_5, self.extWidget.radioButton_6, self.cropButton,
                                   self.exposureCheckBox, self.mfaceCheckBox, self.tiltCheckBox))
        self.crop_worker.folder_started.connect(lambda: window_functions.enable_widget(self.cancelButton))

        # Folder end connection
        self.crop_worker.folder_finished.connect(
            lambda: window_functions.enable_widget(self.widthLineEdit, self.heightLineEdit, self.sensitivity_dialArea.dial,
                                  self.face_dialArea.dial, self.gamma_dialArea.dial, self.top_dialArea.dial,
                                  self.bottom_dialArea.dial, self.left_dialArea.dial, self.right_dialArea.dial,
                                  self.folderLineEdit, self.destinationLineEdit, self.destinationButton,
                                  self.folderButton, self.extWidget.radioButton_1, self.extWidget.radioButton_2,
                                  self.extWidget.radioButton_3, self.extWidget.radioButton_4,
                                  self.extWidget.radioButton_5, self.extWidget.radioButton_6, self.cropButton,
                                  self.exposureCheckBox, self.mfaceCheckBox, self.tiltCheckBox))
        self.crop_worker.folder_finished.connect(lambda: window_functions.disable_widget(self.cancelButton))
        self.crop_worker.folder_finished.connect(lambda: window_functions.show_message_box(self.destinationLineEdit))
        self.crop_worker.folder_progress.connect(lambda: self.update_progress(self.crop_worker.bar_value_f))

        self.retranslateUi()
        self.disable_buttons()
        window_functions.change_widget_state(False, self.cropButton, self.cancelButton)
        QtCore.QMetaObject.connectSlotsByName(self)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate('Form', 'Form'))
        self.folderLineEdit.setPlaceholderText(_translate('Form', 'Choose the folder you want to crop'))
        self.folderButton.setText(_translate('Form', 'Select Folder'))
        self.mfaceCheckBox.setText(_translate('Form', 'Multi-Face'))
        self.tiltCheckBox.setText(_translate('Form', 'Autotilt'))
        self.exposureCheckBox.setText(_translate('Form', 'Autocorrect'))
        self.destinationLineEdit.setPlaceholderText(
            _translate('Form', 'Choose where you want to save the cropped images'))
        self.destinationButton.setText(_translate('Form', 'Destination Folder'))

    def display_crop(self, selection: Optional[Path] = None) -> None:
        job = self.create_job(self.exposureCheckBox, self.mfaceCheckBox, self.tiltCheckBox)
        if selection is None:
            self.crop_worker.display_crop(job, self.folderLineEdit, self.imageWidget)
        else:
            self.crop_worker.display_crop(job, selection, self.imageWidget)

    def load_data(self) -> None:
        try:
            f_name =  self.folderLineEdit.text()
            self.file_model.setRootPath(f_name)
            self.treeView.setRootIndex(self.file_model.index(f_name))
            self.display_crop()
        except (IndexError, FileNotFoundError, ValueError, AttributeError):
            return None

    def reload_widgets(self) -> None:
        def callback(input_path: Path) -> None:
            if not input_path.as_posix():
                return None
            self.display_crop(input_path)
        if not self.widthLineEdit.text() or not self.heightLineEdit.text():
            return None
        if self.selection_state == FunctionTabSelectionState.NOT_SELECTED:
            return None
        if self.treeView.currentIndex().isValid():
            f_name = Path(self.file_model.filePath(self.treeView.currentIndex()))
        else:
            f_name = Path(self.folderLineEdit.text())
        callback(f_name)

    def disable_buttons(self) -> None:
        def all_filled(*line_edits: Union[PathLineEdit, NumberLineEdit, QtWidgets.QComboBox]) -> bool:
            x = all(edit.state == LineEditState.VALID_INPUT
                    for edit in line_edits if isinstance(edit, (PathLineEdit, NumberLineEdit)))
            y = all(edit.currentText() for edit in line_edits if isinstance(edit, QtWidgets.QComboBox))
            return x and y

        def update_widget_state(condition: bool, *widgets: QtWidgets.QWidget) -> None:
            for widget in widgets:
                window_functions.change_widget_state(condition, widget)

        # Folder logic
        update_widget_state(
            all_filled(self.folderLineEdit, self.destinationLineEdit, self.widthLineEdit, self.heightLineEdit),
            self.cropButton)

    def folder_process(self) -> None:
        job = self.create_job(self.exposureCheckBox, 
                              self.mfaceCheckBox, 
                              self.tiltCheckBox,
                              folder_path=self.folderLineEdit, 
                              destination=self.destinationLineEdit)
        self.run_batch_process(self.crop_worker.crop_dir, self.crop_worker.reset_f_task, job)