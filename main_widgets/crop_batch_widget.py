from multiprocessing import Process
from typing import Optional, Callable, Any

from PyQt6 import QtCore, QtGui, QtWidgets

from core import Cropper, CustomDialWidget, ExtWidget, Job
from line_edits import NumberLineEdit, PathLineEdit
from .custom_crop_widget import CustomCropWidget


class CropBatchWidget(CustomCropWidget):
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
        self.cancelButton = QtWidgets.QPushButton()
        self.cancelButton.setMinimumSize(QtCore.QSize(0, 24))
        self.cancelButton.setMaximumSize(QtCore.QSize(16777215, 24))
        self.cancelButton.setText('')
        cancel_icon = QtGui.QIcon()
        cancel_icon.addPixmap(QtGui.QPixmap('resources\\icons\\cancel.svg'), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.cancelButton.setIcon(cancel_icon)
        self.cancelButton.setObjectName('cancelButton')
        self.progressBar = QtWidgets.QProgressBar()
        self.progressBar.setMinimumSize(QtCore.QSize(0, 12))
        self.progressBar.setMaximumSize(QtCore.QSize(16777215, 12))
        self.progressBar.setProperty('value', 0)
        self.progressBar.setTextVisible(False)

        self.folderLineEdit: PathLineEdit = PathLineEdit()
        self.folderLineEdit.setMinimumSize(QtCore.QSize(0, 24))
        self.folderLineEdit.setMaximumSize(QtCore.QSize(16777215, 24))
        self.folderLineEdit.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhUrlCharactersOnly)
        self.folderLineEdit.setObjectName('folderLineEdit')
        self.folderButton = QtWidgets.QPushButton()
        self.folderButton.setMinimumSize(QtCore.QSize(124, 24))
        self.folderButton.setMaximumSize(QtCore.QSize(16777215, 24))
        folder_icon = QtGui.QIcon()
        folder_icon.addPixmap(QtGui.QPixmap('resources\\icons\\folder.svg'), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.folderButton.setIcon(folder_icon)
        self.folderButton.setObjectName('folderButton')

    def update_progress(self, value: int) -> None:
        self.progressBar.setValue(value)
        QtWidgets.QApplication.processEvents()
    
    @staticmethod
    def run_batch_process(function: Callable[..., Any],
                          reset_worker_func: Callable[..., Any],
                          job: Job) -> None:
        reset_worker_func()
        process = Process(target=function, daemon=True, args=(job,))
        process.run()

    @staticmethod
    def cancel_button_operation(cancel_button: QtWidgets.QPushButton, *crop_buttons: QtWidgets.QPushButton):
        cancel_button.setDisabled(True)
        for crop_button in crop_buttons:
            crop_button.setEnabled(True)
