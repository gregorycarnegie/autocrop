from os import startfile

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QCheckBox, QLineEdit, QMessageBox, QWidget

from .cropper import Cropper
from .enums import Terminator
from .dialog import UiDialog

def uncheck_boxes(*checkboxes: QCheckBox) -> None:
    for checkbox in checkboxes:
        checkbox.setCheckState(Qt.CheckState.Unchecked)

def load_about_form() -> None:
    about_ui = UiDialog()
    about_ui.exec()

def show_message_box(destination: QLineEdit) -> None:
    def message_button(answer: QMessageBox) -> None:
        if answer.text() == '&Yes':
            startfile(destination.text())

    def helper_function(msg_box: QMessageBox) -> None:
        msg_box.setWindowTitle('Open Destination Folder')
        msg_box.setText('Open destination folder?')
        msg_box.setIcon(QMessageBox.Icon.Question)
        msg_box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        msg_box.buttonClicked.connect(message_button)
        x = msg_box.exec()

    msg = QMessageBox()
    helper_function(msg)

def disable_widget(*args: QWidget) -> None:
    for arg in args:
        arg.setDisabled(True)

def enable_widget(*args: QWidget) -> None:
    for arg in args:
        arg.setEnabled(True)

def change_widget_state(boolean: bool, *args: QWidget) -> None:
    for arg in args:
        if boolean:
            arg.setEnabled(boolean)
        else:
            arg.setDisabled(not boolean)

def terminate(cropper: Cropper, series: Terminator) -> None:
    if series == Terminator.END_FOLDER_TASK:
        cropper.end_f_task = True
    elif series == Terminator.END_MAPPING_TASK:
        cropper.end_m_task = True
    elif series == Terminator.END_VIDEO_TASK:
        cropper.end_v_task = True
