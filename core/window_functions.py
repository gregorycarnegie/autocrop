from os import startfile
from typing import Optional, Union

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import QCheckBox, QLineEdit, QMessageBox, QWidget, QPushButton, QHBoxLayout, QVBoxLayout

from .dialog import UiDialog
from .literals import MediaIconAlias

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

def create_media_button(name: str,
                        icon_resource: MediaIconAlias,
                        layout: Union[QHBoxLayout, QVBoxLayout],
                        size: int = 48,
                        icon_size: int = 32,
                        icon_mode: QIcon.Mode = QIcon.Mode.Normal,
                        icon_state: QIcon.State = QIcon.State.Off,
                        parent: Optional[QWidget] = None) -> QPushButton:
    playButton = QPushButton(parent=parent)
    playButton.setEnabled(True)
    playButton.setMinimumSize(QSize(size, size))
    playButton.setMaximumSize(QSize(size, size))
    playButton.setText('')
    icon = QIcon()
    icon.addPixmap(QPixmap(f'resources\\icons\\multimedia_{icon_resource}.svg'), icon_mode, icon_state)
    playButton.setIcon(icon)
    playButton.setIconSize(QSize(icon_size, icon_size))
    playButton.setObjectName(name)
    layout.addWidget(playButton)
    return playButton

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
