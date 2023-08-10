from os import startfile
from typing import Optional, Union
from pathlib import Path

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import QCheckBox, QMessageBox, QWidget, QPushButton, QHBoxLayout, QVBoxLayout

from .enums import FunctionType
from .dialog import UiDialog
from .literals import MediaIconAlias

def add_widgets(base_widget: Union[QHBoxLayout, QVBoxLayout], *widgets: QWidget) -> None:
    for widget in widgets:
        base_widget.addWidget(widget)

def uncheck_boxes(*checkboxes: QCheckBox) -> None:
    for checkbox in checkboxes:
        checkbox.setCheckState(Qt.CheckState.Unchecked)

def load_about_form() -> None:
    about_ui = UiDialog()
    about_ui.exec()

def show_message_box(destination: Path) -> None:
    msg_box = QMessageBox()
    msg_box.setWindowIcon(QIcon('resources\\logos\\logo.ico'))
    msg_box.setWindowTitle('Open Destination Folder')
    msg_box.setText('Open destination folder?')
    msg_box.setIcon(QMessageBox.Icon.Question)
    msg_box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
    
    match msg_box.exec():
        case QMessageBox.StandardButton.Yes:
            startfile(destination)
        case _: pass

def show_warning(function_type: FunctionType) -> int:
    msg_box = QMessageBox()
    msg_box.setWindowIcon(QIcon('resources\\logos\\logo.ico'))
    msg_box.setWindowTitle('Paths Match')
    msg_box.setIcon(QMessageBox.Icon.Warning)
    match function_type:
        case FunctionType.PHOTO:
            msg = 'This will overwrite the original.'
        case FunctionType.FOLDER | FunctionType.MAPPING:
            msg = 'If potential overwites are detected, the images will save to a new folder.'
        case FunctionType.VIDEO:
            msg = 'If potential overwites are detected, the frames will save to a new folder.'
        case FunctionType.FRAME:
            msg = 'This will overwrite any cropped frames with the same name.'
    msg_box.setText(f'The paths are the same.\n{msg}\nAre you OK to proceed?')
    msg_box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
    return msg_box.exec()

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
