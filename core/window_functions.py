from os import startfile
from typing import Optional, Union
from pathlib import Path

import cv2.typing as cvt
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QIcon, QImage, QPixmap
from PyQt6.QtWidgets import QCheckBox, QHBoxLayout, QMessageBox, QPushButton, QVBoxLayout, QWidget

from .enums import FunctionType
from .dialog import UiDialog
from .image_widget import ImageWidget
from .literals import MediaIconAlias


def display_image_on_widget(image: cvt.MatLike, image_widget: ImageWidget) -> None:
    """
    Display an OpenCV image on a Qt ImageWidget.

    Args:
    - image (cvt.MatLike): OpenCV image to display.
    - image_widget (ImageWidget): Qt widget where the image should be displayed.
    """
    height, width, channel = image.shape
    bytes_per_line = channel * width
    q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_BGR888)
    image_widget.setImage(QPixmap.fromImage(q_image))


def add_widgets(base_widget: Union[QHBoxLayout, QVBoxLayout], *widgets: QWidget) -> None:
    for widget in widgets:
        base_widget.addWidget(widget)


def uncheck_boxes(*checkboxes: QCheckBox) -> None:
    for checkbox in checkboxes:
        checkbox.setCheckState(Qt.CheckState.Unchecked)


def load_about_form() -> None:
    about_ui = UiDialog()
    about_ui.exec()


def initialise_message_box(window_title: str) -> QMessageBox:
    msg_box = QMessageBox()
    msg_box.setWindowIcon(QIcon('resources\\logos\\logo.ico'))
    msg_box.setWindowTitle(window_title)
    return msg_box


def show_message_box(destination: Path) -> None:
    msg_box = initialise_message_box('Open Destination Folder')
    msg_box.setText('Open destination folder?')
    msg_box.setIcon(QMessageBox.Icon.Question)
    msg_box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
    match msg_box.exec():
        case QMessageBox.StandardButton.Yes:
            startfile(destination)
        case _:
            pass


def generate_message(msg_box: QMessageBox, message: str) -> None:
    msg_box.setText(f'The paths are the same.\n{message}\nAre you OK to proceed?')


def show_warning(function_type: FunctionType) -> int:
    msg_box = initialise_message_box('Paths Match')
    msg_box.setIcon(QMessageBox.Icon.Warning)
    match function_type:
        case FunctionType.PHOTO:
            generate_message(msg_box, 'This will overwrite the original.')
        case FunctionType.FOLDER | FunctionType.MAPPING:
            generate_message(msg_box, 'If potential overwrites are detected, the images will save to a new folder.')
        case FunctionType.VIDEO:
            generate_message(msg_box, 'If potential overwrites are detected, the frames will save to a new folder.')
        case FunctionType.FRAME:
            generate_message(msg_box, 'This will overwrite any cropped frames with the same name.')
    msg_box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
    return msg_box.exec()


def create_media_button(name: str,
                        icon_resource: MediaIconAlias,
                        layout: Union[QHBoxLayout, QVBoxLayout],
                        size: int = 88,
                        icon_size: int = 58,
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
