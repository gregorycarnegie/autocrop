from os import startfile
from pathlib import Path
from typing import Optional, Union

import cv2.typing as cvt
from PyQt6 import QtCore, QtGui, QtWidgets

from line_edits import LineEditState, NumberLineEdit, PathLineEdit
from .dialog import UiDialog
from .enums import FunctionType, GuiIcon
from .image_widget import ImageWidget
from .resource_path import ResourcePath


def setup_hbox(name: str, parent: Optional[QtWidgets.QWidget] = None) -> QtWidgets.QHBoxLayout:
    horizontal_layout = QtWidgets.QHBoxLayout(parent)
    horizontal_layout.setObjectName(name)
    return horizontal_layout


def setup_vbox(name: str, parent: Optional[QtWidgets.QWidget] = None) -> QtWidgets.QVBoxLayout:
    vertical_layout = QtWidgets.QVBoxLayout(parent)
    vertical_layout.setObjectName(name)
    return vertical_layout


def apply_size_policy(widget: QtWidgets.QWidget,
                      size_policy: QtWidgets.QSizePolicy,
                      min_size: QtCore.QSize = QtCore.QSize(0, 30),
                      max_size: QtCore.QSize = QtCore.QSize(16_777_215, 30)) -> None:
    size_policy.setHeightForWidth(widget.sizePolicy().hasHeightForWidth())
    widget.setSizePolicy(size_policy)
    widget.setMinimumSize(min_size)
    widget.setMaximumSize(max_size)


def create_main_button(name: str,
                       size_policy: QtWidgets.QSizePolicy,
                       icon_file: GuiIcon,
                       parent: QtWidgets.QWidget) -> QtWidgets.QPushButton:
    button = QtWidgets.QPushButton(parent)
    button.setObjectName(name)
    size_policy.setHeightForWidth(button.sizePolicy().hasHeightForWidth())
    button.setSizePolicy(size_policy)
    button.setMinimumSize(QtCore.QSize(0, 40))
    button.setMaximumSize(QtCore.QSize(16_777_215, 40))
    icon = QtGui.QIcon()
    icon.addFile(icon_file.value, QtCore.QSize(), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
    button.setIcon(icon)
    button.setIconSize(QtCore.QSize(18, 18))
    return button


def create_frame(name: str,
                 parent: QtWidgets.QWidget,
                 size_policy: QtWidgets.QSizePolicy) -> QtWidgets.QFrame:
    frame = QtWidgets.QFrame(parent)
    frame.setObjectName(name)
    size_policy.setHeightForWidth(frame.sizePolicy().hasHeightForWidth())
    frame.setSizePolicy(size_policy)
    frame.setStyleSheet(u"background: #1f2c33")
    frame.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
    frame.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
    return frame


def create_button_icon(icon_resource: GuiIcon,
                       size: QtCore.QSize = QtCore.QSize(),
                       mode: QtGui.QIcon.Mode = QtGui.QIcon.Mode.Normal,
                       state: QtGui.QIcon.State = QtGui.QIcon.State.Off) -> QtGui.QIcon:
    icon = QtGui.QIcon()
    icon.addFile(icon_resource.value, size, mode, state)
    return icon


def all_filled(*input_widgets: Union[PathLineEdit, NumberLineEdit, QtWidgets.QComboBox]) -> bool:
    x = all(widget.state == LineEditState.VALID_INPUT
            for widget in input_widgets if isinstance(widget, (PathLineEdit, NumberLineEdit)))
    y = all(widget.text() for widget in input_widgets if isinstance(widget, (PathLineEdit, NumberLineEdit)))
    z = all(widget.currentText() for widget in input_widgets if isinstance(widget, QtWidgets.QComboBox))
    return all((x, y, z))


def display_image_on_widget(image: cvt.MatLike, image_widget: ImageWidget) -> None:
    """
    Displays the specified image on the image widget.

    Args:
        image (cvt.MatLike): The image to display.
        image_widget (ImageWidget): The image widget to display the image on.

    Returns:`
        None

    Example:
        ```python
        image = cvt.imread('path/to/image.jpg')
        widget = ImageWidget()

        # Display the image on the widget
        display_image_on_widget(image, widget)
        ```
    """

    height, width, channel = image.shape
    bytes_per_line = channel * width
    q_image = QtGui.QImage(image.data.tobytes(), width, height, bytes_per_line, QtGui.QImage.Format.Format_BGR888)
    image_widget.setImage(QtGui.QPixmap.fromImage(q_image))


def uncheck_boxes(*checkboxes: QtWidgets.QCheckBox) -> None:
    """
    Unchecks the specified checkboxes.

    Args:
        *checkboxes (QtWidgets.QCheckBox): Variable number of checkboxes to uncheck.

    Returns:
        None

    Example:
        ```python
        checkbox1 = QtWidgets.QCheckBox()
        checkbox2 = QtWidgets.QCheckBox()
        checkbox3 = QtWidgets.QCheckBox()

        # Uncheck the checkboxes
        uncheck_boxes(checkbox1, checkbox2, checkbox3)
        ```
    """

    for checkbox in checkboxes:
        checkbox.setCheckState(QtCore.Qt.CheckState.Unchecked)


def load_about_form() -> None:
    """
    Loads and displays the `about` form.

    Returns:
        None

    Example:
        ```python
        # Load and display the `about` form
        load_about_form()
        ```
    """

    about_ui = UiDialog()
    about_ui.exec()


def initialise_message_box(window_title: str) -> QtWidgets.QMessageBox:
    """
    Initializes a message box with the specified window title.

    Args:
        window_title (str): The title of the message box window.

    Returns:
        QtWidgets.QMessageBox: The initialized message box object.

    Example:
        ```python
        title = 'Warning'

        # Initialize a message box with the specified window title
        message_box = initialise_message_box(title)
        ```
    """
    path = ResourcePath('resources\\logos\\logo.ico').meipass_path
    msg_box = QtWidgets.QMessageBox()
    msg_box.setWindowIcon(QtGui.QIcon(path))
    msg_box.setWindowTitle(window_title)
    return msg_box


def show_message_box(destination: Path) -> None:
    """
    Shows a message box with the option to open the destination folder.

    Args:
        destination (Path): The path of the destination folder.

    Returns:
        None

    Example:
        ```python
        destination_folder = Path('path/to/destination')

        # Show a message box to open the destination folder
        show_message_box(destination_folder)
        ```
    """
    msg_box = initialise_message_box('Open Destination Folder')
    msg_box.setText('Open destination folder?')
    msg_box.setIcon(QtWidgets.QMessageBox.Icon.Question)
    msg_box.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No)
    match msg_box.exec():
        case QtWidgets.QMessageBox.StandardButton.Yes:
            startfile(destination)
        case _:
            pass

def show_error_box(*messages: str) -> None:
    msg_box = initialise_message_box('Error')
    msg_box.setText('\n'.join(messages))
    msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
    msg_box.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
    msg_box.exec()

def generate_message(msg_box: QtWidgets.QMessageBox, message: str) -> None:
    """
    Generates a message for the message box with the specified text.

    Args:
        msg_box (QtWidgets.QMessageBox): The message box object.
        message (str): The message to include in the text.

    Returns:
        None

    Example:
        ```python
        message_box = QtWidgets.QMessageBox()
        text_message = 'This is a warning message.'

        # Generate a message for the message box
        generate_message(message_box, text_message)
        ```
    """

    msg_box.setText(f'The paths are the same.\n{message}\nAre you OK to proceed?')


def show_warning(function_type: FunctionType) -> int:
    """
    Shows a warning message box with a specific message based on the function type.

    Args:
        function_type (FunctionType): The type of the function.

    Returns:
        int: The result of the message box execution.

    Example:
        ```python
        function_type = FunctionType.PHOTO

        # Show a warning message box for the specified function type
        result = show_warning(function_type)
        ```
    """

    msg_box = initialise_message_box('Paths Match')
    msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
    match function_type:
        case FunctionType.PHOTO:
            generate_message(msg_box, 'This will overwrite the original.')
        case FunctionType.FOLDER | FunctionType.MAPPING | FunctionType.VIDEO:
            generate_message(msg_box, 'If potential overwrites are detected, the images will save to a new folder.')
        case FunctionType.FRAME:
            generate_message(msg_box, 'This will overwrite any cropped frames with the same name.')
    msg_box.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No)
    return msg_box.exec()


def disable_widget(*args: QtWidgets.QWidget) -> None:
    """
    Disables multiple widgets.

    Args:
        *args (QtWidgets.QWidget): Variable number of widgets to disable.

    Returns:
        None

    Example:
        ```python
        button1 = QtWidgets.QPushButton()
        button2 = QtWidgets.QPushButton()
        button3 = QtWidgets.QPushButton()

        # Disable the buttons
        disable_widget(button1, button2, button3)
        ```
    """

    for arg in args:
        arg.setDisabled(True)


def enable_widget(*args: QtWidgets.QWidget) -> None:
    """
    Enables multiple widgets.

    Args:
        *args (QtWidgets.QWidget): Variable number of widgets to enable.

    Returns:
        None

    Example:
        ```python
        button1 = QtWidgets.QPushButton()
        button2 = QtWidgets.QPushButton()
        button3 = QtWidgets.QPushButton()

        # Enable the buttons
        enable_widget(button1, button2, button3)
        ```
    """

    for arg in args:
        arg.setEnabled(True)


def change_widget_state(boolean: bool, *args: QtWidgets.QWidget) -> None:
    """
    Changes the state of multiple widgets based on a boolean value.

    Args:
        boolean (bool): The boolean value to determine the state of the widgets.
        *args (QtWidgets.QWidget): Variable number of widgets to change the state of.

    Returns:
        None

    Example:
        ```python
        button1 = QtWidgets.QPushButton()
        button2 = QtWidgets.QPushButton()
        button3 = QtWidgets.QPushButton()

        # Change the state of the buttons based on the boolean value
        change_widget_state(boolean=True, button1, button2, button3)
        ```
    """

    for arg in args:
        if boolean:
            arg.setEnabled(boolean)
        else:
            arg.setDisabled(not boolean)


def check_mime_data(event: Union[QtGui.QDragEnterEvent, QtGui.QDragMoveEvent]) -> None:
    """
    Checks the mime data of a drag enter or drag move event and accepts or ignores the event based on the presence of URLs in the mime data.

    Args:
        event (Union[QtGui.QDragEnterEvent, QtGui.QDragMoveEvent]): The drag enter or drag move event.

    Returns:
        None

    Example:
        ```python
        drag_event = QtGui.QDragEnterEvent()
        check_mime_data(drag_event)
        ```
    """

    if (mime_data := event.mimeData()) is None:
        return

    if mime_data.hasUrls():
        event.accept()
    else:
        event.ignore()


def setup_frame(name: str, *, parent: QtWidgets.QWidget) -> QtWidgets.QFrame:
    """
    Sets up and returns a QFrame with the specified name and parent.

    Args:
        name (str): The name of the QFrame object.
        parent (QtWidgets.QWidget): The parent widget for the QFrame.

    Returns:
        QtWidgets.QFrame: The created QFrame object.

    Example:
        ```python
        parent_widget = QtWidgets.QWidget()
        frame_name = 'myFrame'

        # Set up a QFrame with the specified name and parent
        frame = setup_frame(frame_name, parent=parent_widget)
        ```
    """

    frame = QtWidgets.QFrame(parent=parent)
    frame.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    frame.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
    frame.setObjectName(name)
    return frame

def create_media_button(parent: QtWidgets.QWidget, size_policy: QtWidgets.QSizePolicy,
                        *, name: str,
                        icon_resource: GuiIcon) -> QtWidgets.QPushButton:
    button = QtWidgets.QPushButton(parent)
    button.setObjectName(name)
    size_policy.setHeightForWidth(button.sizePolicy().hasHeightForWidth())
    button.setSizePolicy(size_policy)
    size = QtCore.QSize(40, 40)
    button.setMinimumSize(size)
    button.setMaximumSize(size)
    button.setBaseSize(size)
    icon = QtGui.QIcon()
    icon.addFile(icon_resource.value, QtCore.QSize(), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
    button.setIcon(icon)
    button.setIconSize(QtCore.QSize(24, 24))
    return button

def create_function_button(parent: QtWidgets.QWidget, size_policy: QtWidgets.QSizePolicy,
                            *, name: str,
                            icon_resource: GuiIcon) -> QtWidgets.QPushButton:
    button = QtWidgets.QPushButton(parent)
    button.setObjectName(name)
    size_policy.setHeightForWidth(button.sizePolicy().hasHeightForWidth())
    button.setSizePolicy(size_policy)
    button.setMinimumSize(QtCore.QSize(40, 40))
    button.setMaximumSize(QtCore.QSize(16_777_215, 40))
    icon = QtGui.QIcon()
    icon.addFile(icon_resource.value, QtCore.QSize(), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
    button.setIcon(icon)
    button.setIconSize(QtCore.QSize(18, 18))
    return button

def create_label(parent: QtWidgets.QWidget, size_policy: QtWidgets.QSizePolicy,
                    *, name: str,
                    icon_resource: GuiIcon) -> QtWidgets.QLabel:
    label = QtWidgets.QLabel(parent)
    label.setObjectName(name)
    size_policy.setHeightForWidth(label.sizePolicy().hasHeightForWidth())
    label.setSizePolicy(size_policy)
    size = QtCore.QSize(20, 20)
    label.setMinimumSize(size)
    label.setMaximumSize(size)
    label.setBaseSize(size)
    label.setPixmap(QtGui.QPixmap(icon_resource.value))
    label.setScaledContents(True)
    return label

def create_marker_button(parent: QtWidgets.QWidget, size_policy: QtWidgets.QSizePolicy,
                            name: str) -> QtWidgets.QPushButton:
    button = QtWidgets.QPushButton(parent)
    button.setObjectName(name)
    size_policy.setHeightForWidth(button.sizePolicy().hasHeightForWidth())
    button.setSizePolicy(size_policy)
    marker_button_size = QtCore.QSize(150, 20)
    button.setMinimumSize(marker_button_size)
    button.setMaximumSize(marker_button_size)
    return button

def get_qtime(position: int) -> QtCore.QTime:
    """
    Converts a given number of seconds into a QTime object.
    """
    minutes, seconds = divmod(round(position * .001), 60)
    hours, minutes = divmod(minutes, 60)
    return QtCore.QTime(hours, minutes, seconds)

def set_marker_time(button: QtWidgets.QPushButton, position: Union[int, float]) -> None:
    button.setText(get_qtime(position * 1000).toString())

def pos_from_marker(text: str) -> int:
    return sum(60 ** (2 - i) * int(x) for i, x in enumerate(text.split(':')))
