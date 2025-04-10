from functools import cache, partial
import os
from pathlib import Path
from typing import Optional, Union

import cv2.typing as cvt
from PyQt6 import QtCore, QtGui, QtWidgets

from core.enums import FunctionType
from .dialog import UiDialog
from .enums import GuiIcon
from .image_widget import ImageWidget

def register_button_dependencies(widget, button: QtWidgets.QPushButton,
                                 dependent_widgets: set[QtWidgets.QWidget]) -> None:
    """
    Register button dependencies with path validation support

    Args:
        button: The button whose enabled state depends on other widgets
        dependent_widgets: Set of widgets that must be valid for the button to be enabled
    """
    widget._button_dependencies[button] = dependent_widgets

    # Get the tab widget (parent of the button)
    tab_widget = button.parent()
    while tab_widget: # and not isinstance(tab_widget, UiCropWidget):
        tab_widget = tab_widget.parent()

    if tab_widget:
        # Register a custom path validation handler
        widget.register_validation_handler(button, lambda: check_paths_valid(tab_widget))

    widget.update_button_states()

def check_paths_valid(tab_widget) -> bool:
    """
    Check if the paths stored in the tab widget are valid

    Args:
        tab_widget: The tab widget to check

    Returns:
        bool: True if all required paths are valid
    """
    # Get the main window
    main_window = tab_widget.parent().parent().parent()

    # Check the current tab type
    current_index = main_window.function_tabWidget.currentIndex()

    # For all tabs, we need at least input_path and destination_path
    input_valid = bool(tab_widget.input_path) and Path(tab_widget.input_path).exists()
    dest_valid = bool(tab_widget.destination_path) and Path(tab_widget.destination_path).exists()

    # For mapping tab, we also need table_path
    if current_index == FunctionType.MAPPING:
        table_valid = bool(tab_widget.table_path) and Path(tab_widget.table_path).exists()
        return input_valid and dest_valid and table_valid

    # For other tabs, just check input and destination
    return input_valid and dest_valid

def sanitize_path(path_str: str) -> Optional[str]:
    """Sanitize path input to prevent path traversal attacks."""
    # Remove control characters and normalize
    path_str = ''.join(c for c in path_str if c.isprintable())
    
    # Normalize path separators
    path_str = path_str.replace('\\', '/').replace('//', '/')
    
    # Remove any attempts at directory traversal
    while '..' in path_str:
        path_str = path_str.replace('..', '')
    
    path = Path(path_str)
    if not path.exists() or not os.access(path, os.R_OK):
        show_error_box("Selected path is not accessible")
        return

    return path_str

def setup_combobox(combobox: QtWidgets.QComboBox,
                   layout: Union[QtWidgets.QHBoxLayout, QtWidgets.QHBoxLayout],
                   policy: QtWidgets.QSizePolicy,
                   name: str) -> None:
    """Set up the combo boxes"""
    combobox.setObjectName(name)
    policy.setHeightForWidth(combobox.sizePolicy().hasHeightForWidth())
    combobox.setSizePolicy(policy)
    combobox.setMinimumSize(QtCore.QSize(0, 40))
    combobox.setMaximumSize(QtCore.QSize(16_777_215, 40))
    layout.addWidget(combobox)


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
    icon.addFile(icon_file, QtCore.QSize(), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
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
    icon.addFile(icon_resource, size, mode, state)
    return icon


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
    image_widget.setImage(q_image)


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


@cache
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
    path = GuiIcon.ICON
    msg_box = QtWidgets.QMessageBox()
    msg_box.setWindowIcon(QtGui.QIcon(path))
    msg_box.setWindowTitle(window_title)
    return msg_box

def create_message_box(title: str, icon: QtWidgets.QMessageBox.Icon, 
                     buttons: QtWidgets.QMessageBox.StandardButton = QtWidgets.QMessageBox.StandardButton.Ok) -> QtWidgets.QMessageBox:
    """Factory function to create message boxes with standard settings"""
    msg_box = initialise_message_box(title)
    msg_box.setIcon(icon)
    msg_box.setStandardButtons(buttons)
    return msg_box

# Create specialized message box creators using partial
create_error_box = partial(
    create_message_box, 
    title='Error', 
    icon=QtWidgets.QMessageBox.Icon.Warning
)

create_warning_box = partial(
    create_message_box, 
    title='Paths Match', 
    icon=QtWidgets.QMessageBox.Icon.Warning, 
    buttons=QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
)

create_question_box = partial(
    create_message_box, 
    title='Open Destination Folder', 
    icon=QtWidgets.QMessageBox.Icon.Question,
    buttons=QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
)

def show_message_box(destination: Path) -> None:
    """Shows a message box with the option to open the destination folder."""
    msg_box = create_question_box()
    msg_box.setText('Open destination folder?')
    if msg_box.exec() == QtWidgets.QMessageBox.StandardButton.Yes:
        os.startfile(destination)

def show_error_box(*messages: str) -> None:
    """Shows an error message box with the given messages."""
    msg_box = create_error_box()
    msg_box.setText('\n'.join(messages))
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
    """Shows a warning message based on function type."""
    msg_box = create_warning_box()
    
    # Map function types to warning messages
    warnings = {
        FunctionType.PHOTO: 'This will overwrite the original.',
        FunctionType.FOLDER: 'If potential overwrites are detected, the images will save to a new folder.',
        FunctionType.MAPPING: 'If potential overwrites are detected, the images will save to a new folder.',
        FunctionType.VIDEO: 'If potential overwrites are detected, the images will save to a new folder.',
        FunctionType.FRAME: 'This will overwrite any cropped frames with the same name.'
    }
    
    # Get the appropriate message or use a default
    message = warnings.get(function_type, 'This operation may overwrite existing files.')
    generate_message(msg_box, message)
    
    return msg_box.exec()


def disable_widget(*args: QtWidgets.QWidget) -> None:
    """
    Disables multiple widgets with improved state handling.

    Args:
        *args (QtWidgets.QWidget): Variable number of widgets to disable.
    """
    for arg in args:
        arg.blockSignals(True)  # Block signals during state change
        arg.setDisabled(True)
        arg.blockSignals(False)  # Unblock signals
        arg.repaint()  # Force immediate repaint

def enable_widget(*args: QtWidgets.QWidget) -> None:
    """
    Enables multiple widgets with improved state handling.

    Args:
        *args (QtWidgets.QWidget): Variable number of widgets to enable.
    """
    for arg in args:
        arg.blockSignals(True)  # Block signals during state change
        arg.setEnabled(True)
        arg.blockSignals(False)  # Unblock signals
        arg.repaint()  # Force immediate repaint

def change_widget_state(boolean: bool, *args: QtWidgets.QWidget) -> None:
    """
    Changes the state of multiple widgets based on a boolean value with improved state handling.

    Args:
        boolean (bool): The boolean value to determine the state of the widgets.
        *args (QtWidgets.QWidget): Variable number of widgets to change the state of.
    """
    for arg in args:
        arg.blockSignals(True)  # Block signals during state change
        if boolean:
            arg.setEnabled(boolean)
        else:
            arg.setDisabled(not boolean)
        arg.blockSignals(False)  # Unblock signals
        arg.repaint()  # Force immediate repaint
    
    # Process events to ensure UI updates
    QtWidgets.QApplication.processEvents()

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
    icon.addFile(icon_resource, QtCore.QSize(), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
    button.setIcon(icon)
    button.setIconSize(QtCore.QSize(24, 24))
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

@cache
def get_qtime(position: int) -> QtCore.QTime:
    """
    Converts a given number of seconds into a QTime object.
    """
    minutes, seconds = divmod(round(position * .001), 60)
    hours, minutes = divmod(minutes, 60)
    return QtCore.QTime(hours, minutes, seconds)

def set_marker_time(button: QtWidgets.QPushButton, position: Union[int, float]) -> None:
    button.setText(get_qtime(position * 1000).toString())

@cache
def pos_from_marker(text: str) -> int:
    return sum(60 ** (2 - i) * int(x) for i, x in enumerate(text.split(':')))
