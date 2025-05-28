import platform
import subprocess
from functools import cache, partial
from pathlib import Path

import autocrop_rs.security as r_sec  # type: ignore
import cv2.typing as cvt
from PyQt6.QtCore import QSize, QTime
from PyQt6.QtGui import QDragEnterEvent, QDragMoveEvent, QIcon, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from core.enums import FunctionType

from .dialog import UiDialog
from .enums import GuiIcon
from .image_widget import ImageWidget


def register_button_dependencies(
        widget,
        button: QPushButton,
        dependent_widgets: set[QWidget]
) -> None:
    """
    Register button dependencies with path validation support

    Args:
        widget: TabStateManager
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


def sanitize_path(path_str: str) -> str:
    """
    Sanitize path input to prevent security vulnerabilities.
    Uses the Rust implementation for better security guarantees.
    """
    if not path_str:
        return ''

    try:
        return r_sec.sanitize_path(path_str)
    except r_sec.PathSecurityError as e:
        # Use get_safe_error_message to sanitize the error message
        safe_msg = r_sec.get_safe_error_message(str(e))
        show_error_box(f"Path security error: {safe_msg}")
        return ''
    except Exception as e:
        show_error_box(f"Invalid path:\n {e}")
        return ''


def setup_combobox(
        combobox: QComboBox,
        layout: QHBoxLayout | QHBoxLayout,
        policy: QSizePolicy,
        name: str
) -> None:
    """Set up the combo boxes"""
    combobox.setObjectName(name)
    policy.setHeightForWidth(combobox.sizePolicy().hasHeightForWidth())
    combobox.setSizePolicy(policy)
    combobox.setMinimumSize(QSize(0, 40))
    combobox.setMaximumSize(QSize(16_777_215, 40))
    layout.addWidget(combobox)


def setup_hbox(name: str, parent: QWidget | None = None) -> QHBoxLayout:
    horizontal_layout = QHBoxLayout(parent)
    horizontal_layout.setObjectName(name)
    return horizontal_layout


def setup_vbox(name: str, parent: QWidget | None = None) -> QVBoxLayout:
    vertical_layout = QVBoxLayout(parent)
    vertical_layout.setObjectName(name)
    return vertical_layout


def apply_size_policy(
        widget: QWidget,
        size_policy: QSizePolicy,
        min_size: QSize = QSize(0, 30),
        max_size: QSize = QSize(16_777_215, 30)
) -> None:
    size_policy.setHeightForWidth(widget.sizePolicy().hasHeightForWidth())
    widget.setSizePolicy(size_policy)
    widget.setMinimumSize(min_size)
    widget.setMaximumSize(max_size)


def create_main_button(
        name: str,
        size_policy: QSizePolicy,
        icon_file: GuiIcon,
        parent: QWidget
) -> QPushButton:
    button = QPushButton(parent)
    button.setObjectName(name)
    size_policy.setHeightForWidth(button.sizePolicy().hasHeightForWidth())
    button.setSizePolicy(size_policy)
    button.setMinimumSize(QSize(0, 40))
    button.setMaximumSize(QSize(16_777_215, 40))
    icon = QIcon()
    icon.addFile(icon_file, QSize(), QIcon.Mode.Normal, QIcon.State.Off)
    button.setIcon(icon)
    button.setIconSize(QSize(18, 18))
    return button


def create_frame(
        name: str,
        parent: QWidget,
        size_policy: QSizePolicy
) -> QFrame:
    frame = QFrame(parent)
    frame.setObjectName(name)
    size_policy.setHeightForWidth(frame.sizePolicy().hasHeightForWidth())
    frame.setSizePolicy(size_policy)
    frame.setStyleSheet("background: #1f2c33")
    frame.setFrameShape(QFrame.Shape.StyledPanel)
    frame.setFrameShadow(QFrame.Shadow.Raised)
    return frame


def create_button_icon(
        icon_resource: GuiIcon,
        size: QSize = QSize(),
        mode: QIcon.Mode = QIcon.Mode.Normal,
        state: QIcon.State = QIcon.State.Off
) -> QIcon:
    icon = QIcon()
    icon.addFile(icon_resource, size, mode, state)
    return icon


def display_image_on_widget(image: cvt.MatLike, image_widget: ImageWidget) -> None:
    """
    Displays the specified image on the image widget.

    Args:
        image (cvt.MatLike): The image to display.
        image_widget (ImageWidget): The image widget to display the image on.

    Returns:
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
    q_image = QImage(image.data.tobytes(), width, height, bytes_per_line, QImage.Format.Format_BGR888)
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


def initialise_message_box(window_title: str) -> QMessageBox:
    """
    Initializes a message box with the specified window title.

    Args:
        window_title (str): The title of the message box window.

    Returns:
        QMessageBox: The initialized message box object.

    Example:
        ```python
        title = 'Warning'

        # Initialize a message box with the specified window title
        message_box = initialise_message_box(title)
        ```
    """
    msg_box = QMessageBox()
    msg_box.setWindowIcon(QIcon(GuiIcon.ICON))
    msg_box.setWindowTitle(window_title)
    return msg_box


def create_message_box(
        title: str,
        icon: QMessageBox.Icon,
        buttons: QMessageBox.StandardButton = QMessageBox.StandardButton.Ok
) -> QMessageBox:
    """Factory function to create message boxes with standard settings"""
    msg_box = initialise_message_box(title)
    msg_box.setIcon(icon)
    msg_box.setStandardButtons(buttons)
    return msg_box


# Create specialized message box creators using partial
create_error_box = partial(
    create_message_box,
    title='Error',
    icon=QMessageBox.Icon.Warning
)


create_warning_box = partial(
    create_message_box,
    title='Paths Match',
    icon=QMessageBox.Icon.Warning,
    buttons=QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
)


create_question_box = partial(
    create_message_box,
    title='Open Destination Folder',
    icon=QMessageBox.Icon.Question,
    buttons=QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
)


def show_message_box(destination: Path) -> None:
    """Shows a message box with the option to open the destination folder."""
    msg_box = create_question_box()
    msg_box.setText('Open destination folder?')
    if msg_box.exec() == QMessageBox.StandardButton.Yes:
        # Replace the insecure os.startfile with a safer alternative
        try:
            # Use platform-specific methods to open file explorer
            # Make sure destination path is absolute and normalized
            abs_path = str(destination.resolve())

            # Use different commands based on operating system
            if platform.system() == "Windows":
                # On Windows, use subprocess.run with explorer.exe
                subprocess.run(['explorer', abs_path], check=False, shell=False)
            elif platform.system() == "Darwin":  # macOS
                # On macOS, use 'open' command
                subprocess.run(['open', abs_path], check=False, shell=False)
            else:  # Linux and other Unix-like
                # On Linux, try xdg-open, if it fails fall back to other options
                try:
                    subprocess.run(['xdg-open', abs_path], check=False, shell=False)
                except FileNotFoundError:
                    # Try other common file managers
                    for cmd in ['gnome-open', 'kde-open', 'exo-open']:
                        try:
                            subprocess.run([cmd, abs_path], check=False, shell=False)
                            break
                        except FileNotFoundError:
                            continue
        except Exception as e:
            # Show a simple error message
            error_box = create_error_box()
            error_box.setText(f"Error opening destination folder: {e}")
            error_box.exec()


def show_error_box(*messages: str) -> None:
    """Shows an error message box with the given messages."""
    msg_box = create_error_box()
    msg_box.setText('\n'.join(messages))
    msg_box.exec()


def generate_message(msg_box: QMessageBox, message: str) -> None:
    """
    Generates a message for the message box with the specified text.

    Args:
        msg_box (QMessageBox): The message box object.
        message (str): The message to include in the text.

    Returns:
        None

    Example:
        ```python
        message_box = QMessageBox()
        text_message = 'This is a warning message.'

        # Generate a message for the message box
        generate_message(message_box, text_message)
        ```
    """

    msg_box.setText(f'The paths are the same.\n{message}\nAre you OK to proceed?')


def show_warning(function_type: FunctionType) -> int:
    """Shows a warning message based on the function type."""
    msg_box = create_warning_box()

    match function_type:
        case FunctionType.PHOTO:
            message = 'This will overwrite the original.'
        case FunctionType.FOLDER | FunctionType.MAPPING | FunctionType.VIDEO:
            message = 'If potential overwrites are detected, the images will save to a new folder.'
        case FunctionType.FRAME:
            message = 'This will overwrite any cropped frames with the same name.'

    generate_message(msg_box, message)

    return msg_box.exec()


def disable_widget(*args: QWidget) -> None:
    """
    Disables multiple widgets with improved state handling.

    Args:
        *args (QWidget): Variable number of widgets to disable.
    """
    for arg in args:
        arg.blockSignals(True)  # Block signals during state change
        arg.setDisabled(True)
        arg.blockSignals(False)  # Unblock signals
        arg.repaint()  # Force immediate repaint


def enable_widget(*args: QWidget) -> None:
    """
    Enables multiple widgets with improved state handling.

    Args:
        *args (QWidget): Variable number of widgets to enable.
    """
    for arg in args:
        arg.blockSignals(True)  # Block signals during state change
        arg.setEnabled(True)
        arg.blockSignals(False)  # Unblock signals
        arg.repaint()  # Force immediate repaint


def change_widget_state(boolean: bool, *args: QWidget) -> None:
    """
    Changes the state of multiple widgets based on a boolean value with improved state handling.

    Args:
        boolean (bool): The boolean value to determine the state of the widgets.
        *args (QWidget): Variable number of widgets to change the state of.
    """
    if not args:
        return

    for arg in args:
        arg.blockSignals(True)  # Block signals during state change
        if boolean:
            arg.setEnabled(boolean)
        else:
            arg.setDisabled(not boolean)
        arg.blockSignals(False)  # Unblock signals
        arg.repaint()  # Force immediate repaint

    # Process events to ensure UI updates
    QApplication.processEvents()


def check_mime_data(event: QDragEnterEvent | QDragMoveEvent) -> None:
    """
    Checks the mime data of a drag enter or drag move event and accepts/ignores the event based on the presence of URLs.

    Args:
        event (Union[QDragEnterEvent, QDragMoveEvent]): The drag enter or drag move event.

    Returns:
        None

    Example:
        ```python
        drag_event = QDragEnterEvent()
        check_mime_data(drag_event)
        ```
    """

    if (mime_data := event.mimeData()) is None:
        return

    if mime_data.hasUrls():
        event.accept()
    else:
        event.ignore()


def setup_frame(name: str, *, parent: QWidget) -> QFrame:
    """
    Sets up and returns a QFrame with the specified name and parent.

    Args:
        name (str): The name of the QFrame object.
        parent (QWidget): The parent widget for the QFrame.

    Returns:
        QFrame: The created QFrame object.

    Example:
        ```python
        parent_widget = QWidget()
        frame_name = 'myFrame'

        # Set up a QFrame with the specified name and parent
        frame = setup_frame(frame_name, parent=parent_widget)
        ```
    """

    frame = QFrame(parent=parent)
    frame.setFrameShape(QFrame.Shape.NoFrame)
    frame.setFrameShadow(QFrame.Shadow.Raised)
    frame.setObjectName(name)
    return frame


def create_media_button(parent: QWidget, size_policy: QSizePolicy,
                        *, name: str,
                        icon_resource: GuiIcon) -> QPushButton:
    button = QPushButton(parent)
    button.setObjectName(name)
    size_policy.setHeightForWidth(button.sizePolicy().hasHeightForWidth())
    button.setSizePolicy(size_policy)
    size = QSize(40, 40)
    button.setMinimumSize(size)
    button.setMaximumSize(size)
    button.setBaseSize(size)
    icon = QIcon()
    icon.addFile(icon_resource, QSize(), QIcon.Mode.Normal, QIcon.State.Off)
    button.setIcon(icon)
    button.setIconSize(QSize(24, 24))
    return button


def create_label(parent: QWidget, size_policy: QSizePolicy,
                    *, name: str,
                    icon_resource: GuiIcon) -> QLabel:
    label = QLabel(parent)
    label.setObjectName(name)
    size_policy.setHeightForWidth(label.sizePolicy().hasHeightForWidth())
    label.setSizePolicy(size_policy)
    size = QSize(20, 20)
    label.setMinimumSize(size)
    label.setMaximumSize(size)
    label.setBaseSize(size)
    label.setPixmap(QPixmap(icon_resource.value))
    label.setScaledContents(True)
    return label


def create_marker_button(parent: QWidget, size_policy: QSizePolicy,
                            name: str) -> QPushButton:
    button = QPushButton(parent)
    button.setObjectName(name)
    size_policy.setHeightForWidth(button.sizePolicy().hasHeightForWidth())
    button.setSizePolicy(size_policy)
    marker_button_size = QSize(150, 20)
    button.setMinimumSize(marker_button_size)
    button.setMaximumSize(marker_button_size)
    return button


@cache
def get_qtime(position: int) -> QTime:
    """
    Converts a given number of seconds into a QTime object.
    """
    minutes, seconds = divmod(round(position * .001), 60)
    hours, minutes = divmod(minutes, 60)
    return QTime(hours, minutes, seconds)


def set_marker_time(button: QPushButton, position: int | float) -> None:
    button.setText(get_qtime(position * 1000).toString())


@cache
def pos_from_marker(text: str) -> int:
    return sum(60 ** (2 - i) * int(x) for i, x in enumerate(text.split(':')))
