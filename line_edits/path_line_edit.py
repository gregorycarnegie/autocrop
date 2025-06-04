import contextlib
from pathlib import Path

import autocrop_rs.security as r_sec  # type: ignore
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMainWindow, QWidget

from file_types import FileCategory, file_manager

from .custom_line_edit import CustomLineEdit
from .enums import PathType
from .file_path_validator import FilePathValidator


class PathLineEdit(CustomLineEdit):
    """
    Represents a PathLineEdit class that inherits from the CustomLineEdit class.

    A custom line edit widget for handling file paths.

    Args:
        path_type (PathType): The type of path to be validated. Default to PathType.FOLDER.
        parent (Optional[QWidget]): The parent widget. Defaults to None.

    Methods:
        insert_clipboard_path(self) -> None: Inserts the clipboard path into the text input.
        validate_path(self) -> None: Validates the path entered in the text input based on the selected path type.
        is_valid_image(path: Path) -> bool: Checks if the given path is a valid image file.
        is_valid_table(path: Path) -> bool: Checks if the given path is a valid table file.
        is_valid_video(path: Path) -> bool: Checks if the given path is a valid video file.

    Returns:
        None
    """

    def __init__(self, path_type: PathType = PathType.FOLDER, parent: QWidget | None = None):
        super().__init__(parent)
        self.setInputMethodHints(Qt.InputMethodHint.ImhUrlCharactersOnly)
        self.setValidator(FilePathValidator(parent=self))
        self.path_type = path_type
        self.textChanged.connect(self.insert_clipboard_path)

    def insert_clipboard_path(self):
        """
        Removes the quotation marks from the passed path
        """
        text = self.text()

        try:
            self.blockSignals(True)
            if sanitized := r_sec.sanitize_path(text):
                self.setText(sanitized)
                self.validate_path()
            self.blockSignals(False)
        except r_sec.PathSecurityError:
            self.set_invalid_color()
            self.update_style()

    def validate_path(self) -> None:
        """
        Validates the path entered in the text input based on the selected path type.
        Also triggers button state updates.
        """
        if not self.text():
            self.set_invalid_color()
            self.update_style()
            return

        # file_path = Path(self.text())
        self.color_logic(self.is_valid_path(Path(self.text())))
        self.update_style()

        # Find the main window to trigger button state updates
        parent = self.parent()
        while parent and not isinstance(parent, QMainWindow):
            parent = parent.parent()

        if parent:
            # Try to find the current tab and update button states
            with contextlib.suppress(AttributeError, IndexError):
                tab_index = parent.function_tabWidget.currentIndex()  # type: ignore
                current_tab = None

                # Get the current tab widget
                if tab_index == 0 and hasattr(parent, 'photo_tab_widget'):
                    current_tab = parent.photo_tab_widget  # type: ignore
                elif tab_index == 1 and hasattr(parent, 'folder_tab_widget'):
                    current_tab = parent.folder_tab_widget  # type: ignore
                elif tab_index == 2 and hasattr(parent, 'mapping_tab_widget'):
                    current_tab = parent.mapping_tab_widget  # type: ignore
                elif tab_index == 3 and hasattr(parent, 'video_tab_widget'):
                    current_tab = parent.video_tab_widget  # type: ignore

                # Update button states if possible
                if current_tab and hasattr(current_tab, 'disable_buttons'):
                    current_tab.disable_buttons()

    def focusInEvent(self, event) -> None:
        """
        Override focus in event to update clear button visibility

        Args:
            event: The focus event

        Returns:
            None
        """
        super().focusInEvent(event)
        # Ensure clear button visibility is correct
        self.update_clear_button(self.text())

    def focusOutEvent(self, event) -> None:
        """
        Override focus out event to update clear button visibility

        Args:
            event: The focus event

        Returns:
            None
        """
        super().focusOutEvent(event)
        # Ensure clear button visibility is correct
        self.update_clear_button(self.text())

    def is_valid_path(self, path: Path) -> bool:
        """
        Checks if the given path is valid.

        Args:
            path (Path): The path to the file/folder.

        Returns:
            bool: True if the path is valid, False otherwise.
        """
        is_file = path.is_file()

        match self.path_type:
            case PathType.IMAGE:
                return is_file and (file_manager.is_valid_type(path, FileCategory.PHOTO) or
                                    file_manager.is_valid_type(path, FileCategory.RAW))
            case PathType.TABLE:
                return is_file and file_manager.is_valid_type(path, FileCategory.TABLE)
            case PathType.VIDEO:
                return is_file and file_manager.is_valid_type(path, FileCategory.VIDEO)
            case PathType.FOLDER:
                return path.is_dir()
            case _:
                return False

    def set_path_type(self, path_type: PathType) -> None:
        """
        Sets the path type for the line edit and updates validation accordingly.

        Args:
            path_type (PathType): The type of path to validate against.
        """
        self.path_type = path_type
        # Re-validate with a new path type
        self.validate_path()
