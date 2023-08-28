from pathlib import Path
from typing import Optional

from PyQt6 import QtCore, QtWidgets

from file_types import Photo, Table, Video
from .custom_line_edit import CustomLineEdit
from .enums import PathType
from .file_path_validator import FilePathValidator


class PathLineEdit(CustomLineEdit):
    """
Represents a PathLineEdit class that inherits from the CustomLineEdit class.
    
A custom line edit widget for handling file paths.

Args:
    path_type (PathType): The type of path to be validated. Defaults to PathType.FOLDER.
    parent (Optional[QtWidgets.QWidget]): The parent widget. Defaults to None.

Attributes:
    INVALID_CHARS_PATTERN (ClassVar[Pattern[str]]): A regular expression pattern to match invalid characters in a file path.

Methods:
    insert_clipboard_path(self, text: str) -> None: Inserts the clipboard path into the text input.
    validate_path(self) -> None: Validates the path entered in the text input based on the selected path type.
    is_valid_image(path: Path) -> bool: Checks if the given path is a valid image file.
    is_valid_table(path: Path) -> bool: Checks if the given path is a valid table file.
    is_valid_video(path: Path) -> bool: Checks if the given path is a valid video file.

Returns:
    None
"""

    def __init__(self, path_type: PathType = PathType.FOLDER, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhUrlCharactersOnly)
        self.setValidator(FilePathValidator(parent=self))
        self.path_type = path_type
        self.textChanged.connect(self.insert_clipboard_path)

    def insert_clipboard_path(self, text: str) -> None:
        """
Inserts the clipboard path into the text input with quotation marks removed.

Args:
    self: The PathLineEdit instance.
    text (str): The text to be inserted.

Returns:
    None
"""

        if (text.startswith('"') & text.endswith('"')) ^ (text.startswith("'") & text.endswith("'")):
            text = text[1:-1]
        self.setText(text)

    def validate_path(self) -> None:
        """
Validates the path entered in the text input based on the selected path type.

Args:
    self: The PathLineEdit instance.

Returns:
    None
"""

        if not (path := self.text()):
            self.set_invalid_color()
            return

        file_path = Path(path)

        match self.path_type:
            case PathType.IMAGE:
                self.color_logic(self.is_valid_image(file_path))
            case PathType.TABLE:
                self.color_logic(self.is_valid_table(file_path))
            case PathType.VIDEO:
                self.color_logic(self.is_valid_video(file_path))
            case PathType.FOLDER:
                self.color_logic(file_path.is_dir())

        self.update_style()

    @staticmethod
    def is_valid_image(path: Path) -> bool:
        """
Checks if the given path is a valid image file.

Args:
    path (Path): The path to the file.

Returns:
    bool: True if the path is a valid image file, False otherwise.
"""

        return path.is_file() and path.suffix.lower() in Photo.file_types

    @staticmethod
    def is_valid_table(path: Path) -> bool:
        """
Checks if the given path is a valid table file.

Args:
    path (Path): The path to the file.

Returns:
    bool: True if the path is a valid table file, False otherwise.
"""

        return path.is_file() and path.suffix.lower() in Table.file_types

    @staticmethod
    def is_valid_video(path: Path) -> bool:
        """
Checks if the given path is a valid video file.

Args:
    path (Path): The path to the file.

Returns:
    bool: True if the path is a valid video file, False otherwise.
"""

        return path.is_file() and path.suffix.lower() in Video.file_types
