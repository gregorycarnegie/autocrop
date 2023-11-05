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


    def insert_clipboard_path(self):
        """
        Removes the quotation marks from the passed path
        """
        text = self.text()
        x, y = text.startswith("'") & text.endswith("'"), text.startswith('"') & text.endswith('"')
        if x ^ y:
            self.setText(text[1:][:-1])


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
        self.color_logic(self.is_valid_path(file_path))
        self.update_style()

    
    def is_valid_path(self, path: Path) -> bool:
        """
        Checks if the given path is valid.

        Args:
            path (Path): The path to the file/folder.

        Returns:
            bool: True if the path is valid, False otherwise.
        """
        is_file, suffix = path.is_file(), path.suffix.lower()

        match self.path_type:
            case PathType.IMAGE:
                valid_suffix = suffix in Photo.file_types
                return is_file & valid_suffix
            case PathType.TABLE:
                valid_suffix = suffix in Table.file_types
                return is_file & valid_suffix
            case PathType.VIDEO:
                valid_suffix = suffix in Video.file_types
                return is_file & valid_suffix
            case PathType.FOLDER:
                return path.is_dir()
