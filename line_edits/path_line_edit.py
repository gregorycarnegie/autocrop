from pathlib import Path
from typing import Optional

from PyQt6 import QtCore, QtWidgets

from file_types import file_manager, FileCategory
from .custom_line_edit import CustomLineEdit
from .enums import PathType
from .file_path_validator import FilePathValidator


class PathLineEdit(CustomLineEdit):
    """
    Represents a PathLineEdit class that inherits from the CustomLineEdit class.
        
    A custom line edit widget for handling file paths.

    Args:
        path_type (PathType): The type of path to be validated. Default to PathType.FOLDER.
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
            text = text[1:][:-1]
        text = text.replace('\\', '/')
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

    def setPathType(self, path_type: PathType) -> None:
        """
        Sets the path type for the line edit and updates validation accordingly.
        
        Args:
            path_type (PathType): The type of path to validate against.
        """
        self.path_type = path_type
        # Re-validate with a new path type
        self.validate_path()
