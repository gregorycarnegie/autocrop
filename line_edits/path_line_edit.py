from pathlib import Path
from typing import Optional

from PyQt6 import QtCore, QtWidgets

from file_types import Photo, Table, Video
from .custom_line_edit import CustomLineEdit
from .enums import PathType
from .file_path_validator import FilePathValidator


class PathLineEdit(CustomLineEdit):
    def __init__(self, path_type: PathType=PathType.FOLDER, parent: Optional[QtWidgets.QWidget]=None):
        super().__init__(parent)
        self.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhUrlCharactersOnly)
        self.setValidator(FilePathValidator(parent=self))
        self.path_type = path_type
        self.textChanged.connect(self.insert_clipboard_path)

    def insert_clipboard_path(self, text: str) -> None:
        """Remove quotation marks from the path"""
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        elif text.startswith("'") and text.endswith("'"):
            text = text[1:-1]
        self.setText(text)

    def validate_path(self) -> None:
        """Validate QLineEdit based on input and set color accordingly."""
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
        return path.is_file() and path.suffix.lower() in Photo().file_types
    
    @staticmethod
    def is_valid_table(path: Path) -> bool:
        return path.is_file() and path.suffix.lower() in Table().file_types
    
    @staticmethod
    def is_valid_video(path: Path) -> bool:
        return path.is_file() and path.suffix.lower() in Video().file_types
