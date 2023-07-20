from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import QWidget

from .custom_line_edit import CustomLineEdit
from .enums import PathType
from ..main_objects.f_type_photo import IMAGE_TYPES
from ..main_objects.f_type_table import PANDAS_TYPES
from ..main_objects.f_type_video import VIDEO_TYPES


class PathLineEdit(CustomLineEdit):
    def __init__(self, path_type: PathType=PathType.FOLDER, parent: Optional[QWidget]=None):
        super().__init__(parent)
        self.path_type = path_type

    def validate_path(self) -> None:
        """Validate QLineEdit based on input and set color accordingly."""
        if not (path := self.text()):
            self.set_invalid_color()
            return

        file_path = Path(path)
        
        if self.path_type == PathType.IMAGE: self.color_logic(self.is_valid_image(file_path))
        elif self.path_type == PathType.TABLE: self.color_logic(self.is_valid_table(file_path))
        elif self.path_type == PathType.VIDEO: self.color_logic(self.is_valid_video(file_path))
        elif self.path_type == PathType.FOLDER: self.color_logic(file_path.is_dir())

        self.update_style()

    @staticmethod
    def is_valid_image(path: Path) -> bool:
        return path.is_file() and path.suffix.lower() in IMAGE_TYPES
    
    @staticmethod
    def is_valid_table(path: Path) -> bool:
        return path.is_file() and path.suffix.lower() in PANDAS_TYPES
    
    @staticmethod
    def is_valid_video(path: Path) -> bool:
        return path.is_file() and path.suffix.lower() in VIDEO_TYPES
