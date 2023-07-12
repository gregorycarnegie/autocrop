from typing import Optional
from pathlib import Path

from PyQt6.QtWidgets import QWidget

from .custom_line_edit import CustomLineEdit
from .path_type import PathType
from ..file_types import IMAGE_TYPES, PANDAS_TYPES, VIDEO_TYPES


class PathLineEdit(CustomLineEdit):
    def __init__(self, path_type: PathType, parent: Optional[QWidget]=None):
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

    def is_valid_image(self, path: Path) -> bool:
        return path.is_file() and path.suffix.lower() in IMAGE_TYPES
    
    def is_valid_table(self, path: Path) -> bool:
        return path.is_file() and path.suffix.lower() in PANDAS_TYPES
    
    def is_valid_video(self, path: Path) -> bool:
        return path.is_file() and path.suffix.lower() in VIDEO_TYPES
