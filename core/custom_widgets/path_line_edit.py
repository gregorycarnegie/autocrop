from typing import Optional

from PyQt6.QtCore import QFileInfo
from PyQt6.QtWidgets import QWidget

from .custom_line_edit import CustomLineEdit
from ..file_types import IMAGE_TYPES, PANDAS_TYPES, VIDEO_TYPES


class PathLineEdit(CustomLineEdit):
    def __init__(self, path_type: str, parent: Optional[QWidget]=None):
        super().__init__(parent)
        self.path_type = path_type

    def validate_path(self):
        path = self.text()

        if not path:
            self.setStyleSheet(f'background-color: {self.invalidColour}; color: black;')
            return

        if self.path_type == 'image':
            if QFileInfo(path).isFile() and any(path.lower().endswith(ext) for ext in IMAGE_TYPES):
                self.colour = self.validColour
            else:
                self.colour = self.invalidColour
        elif self.path_type == 'table':
            if QFileInfo(path).isFile() and any(path.lower().endswith(ext) for ext in PANDAS_TYPES):
                self.colour = self.validColour
            else:
                self.colour = self.invalidColour
        elif self.path_type == 'video':
            if QFileInfo(path).isFile() and any(path.lower().endswith(ext) for ext in VIDEO_TYPES):
                self.colour = self.validColour
            else:
                self.colour = self.invalidColour
        elif self.path_type == 'folder':
            self.colour = self.validColour if QFileInfo(path).isDir() else self.invalidColour
        else:
            self.setStyleSheet(f'background-color: {self.invalidColour}; color: black;')
            return

        self.setStyleSheet(f'background-color: {self.colour}; color: black;')
