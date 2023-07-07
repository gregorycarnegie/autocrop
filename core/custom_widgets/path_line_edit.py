from ..file_types import IMAGE_TYPES, PANDAS_TYPES, VIDEO_TYPES
from typing import Optional
from PyQt6.QtCore import QFileInfo
from PyQt6.QtWidgets import QLineEdit, QWidget
    

class PathLineEdit(QLineEdit):
    def __init__(self, path_type: str, parent: Optional[QWidget]=None):
        super(PathLineEdit, self).__init__(parent)
        self.path_type = path_type
        self.textChanged.connect(self.validate_path)
        self.validColour = '#7fda91' # light green
        self.invalidColour = '#ff6c6c' # rose
        self.setStyleSheet(f'background-color: {self.invalidColour}; color: black;')

    def validate_path(self):
        path = self.text()

        if not path:
            self.setStyleSheet(f'background-color: {self.invalidColour}; color: black;')
            return

        if self.path_type == 'image':
            if QFileInfo(path).isFile() and any(path.lower().endswith(ext) for ext in IMAGE_TYPES):
                color = self.validColour
            else:
                color = self.invalidColour
        elif self.path_type == 'table':
            if QFileInfo(path).isFile() and any(path.lower().endswith(ext) for ext in PANDAS_TYPES):
                color = self.validColour
            else:
                color = self.invalidColour
        elif self.path_type == 'video':
            if QFileInfo(path).isFile() and any(path.lower().endswith(ext) for ext in VIDEO_TYPES):
                color = self.validColour
            else:
                color = self.invalidColour
        elif self.path_type == 'folder':
            color = self.validColour if QFileInfo(path).isDir() else self.invalidColour
        else:
            self.setStyleSheet(f'background-color: {self.invalidColour}; color: black;')
            return
        
        self.setStyleSheet(f'background-color: {color}; color: black;')
