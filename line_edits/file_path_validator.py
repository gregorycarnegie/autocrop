from typing import Optional, Tuple

from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QValidator
import re

class FilePathValidator(QValidator):
    def __init__(self, parent: Optional[QWidget]=None):
        super(FilePathValidator, self).__init__(parent)
        # Regular expression to match forbidden characters in file paths
        self.invalid_chars_pattern = re.compile(r'[*?<>|]')

    def validate(self, a0: str, a1: int) -> Tuple[QValidator.State, str, int]:
        if self.invalid_chars_pattern.search(a0):
            return QValidator.State.Invalid, a0, a1
        return QValidator.State.Acceptable, a0, a1
