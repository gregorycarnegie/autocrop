import re
from typing import Optional, Tuple

from PyQt6.QtGui import QValidator
from PyQt6.QtWidgets import QWidget


class FilePathValidator(QValidator):
    def __init__(self, parent: Optional[QWidget]=None):
        super(FilePathValidator, self).__init__(parent)
        # Regular expression to match forbidden characters in file paths
        self.invalid_chars_pattern = re.compile(r'[*?<>|]')

    def validate(self, a0: Optional[str], a1: int) -> Tuple[QValidator.State, str, int]:
        try:
            assert isinstance(a0, str)
        except AssertionError:
            a0 = ''
        
        if self.invalid_chars_pattern.search(a0):
            return QValidator.State.Invalid, a0, a1
        return QValidator.State.Acceptable, a0, a1
