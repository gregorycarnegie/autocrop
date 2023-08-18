import re
from typing import Optional, Tuple

from PyQt6.QtGui import QValidator


class FilePathValidator(QValidator):
    INVALID_CHARS_PATTERN = re.compile(r'[*?<>|]')

    def validate(self, a0: Optional[str], a1: int) -> Tuple[QValidator.State, str, int]:
        try:
            assert isinstance(a0, str)
        except AssertionError:
            a0 = ''
        
        if self.INVALID_CHARS_PATTERN.search(a0):
            return QValidator.State.Invalid, a0, a1
        return QValidator.State.Acceptable, a0, a1
