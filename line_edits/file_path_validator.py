import re
from typing import ClassVar, Optional, Pattern, Tuple

from PyQt6.QtGui import QValidator


class FilePathValidator(QValidator):
    """
A custom validator for file paths that inherits from QValidator.

Attributes:
    INVALID_CHARS_PATTERN (ClassVar[Pattern[str]]): A regular expression pattern to match invalid characters in a file path.

Methods:
    validate(self, a0: Optional[str], a1: int) -> Tuple[QValidator.State, str, int]: Validates the input string based on the INVALID_CHARS_PATTERN and returns the validation state, the modified string, and an integer value.
"""

    INVALID_CHARS_PATTERN: ClassVar[Pattern[str]] = re.compile(r'[*?<>|]')

    def validate(self, a0: Optional[str], a1: int) -> Tuple[QValidator.State, str, int]:
        """
Validates the input string based on a regular expression pattern and returns the validation state, the modified string, and an integer value.

Args:
    self: The CustomLineEdit instance.
    a0 (Optional[str]): The input string to be validated.
    a1 (int): An integer value.

Returns:
    Tuple[QValidator.State, str, int]: A tuple containing the validation state, the modified string, and the integer value.
"""

        try:
            assert isinstance(a0, str)
        except AssertionError:
            a0 = ''

        if self.INVALID_CHARS_PATTERN.search(a0):
            return QValidator.State.Invalid, a0, a1
        return QValidator.State.Acceptable, a0, a1
