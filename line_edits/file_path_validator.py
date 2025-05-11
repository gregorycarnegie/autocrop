import re
from typing import ClassVar

from PyQt6.QtGui import QValidator


class FilePathValidator(QValidator):
    """
    A custom validator for file paths that inherits from QValidator.

    Attributes:
        INVALID_CHARS_PATTERN (ClassVar[Pattern[str]]): A regex pattern to match invalid characters in a file path.

    Methods:
        validate(self, a0: Optional[str], a1: int) -> Tuple[QValidator.State, str, int]:
    """

    INVALID_CHARS_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r'[*?<>|]')

    def validate(self, a0: str | None, a1: int) -> tuple[QValidator.State, str, int]:
        """
    Validates the input string based on a regex pattern and returns the validation state, the modified string.

    Args:
        self: The CustomLineEdit instance.
        a0 (Optional[str]): The input string to be validated.
        a1 (int): An integer value.

    Returns:
        Tuple[QValidator.State, str, int]: A tuple of the validation state, the modified string, and the integer value.
    """

        try:
            assert isinstance(a0, str)
        except AssertionError:
            a0 = ''

        if self.INVALID_CHARS_PATTERN.search(a0):
            return QValidator.State.Invalid, a0, a1
        return QValidator.State.Acceptable, a0, a1
