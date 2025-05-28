from PyQt6.QtCore import QCoreApplication, Qt
from PyQt6.QtGui import QIntValidator
from PyQt6.QtWidgets import QWidget

from .custom_line_edit import CustomLineEdit


class NumberLineEdit(CustomLineEdit):
    """
    Represents a NumberLineEdit class that inherits from the CustomLineEdit class.

    A custom line edit widget for handling numeric input.

    Args:
        name (str): The object name of the widget.
        parent (Optional[QWidget]): The parent widget. Defaults to None.

    Methods:
        validate_path(self): Validates the path by performing color logic and updating the style of the NumberLineEdit.
        value(self) -> int: Returns the integer value of the text input.
    """

    def __init__(self, name: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setInputMethodHints(Qt.InputMethodHint.ImhDigitsOnly)
        self.setValidator(QIntValidator(parent=self))
        self.setText('')
        self.setObjectName(name)
        self.setPlaceholderText(QCoreApplication.translate("self", "Type a number", None))

    def validate_path(self):
        """
        Validates the path by performing color logic and updating the style of the NumberLineEdit.

        Args:
            self: The NumberLineEdit instance.

        Returns:
            None
        """

        self.color_logic(self.text().isdigit())
        self.update_style()

    def value(self) -> int:
        """
        Returns the integer value of the text input.

        Args:
            self: The NumberLineEdit instance.

        Returns:
            int: The integer value of the text input. If the text cannot be converted to an integer, returns 0.
        """

        try:
            return int(self.text())
        except ValueError:
            return 0
