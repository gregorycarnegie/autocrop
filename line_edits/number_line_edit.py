from typing import Optional

from PyQt6 import QtCore, QtGui, QtWidgets

from .custom_line_edit import CustomLineEdit


class NumberLineEdit(CustomLineEdit):
    def __init__(self, name: str, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhDigitsOnly)
        self.setValidator(QtGui.QIntValidator(parent=self))
        self.setText('')
        self.setObjectName(name)

    def validate_path(self):
        """Validate QLineEdit based on input and set color accordingly."""
        self.color_logic(self.text().isdigit())
        self.update_style()

    def value(self) -> int:
        try:
            return int(self.text())
        except ValueError:
            return 0
