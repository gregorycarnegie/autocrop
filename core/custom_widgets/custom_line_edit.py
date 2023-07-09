from typing import Optional
from .line_edit_state import LineEditState
from PyQt6.QtWidgets import QLineEdit, QWidget


class CustomLineEdit(QLineEdit):
    def __init__(self, parent: Optional[QWidget] = None):
        super(CustomLineEdit, self).__init__(parent)
        self.state = LineEditState.INVALID_INPUT
        self.textChanged.connect(self.validate_path)
        self.textChanged.connect(self.change_state)
        self.validColour = '#7fda91'  # light green
        self.invalidColour = '#ff6c6c'  # rose
        self.colour = self.invalidColour
        self.setStyleSheet(f'background-color: {self.invalidColour}; color: black;')

    def change_state(self):
        if self.colour == self.validColour:
            self.state = LineEditState.VALID_INPUT
        elif self.colour == self.invalidColour:
            self.state = LineEditState.INVALID_INPUT

    def validate_path(self):
        """Subclasses should implement this!"""
        return None
