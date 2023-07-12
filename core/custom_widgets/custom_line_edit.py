from typing import Optional, ClassVar

from PyQt6.QtWidgets import QLineEdit, QWidget

from .line_edit_state import LineEditState


class CustomLineEdit(QLineEdit):
    VALID_COLOR: ClassVar[str] = '#7fda91'  # light green
    INVALID_COLOR: ClassVar[str] = '#ff6c6c'  # rose
    def __init__(self, parent: Optional[QWidget] = None):
        super(CustomLineEdit, self).__init__(parent)
        self.state = LineEditState.INVALID_INPUT
        self.textChanged.connect(self.validate_path)
        self.colour = self.INVALID_COLOR
        self.update_style()

    def validate_path(self):
        """Validate path based on input and set color accordingly.
        Subclasses should implement this method!"""
        return None

    def set_valid_color(self) -> None:
        self.colour = self.VALID_COLOR

    def set_invalid_color(self) -> None:
        self.colour = self.INVALID_COLOR
    
    def color_logic(self, boolean: bool) -> None:
        if boolean:
            self.set_valid_color()
            self.state = LineEditState.VALID_INPUT
        else:
            self.set_invalid_color()
            self.state = LineEditState.INVALID_INPUT

    def update_style(self) -> None:
        self.setStyleSheet(f'background-color: {self.colour}; color: black;')
