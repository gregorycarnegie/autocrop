from typing import ClassVar, Optional

from PyQt6.QtWidgets import QLineEdit, QStyle, QToolButton, QWidget
from PyQt6.QtGui import QIcon, QPixmap, QResizeEvent
from PyQt6.QtCore import Qt

from .enums import LineEditState


class CustomLineEdit(QLineEdit):
    VALID_COLOR: ClassVar[str] = '#7fda91'  # light green
    INVALID_COLOR: ClassVar[str] = '#ff6c6c'  # rose
    
    def __init__(self, parent: Optional[QWidget] = None):
        super(CustomLineEdit, self).__init__(parent)
        self.clearButton = QToolButton(self)
        self.clearButton.setIcon(QIcon(QPixmap('resources\\icons\\clear.svg')))
        self.clearButton.setCursor(Qt.CursorShape.ArrowCursor)
        self.clearButton.setStyleSheet('QToolButton { border: none; padding: 0px; }')
        self.clearButton.hide()
        self.clearButton.clicked.connect(self.clear)
        self.textChanged.connect(self.updateClearButton)

        self.state = LineEditState.INVALID_INPUT
        self.textChanged.connect(self.validate_path)
        self.colour = self.INVALID_COLOR
        self.update_style()

    def resizeEvent(self, a0: Optional[QResizeEvent]):
        buttonSize = self.clearButton.sizeHint()
        frameWidth = self.style().pixelMetric(QStyle.PixelMetric.PM_DefaultFrameWidth)
        self.clearButton.move(self.rect().right() - frameWidth - buttonSize.width(),
                              (self.rect().bottom() - buttonSize.height() + 1) >> 1)
        super(CustomLineEdit, self).resizeEvent(a0)

    def updateClearButton(self, text: str) -> None:
        self.clearButton.setVisible(bool(text))

    def validate_path(self) -> None:
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
