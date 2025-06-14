from typing import ClassVar

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon, QPixmap, QResizeEvent
from PyQt6.QtWidgets import QLineEdit, QStyle, QToolButton, QWidget

from ui.enums import GuiIcon

from .enums import LineEditState


class CustomLineEdit(QLineEdit):
    """
    Represents a CustomLineEdit class that inherits from the QLineEdit class.

    The CustomLineEdit class provides a customized line edit widget with a clear button and validation
    functionality. It overrides the `resizeEvent` method to handle the positioning of the clear button.
    The class also defines methods for updating the clear button visibility, validating the input path,
    setting the valid and invalid colors, and updating the style.

    Attributes:
        VALID_COLOR (ClassVar[str]): The colour code for valid input.
        INVALID_COLOR (ClassVar[str]): The colour code for invalid input.

    Methods:
        resizeEvent(a0: Optional[QResizeEvent]): Overrides the resize event to handle the positioning
            of the clear button.
        update_clear_button(text: str) -> None: Updates the visibility of the clear button based on
            the text input.
        validate_path() -> None: Validates the input path and sets the colour accordingly.
        set_valid_color() -> None: Sets the colour to the valid colour.
        set_invalid_color() -> None: Sets the colour to the invalid colour.
        color_logic(boolean: bool) -> None: Sets the colour and state based on a boolean value.
        update_style() -> None: Updates the style of the line edit.

    Example:

        # Creating an instance of the CustomLineEdit class
        line_edit = CustomLineEdit()

        # Resizing the line edit
        line_edit.resizeEvent(None)

        # Updating the clear button visibility
        line_edit.update_clear_button("Text")

        # Validating the input path
        line_edit.validate_path()

        # Setting the valid colour
        line_edit.set_valid_color()

        # Setting the invalid colour
        line_edit.set_invalid_color()

        # Updating the style
        line_edit.update_style()

    """
    VALID_COLOR: ClassVar[str] = '#7fda91'  # light green
    INVALID_COLOR: ClassVar[str] = '#ff6c6c'  # rose

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.state = LineEditState.INVALID_INPUT
        self.colour = self.INVALID_COLOR
        self.clearButton = QToolButton(self)
        self.clearButton.setIcon(QIcon(QPixmap(GuiIcon.CLEAR.value)))
        self.clearButton.setCursor(Qt.CursorShape.ArrowCursor)
        self.clearButton.setStyleSheet('QToolButton { border: none; padding: 0px; }')
        self.clearButton.hide()
        self.clearButton.clicked.connect(self.clear)

        self.textChanged.connect(self.update_clear_button)
        self.textChanged.connect(self.validate_path)

        self.setFixedHeight(36)

        self.update_style()

    def resizeEvent(self, a0: QResizeEvent | None):
        """
        Overrides the resize event to handle the positioning of the clear button.

        Args:
            self: The CustomLineEdit instance.
            a0 (Optional[QResizeEvent]): The resize event.

        Returns:
            None
        """

        button_size = self.clearButton.sizeHint()
        if (style := self.style()) is None:
            return

        frame_width = style.pixelMetric(QStyle.PixelMetric.PM_DefaultFrameWidth)
        rect = self.rect()
        self.clearButton.move(rect.right() - frame_width - button_size.width(),
                              (rect.bottom() - button_size.height() + 1) >> 1)

        super().resizeEvent(a0)

    def update_clear_button(self, text: str) -> None:
        """
        Updates the visibility of the clear button based on the text input.
        """
        self.clearButton.setVisible(bool(text))

    def showEvent(self, event) -> None:
        """
        Override showEvent to ensure clear button visibility is properly updated when widget becomes visible.

        Args:
            event: The show event.

        Returns:
            None
        """
        super().showEvent(event)
        # Update clear button visibility based on the current text
        self.update_clear_button(self.text())

    def validate_path(self) -> None:
        """Validate a path based on input and set colour accordingly."""
        raise NotImplementedError('Subclasses should implement this method!')

    def set_valid_color(self) -> None:
        """
        Sets the colour of the CustomLineEdit to the valid colour.

        Args:
            self: The CustomLineEdit instance.

        Returns:
            None
        """

        self.colour = self.VALID_COLOR

    def set_invalid_color(self) -> None:
        """
        Sets the colour of the CustomLineEdit to the invalid colour.

        Args:
            self: The CustomLineEdit instance.

        Returns:
            None
        """

        self.colour = self.INVALID_COLOR

    def color_logic(self, boolean: bool) -> None:
        """
        Sets the colour and state of the CustomLineEdit based on a boolean value.

        Args:
            self: The CustomLineEdit instance.
            boolean (bool): A boolean value indicating the condition.

        Returns:
            None
        """

        if boolean:
            self.set_valid_color()
            self.state = LineEditState.VALID_INPUT
        else:
            self.set_invalid_color()
            self.state = LineEditState.INVALID_INPUT

    def update_style(self) -> None:
        """
        Updates the style of the CustomLineEdit.
        """
        self.setStyleSheet(f'background-color: {self.colour}; color: black;')
        # Update clear button visibility without triggering validation
        self.clearButton.setVisible(bool(self.text()))

    def clear(self) -> None:
        """Override clear to ensure proper validation"""
        super().clear()
        # Trigger validation after clearing, but not through update_clear_button
        if hasattr(self, 'validate_path'):
            self.validate_path()
