from PyQt6.QtCore import QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QPainter
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import QWidget


class SvgRadioButton(QWidget):
    toggled = pyqtSignal(bool)

    def __init__(self, parent=None, checked_icon=None, unchecked_icon=None):
        super().__init__(parent)

        self._checked = False
        self._hover = False
        self._text = ""  # Add text attribute

        self._checked_icon = checked_icon
        self._unchecked_icon = unchecked_icon

        # Create SVG renderer for displaying the icon
        self._svg_renderer = QSvgRenderer(self)
        self._svg_renderer.load(self._unchecked_icon)

        # Set minimum size
        self.setMinimumSize(40, 40)

        # Accept mouse events
        self.setMouseTracking(True)

        # Set focus policy
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # Add setText method for compatibility
    def setText(self, text: str):
        self._text = text
        self.update()  # Trigger repaint in case we want to display the text

    # Add text method for compatibility
    def text(self) -> str:
        return self._text

    def setChecked(self, checked: bool):
        if self._checked != checked:
            self._checked = checked
            self._update_icon()
            self.toggled.emit(self._checked)
            self.update()

    def isChecked(self) -> bool:
        return self._checked

    def _update_icon(self):
        if self._checked or self._hover:
            self._svg_renderer.load(self._checked_icon)
        else:
            self._svg_renderer.load(self._unchecked_icon)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setChecked(True)
            event.accept()
        else:
            super().mousePressEvent(event)

    def enterEvent(self, event):
        self._hover = True
        self._update_icon()
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hover = False
        self._update_icon()
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Get the SVG's default size
        default_size = self._svg_renderer.defaultSize()

        if default_size.width() > 0 and default_size.height() > 0:
            # Calculate aspect ratio
            aspect_ratio = default_size.width() / default_size.height()

            # Calculate the largest rectangle that fits within the widget
            # while maintaining the aspect ratio
            if self.width() / aspect_ratio <= self.height():
                # Width is the limiting factor
                render_width = self.width()
                render_height = render_width / aspect_ratio
            else:
                # Height is the limiting factor
                render_height = self.height()
                render_width = render_height * aspect_ratio

            # Calculate position to center the SVG
            x = (self.width() - render_width) / 2
            y = (self.height() - render_height) / 2

            # Draw the SVG
            self._svg_renderer.render(painter, QRectF(x, y, render_width, render_height))
        else:
            # Fallback if SVG has no default size
            self._svg_renderer.render(painter, QRectF(0, 0, self.width(), self.height()))

        # Draw text if present (not used in this app but added for compatibility)
        if self._text:
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._text)

    def sizeHint(self):
        return QSize(60, 60)
