from PyQt6.QtCore import QSize, Qt, QTimer
from PyQt6.QtGui import QBrush, QColor, QPainter
from PyQt6.QtWidgets import QWidget


class PulsingProgressIndicator(QWidget):
    """
    A pulsing rectangle indicator for showing processing status.
    - Pulsing blue during processing
    - Solid green when complete
    - Transparent when idle
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        # State management
        self.is_processing = False
        self.is_complete = False
        self.opacity = 1.0
        self.increasing = False

        # Animation timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.setInterval(50)  # Update every 50ms for smooth animation

        # Set minimum size
        self.setMinimumSize(20, 20)
        self.setMaximumSize(16_777_215, 40)  # Same height as original progress bars

    def start_processing(self):
        """Start the pulsing animation"""
        self.is_processing = True
        self.is_complete = False
        self.opacity = 1.0
        self.increasing = False
        self.timer.start()

    def finish_processing(self):
        """Stop pulsing and show complete state"""
        self._update_from_environment(True, 1.0)

    def reset(self):
        """Reset to idle state"""
        self._update_from_environment(False, 0.0)

    def _update_from_environment(self, arg0, arg1):
        self.timer.stop()
        self.is_processing = False
        self.is_complete = arg0
        self.opacity = arg1
        self.update()

    def update_animation(self):
        """Update the pulsing animation"""
        if not self.is_processing:
            return

        # Pulse between 0.3 and 1.0 opacity
        if self.increasing:
            self.opacity += 0.05
            if self.opacity >= 1.0:
                self.opacity = 1.0
                self.increasing = False
        else:
            self.opacity -= 0.05
            if self.opacity <= 0.3:
                self.opacity = 0.3
                self.increasing = True

        self.update()

    def paintEvent(self, event):
        """Paint the indicator rectangle"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self.is_processing:
            # Pulsing blue
            color = QColor(66, 133, 244)  # Google blue
            color.setAlphaF(self.opacity)
            self.__recolor(painter, color)
        elif self.is_complete:
            # Solid green
            color = QColor(52, 168, 83)  # Google green
            self.__recolor(painter, color)

    def __recolor(self, painter: QPainter, color: QColor):
        painter.setBrush(QBrush(color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(self.rect())

    def sizeHint(self):
        """Provide size hint for layout"""
        return QSize(200, 20)
