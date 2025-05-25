from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPaintEvent, QPen, QPixmap
from PyQt6.QtWidgets import QWidget


class ImageWidget(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.image: QPixmap | None = None
        self.show_no_face_message = False
        self.error_message = ""

    def setImage(self, image: QImage) -> None:
        """Set the image to display"""
        self.image = QPixmap.fromImage(image)
        self.show_no_face_message = False
        self.error_message = ""
        self.update()

    def showNoFaceDetected(self, message: str = "No face detected") -> None:
        """Show a message when no face is detected"""
        self.image = None
        self.show_no_face_message = True
        self.error_message = message
        self.update()

    def showError(self, message: str) -> None:
        """Show an error message"""
        self.image = None
        self.show_no_face_message = True
        self.error_message = message
        self.update()

    def clear(self) -> None:
        """Clear the widget"""
        self.image = None
        self.show_no_face_message = False
        self.error_message = ""
        self.update()

    def paintEvent(self, a0: QPaintEvent | None) -> None:
        qp = QPainter(self)
        qp.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        if self.image is not None:
            # Draw the image
            scaled_image = self.image.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            x_offset = (self.width() - scaled_image.width()) >> 1
            y_offset = (self.height() - scaled_image.height()) >> 1
            qp.drawPixmap(x_offset, y_offset, scaled_image)
        elif self.show_no_face_message:
            # Draw the error/no face message
            self._draw_message(qp, self.error_message)
        else:
            # Draw placeholder
            self._draw_placeholder(qp)

    def _draw_message(self, painter: QPainter, message: str) -> None:
        """Draw a centered message with visual styling"""
        # Set up the font
        font = QFont("Arial", 12, QFont.Weight.Bold)
        painter.setFont(font)

        # Set colors based on message type
        if "no face" in message.lower():
            bg_color = QColor(255, 193, 7, 180)  # Warning yellow with transparency
            text_color = QColor(0, 0, 0)
            border_color = QColor(255, 193, 7)
        else:
            bg_color = QColor(220, 53, 69, 180)  # Error red with transparency
            text_color = QColor(255, 255, 255)
            border_color = QColor(220, 53, 69)

        # Calculate text rectangle
        text_rect = painter.fontMetrics().boundingRect(
            self.rect(), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, message
        )
        text_rect.adjust(-20, -15, 20, 15)  # Add padding

        # Center the rectangle
        x = (self.width() - text_rect.width()) // 2
        y = (self.height() - text_rect.height()) // 2
        text_rect.moveTo(x, y)

        # Draw background
        painter.fillRect(text_rect, bg_color)

        # Draw border
        painter.setPen(QPen(border_color, 2))
        painter.drawRoundedRect(text_rect, 8, 8)

        # Draw text
        painter.setPen(text_color)
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, message)

    def _draw_placeholder(self, painter: QPainter) -> None:
        """Draw a placeholder when no image is loaded"""
        # Dark theme background to match the app
        bg_color = QColor(31, 44, 51)  # #1f2c33 - matches your app's background
        painter.fillRect(self.rect(), bg_color)

        # Draw subtle border
        border_color = QColor(68, 85, 95)  # Slightly lighter than background
        painter.setPen(QPen(border_color, 1))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))

        # Draw placeholder text in light gray
        font = QFont("Arial", 10)
        painter.setFont(font)
        painter.setPen(QColor(160, 170, 180))  # Light gray text for dark theme
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No image")

