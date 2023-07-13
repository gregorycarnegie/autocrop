from PyQt6.QtWidgets import QPushButton
from PyQt6.QtCore import QPropertyAnimation, QRect, QEasingCurve


class AnimatedButton(QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.animation = QPropertyAnimation(self, b'geometry')
        self.animation.setDuration(10)  # duration in milliseconds
        self.animation.setEasingCurve(QEasingCurve.Type.InOutSine)

    def enterEvent(self, event):
        start_rect = self.geometry()
        end_rect = QRect(start_rect.x() + 1, start_rect.y() - 1, start_rect.width(), start_rect.height())
        self.animation.setStartValue(start_rect)
        self.animation.setEndValue(end_rect)
        self.animation.setDirection(QPropertyAnimation.Direction.Forward)
        self.animation.start()

    def leaveEvent(self, event):
        self.animation.setEndValue(self.geometry())
        self.animation.setDirection(QPropertyAnimation.Direction.Backward)
        self.animation.start()
