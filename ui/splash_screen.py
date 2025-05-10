from typing import ClassVar

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QGuiApplication, QMouseEvent, QPixmap
from PyQt6.QtWidgets import QSplashScreen

from .enums import GuiIcon


class UiClickableSplashScreen(QSplashScreen):
    clicked = pyqtSignal()
    image_path: ClassVar[str] = GuiIcon.LOGO

    def __init__(self):
        pixmap = self.get_scaled_pixmap(self.image_path)
        super().__init__(pixmap, Qt.WindowType.WindowStaysOnTopHint)

    def mousePressEvent(self, a0: QMouseEvent | None) -> None:
        self.clicked.emit()

    def show_message(self, message: str = 'Loading...', color: QColor = QColor(255, 255, 255)) -> None:
        self.show()
        self.showMessage(message, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter, color)

    @staticmethod
    def get_scaled_pixmap(image_path: str) -> QPixmap:
        pixmap = QPixmap(image_path)
        screen = QGuiApplication.primaryScreen()
        screen_height = 255 if screen is None else screen.size().height()
        pixmap = pixmap.scaledToHeight(screen_height // 3, Qt.TransformationMode.SmoothTransformation)
        return pixmap
