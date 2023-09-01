from typing import ClassVar, Optional

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QColor, QGuiApplication, QMouseEvent, QPixmap
from PyQt6.QtWidgets import QSplashScreen


class ClickableSplashScreen(QSplashScreen):
    clicked = pyqtSignal()
    image_path: ClassVar[str] = 'resources\\logos\\logo.svg'

    def __init__(self):
        pixmap = self.get_scaled_pixmap(self.image_path)
        super(ClickableSplashScreen, self).__init__(pixmap, Qt.WindowType.WindowStaysOnTopHint)

    def mousePressEvent(self, a0: Optional[QMouseEvent]) -> None:
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
