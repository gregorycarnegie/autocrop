from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QPixmap, QPaintEvent
from PyQt6.QtWidgets import QWidget
from typing import Optional

class ImageWidget(QWidget):
    def __init__(self, parent: Optional[QWidget]=None):
        super().__init__(parent)
        self.image = None

    def setImage(self, image: QPixmap) -> None:
        self.image = image
        self.update()

    def paintEvent(self, a0: QPaintEvent) -> None:
        if self.image is not None:
            qp = QPainter(self)
            qp.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            scaled_image = self.image.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                             Qt.TransformationMode.SmoothTransformation)
            x_offset = (self.width() - scaled_image.width()) >> 1
            y_offset = (self.height() - scaled_image.height()) >> 1
            qp.drawPixmap(x_offset, y_offset, scaled_image)
