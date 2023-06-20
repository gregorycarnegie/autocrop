from PyQt6 import QtCore, QtGui, QtWidgets

class ImageWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = None

    def setImage(self, image: QtGui.QPixmap) -> None:
        self.image = image
        self.update()

    def paintEvent(self, event) -> None:
        if self.image is not None:
            qp = QtGui.QPainter(self)
            qp.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
            scaled_image = self.image.scaled(self.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                                             QtCore.Qt.TransformationMode.SmoothTransformation)
            x_offset = (self.width() - scaled_image.width()) >> 1
            y_offset = (self.height() - scaled_image.height()) >> 1
            qp.drawPixmap(x_offset, y_offset, scaled_image)
            