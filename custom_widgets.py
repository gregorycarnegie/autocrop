import pandas as pd
from files import IMAGE_TYPES, PANDAS_TYPES, VIDEO_TYPES
from PyQt6 import QtCore, QtGui, QtWidgets
from typing import Optional

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


class DataFrameModel(QtCore.QAbstractTableModel):
    def __init__(self, df: pd.DataFrame, parent=None):
        super(DataFrameModel, self).__init__(parent)
        self._df = df

    def rowCount(self, parent=None):
        if self._df is not None:
            return self._df.shape[0]

    def columnCount(self, parent=None):
        if self._df is not None:
            return self._df.shape[1]

    def data(self, index, role=QtCore.Qt.ItemDataRole.DisplayRole) -> Optional[str]:
        if index.isValid() and role == QtCore.Qt.ItemDataRole.DisplayRole:
            return str(self._df.iloc[index.row(), index.column()])
        return None

    def headerData(self, section, orientation, role=QtCore.Qt.ItemDataRole.DisplayRole) -> Optional[str]:
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            try:
                if orientation == QtCore.Qt.Orientation.Horizontal:
                    return str(self._df.columns[section])
                if orientation == QtCore.Qt.Orientation.Vertical:
                    return str(self._df.index[section])
            except IndexError:
                return None
        return None
    

class CustomQDial(QtWidgets.QDial):
    def __init__(self, *args, **kwargs):
        super(CustomQDial, self).__init__(*args, **kwargs)

    def event(self, event):
        if event.type() == QtCore.QEvent.Type.ToolTip:
            QtWidgets.QToolTip.showText(event.globalPos(), str(self.value()), self)
            return True
        return super(CustomQDial, self).event(event)


class PathLineEdit(QtWidgets.QLineEdit):
    def __init__(self, path_type: str, parent=None):
        super(PathLineEdit, self).__init__(parent)
        self.path_type = path_type
        self.textChanged.connect(self.validate_path)
        self.validColour = "#7fda91" # light green
        self.invalidColour = "#ff6c6c" # rose
        self.setStyleSheet(f"background-color: {self.invalidColour}; color: black;")

    def validate_path(self):
        path = self.text()

        if not path:
            self.setStyleSheet(f"background-color: {self.invalidColour}; color: black;")
            return

        if self.path_type == 'image':
            if QtCore.QFileInfo(path).isFile() and any(path.endswith(ext) for ext in IMAGE_TYPES):
                color = self.validColour
            else:
                color = self.invalidColour
        elif self.path_type == 'table':
            if QtCore.QFileInfo(path).isFile() and any(path.endswith(ext) for ext in PANDAS_TYPES):
                color = self.validColour
            else:
                color = self.invalidColour
        elif self.path_type == 'video':
            if QtCore.QFileInfo(path).isFile() and any(path.endswith(ext) for ext in VIDEO_TYPES):
                color = self.validColour
            else:
                color = self.invalidColour
        elif self.path_type == 'folder':
            color = self.validColour if QtCore.QFileInfo(path).isDir() else self.invalidColour
        else:
            self.setStyleSheet(f"background-color: {self.invalidColour}; color: black;")
            return
        
        self.setStyleSheet(f"background-color: {color}; color: black;")


class NumberLineEdit(QtWidgets.QLineEdit):
    def __init__(self, parent=None):
        super(NumberLineEdit, self).__init__(parent)
        self.textChanged.connect(self.validate_path)
        self.validColour = "#7fda91" # light green
        self.invalidColour = "#ff6c6c" # rose
        self.setStyleSheet(f"background-color: {self.invalidColour}; color: black;")

    def validate_path(self):
        if self.text().isdigit():
            self.setStyleSheet(f"background-color: {self.validColour}; color: black;")
        else:
            self.setStyleSheet(f"background-color: {self.invalidColour}; color: black;")
        return


# class ClickableSlider(QtWidgets.QSlider):
#     def __init__(self, orientation, parent=None):
#         super().__init__(orientation, parent)

#     def mousePressEvent(self, event):
#         if event.button() == QtCore.Qt.MouseButton.LeftButton:
#             value = self.pixelPosToRangeValue(event.pos())
#             self.setValue(value)
#             event.accept()
#         else:
#             super().mousePressEvent(event)

#     def pixelPosToRangeValue(self, pos):
#         opt = self.style().sliderPositionFromValue(
#             self.minimum(), self.maximum(), 0, self.width(), self.orientation()
#         )
#         return self.style().sliderValueFromPosition(
#             self.minimum(), self.maximum(), opt - pos.x(), self.width(), self.orientation()
#         )


class UiDialog(QtWidgets.QDialog):
    def __init__(self):
        super(UiDialog, self).__init__()
        self.setObjectName("Dialog")
        self.resize(347, 442)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(sizePolicy)
        self.setMinimumSize(QtCore.QSize(347, 442))
        self.setMaximumSize(QtCore.QSize(347, 442))
        self.verticalLayout = QtWidgets.QVBoxLayout(self)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(parent=self)
        self.label.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setMinimumSize(QtCore.QSize(329, 329))
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap("resources/logos/logo.png"))
        self.setWindowIcon(QtGui.QIcon("resources/logos/logo.ico"))
        self.label.setScaledContents(True)
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.frame = QtWidgets.QFrame(parent=self)
        self.frame.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.frame.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.frame.setLineWidth(0)
        self.frame.setObjectName("frame")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.frame)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_7 = QtWidgets.QLabel(parent=self.frame)
        self.label_7.setMinimumSize(QtCore.QSize(50, 0))
        self.label_7.setMaximumSize(QtCore.QSize(50, 16_777_215))
        self.label_7.setTextFormat(QtCore.Qt.TextFormat.AutoText)
        self.label_7.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignLeading | QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        self.label_7.setObjectName("label_7")
        self.horizontalLayout.addWidget(self.label_7)
        self.label_8 = QtWidgets.QLabel(parent=self.frame)
        self.label_8.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignLeading | QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        self.label_8.setObjectName("label_8")
        self.horizontalLayout.addWidget(self.label_8)
        self.verticalLayout.addWidget(self.frame)
        self.frame_2 = QtWidgets.QFrame(parent=self)
        self.frame_2.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.frame_2.setLineWidth(0)
        self.frame_2.setObjectName("frame_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.frame_2)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_4 = QtWidgets.QLabel(parent=self.frame_2)
        self.label_4.setMinimumSize(QtCore.QSize(50, 0))
        self.label_4.setMaximumSize(QtCore.QSize(50, 16_777_215))
        self.label_4.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignLeading | QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_2.addWidget(self.label_4)
        self.label_3 = QtWidgets.QLabel(parent=self.frame_2)
        self.label_3.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignLeading | QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_2.addWidget(self.label_3)
        self.verticalLayout.addWidget(self.frame_2)
        self.frame_3 = QtWidgets.QFrame(parent=self)
        self.frame_3.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.frame_3.setLineWidth(0)
        self.frame_3.setObjectName("frame_3")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.frame_3)
        self.horizontalLayout_3.setContentsMargins(0, 0, -1, 0)
        self.horizontalLayout_3.setSpacing(0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.frame_5 = QtWidgets.QFrame(parent=self.frame_3)
        self.frame_5.setMinimumSize(QtCore.QSize(50, 0))
        self.frame_5.setMaximumSize(QtCore.QSize(50, 16_777_215))
        self.frame_5.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.frame_5.setLineWidth(0)
        self.frame_5.setObjectName("frame_5")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.frame_5)
        self.verticalLayout_3.setContentsMargins(-1, -1, 0, -1)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label_5 = QtWidgets.QLabel(parent=self.frame_5)
        self.label_5.setMinimumSize(QtCore.QSize(50, 0))
        self.label_5.setMaximumSize(QtCore.QSize(50, 16_777_215))
        self.label_5.setLineWidth(0)
        self.label_5.setObjectName("label_5")
        self.verticalLayout_3.addWidget(self.label_5)
        self.label_9 = QtWidgets.QLabel(parent=self.frame_5)
        self.label_9.setMinimumSize(QtCore.QSize(50, 0))
        self.label_9.setMaximumSize(QtCore.QSize(50, 16_777_215))
        self.label_9.setLineWidth(0)
        self.label_9.setText("")
        self.label_9.setObjectName("label_9")
        self.verticalLayout_3.addWidget(self.label_9)
        self.horizontalLayout_3.addWidget(self.frame_5)
        self.frame_4 = QtWidgets.QFrame(parent=self.frame_3)
        self.frame_4.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.frame_4.setLineWidth(0)
        self.frame_4.setObjectName("frame_4")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.frame_4)
        self.verticalLayout_2.setContentsMargins(14, -1, 0, -1)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_6 = QtWidgets.QLabel(parent=self.frame_4)
        self.label_6.setStyleSheet("")
        self.label_6.setLineWidth(0)
        self.label_6.setOpenExternalLinks(True)
        self.label_6.setObjectName("label_6")
        self.verticalLayout_2.addWidget(self.label_6)
        self.label_2 = QtWidgets.QLabel(parent=self.frame_4)
        self.label_2.setLineWidth(0)
        self.label_2.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignLeading | QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.label_2.setOpenExternalLinks(True)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_2.addWidget(self.label_2)
        self.horizontalLayout_3.addWidget(self.frame_4)
        self.verticalLayout.addWidget(self.frame_3)

        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("Dialog", "About Face Cropper"))
        self.label_7.setText(_translate("Dialog", "Version:"))
        self.label_8.setText(_translate("Dialog", "2.0.0"))
        self.label_4.setText(_translate("Dialog", "Creator:"))
        self.label_3.setText(_translate("Dialog", "Gregory Carnegie"))
        self.label_5.setText(_translate("Dialog", "Credits:"))
        self.label_6.setText(
            _translate("Dialog",
                       "<a href=\"https://leblancfg.com/autocrop/\">François (leblancfg) Leblanc\'s autocrop library</a>"))
        self.label_2.setText(
            _translate("Dialog",
                       "<a href=\"https://bleedai.com/5-easy-effective-face-detection-algorithms-in-python/\">Bleed AI\'s \'Effective Face Detection\' article</a>"))