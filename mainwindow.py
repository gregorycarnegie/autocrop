import re
import utils
import numpy as np
import pandas as pd
import custom_widgets
from cropper import Cropper
from files import Photo, Video, IMAGE_TYPES, VIDEO_TYPES, PANDAS_TYPES
from multiprocessing import Process
from os import startfile
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from pathlib import Path
from typing import Optional, Union


class UiMainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super(UiMainWindow, self).__init__()
        self.setAcceptDrops(True)
        self.model: Optional[custom_widgets.DataFrameModel] = None
        self.data_frame: Optional[pd.DataFrame] = None
        self.validator = QtGui.QIntValidator(100, 10_000)

        self.file_model = QtGui.QFileSystemModel(self)
        self.file_model.setFilter(QtCore.QDir.Filter.NoDotAndDotDot | QtCore.QDir.Filter.Files)
        self.file_model.setNameFilters(Photo().file_filter())

        self.cropper = Cropper()
        self.player = QMediaPlayer()
        self.audio = QAudioOutput()

        self.exposure_string = """QCheckBox:unchecked{color: red}
        QCheckBox:checked{color: white}
        QCheckBox::indicator{
                width: 20px;
                height: 20px;
        }
        QCheckBox::indicator:checked{
                image: url(resources/icons/checkbox_checked.svg);
        }
        QCheckBox::indicator:unchecked{
                image: url(resources/icons/checkbox_unchecked.svg);
        }
        QCheckBox::indicator:checked:hover{
                image: url(resources/icons/checkbox_checked_hover.svg);
        }
        QCheckBox::indicator:unchecked:hover{
                image: url(resources/icons/checkbox_unchecked_hover.svg);
        }"""

        self.radioButton_string = """QRadioButton::indicator:checked{
            image: url(resources/icons/file_string_checked.svg);
            }
            QRadioButton::indicator:unchecked{
                image: url(resources/icons/file_string_unchecked.svg);
            }"""

        self.setObjectName("MainWindow")
        self.resize(1348, 896)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("resources/logos/logo.ico"), QtGui.QIcon.Mode.Normal,
                       QtGui.QIcon.State.Off)
        self.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(parent=self)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.function_tabWidget = QtWidgets.QTabWidget(parent=self.centralwidget)
        self.function_tabWidget.setMovable(True)
        self.function_tabWidget.setObjectName("function_tabWidget")
        self.photoTab = QtWidgets.QWidget()
        self.photoTab.setObjectName("photoTab")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.photoTab)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.photoLineEdit = custom_widgets.PathLineEdit('image', parent=self.photoTab)
        self.photoLineEdit.setMinimumSize(QtCore.QSize(0, 24))
        self.photoLineEdit.setMaximumSize(QtCore.QSize(16_777_215, 24))
        self.photoLineEdit.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhUrlCharactersOnly)
        self.photoLineEdit.setObjectName("photoLineEdit")
        self.horizontalLayout_5.addWidget(self.photoLineEdit)
        self.photoButton = QtWidgets.QPushButton(parent=self.photoTab)
        self.photoButton.setMinimumSize(QtCore.QSize(0, 24))
        self.photoButton.setMaximumSize(QtCore.QSize(16_777_215, 24))
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("resources/icons/picture.svg"), QtGui.QIcon.Mode.Normal,
                        QtGui.QIcon.State.Off)
        self.photoButton.setIcon(icon1)
        self.photoButton.setObjectName("photoButton")
        self.horizontalLayout_5.addWidget(self.photoButton)
        self.verticalLayout_8.addLayout(self.horizontalLayout_5)
        self.frame_8 = QtWidgets.QFrame(parent=self.photoTab)
        self.frame_8.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.frame_8.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.frame_8.setObjectName("frame_8")
        self.verticalLayout_20 = QtWidgets.QVBoxLayout(self.frame_8)
        self.verticalLayout_20.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_20.setSpacing(0)
        self.verticalLayout_20.setObjectName("verticalLayout_20")
        self.frame = QtWidgets.QFrame(parent=self.frame_8)
        self.frame.setMinimumSize(QtCore.QSize(0, 40))
        self.frame.setMaximumSize(QtCore.QSize(16_777_215, 40))
        self.frame.setStyleSheet("background: #1f2c33")
        self.frame.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.frame.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.frame.setObjectName("frame")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout(self.frame)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.exposureCheckBox_1 = QtWidgets.QCheckBox(parent=self.frame)
        self.exposureCheckBox_1.setStyleSheet(self.exposure_string)
        self.exposureCheckBox_1.setObjectName("exposureCheckBox_1")
        self.horizontalLayout_8.addWidget(self.exposureCheckBox_1, 0, QtCore.Qt.AlignmentFlag.AlignRight)
        self.verticalLayout_20.addWidget(self.frame)
        self.frame_13 = QtWidgets.QFrame(parent=self.frame_8)
        self.frame_13.setStyleSheet("background: #1f2c33")
        self.frame_13.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.frame_13.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.frame_13.setObjectName("frame_13")
        self.verticalLayout_29 = QtWidgets.QVBoxLayout(self.frame_13)
        self.verticalLayout_29.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_29.setSpacing(0)
        self.verticalLayout_29.setObjectName("verticalLayout_29")
        self.photoWidget = custom_widgets.ImageWidget(parent=self.frame_13)
        self.photoWidget.setStyleSheet("")
        self.photoWidget.setObjectName("photoWidget")
        self.verticalLayout_29.addWidget(self.photoWidget)
        self.verticalLayout_20.addWidget(self.frame_13)
        self.frame_2 = QtWidgets.QFrame(parent=self.frame_8)
        self.frame_2.setMinimumSize(QtCore.QSize(0, 40))
        self.frame_2.setMaximumSize(QtCore.QSize(16_777_215, 40))
        self.frame_2.setStyleSheet("background: #1f2c33")
        self.frame_2.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.frame_2.setObjectName("frame_2")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(self.frame_2)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.verticalLayout_20.addWidget(self.frame_2)
        self.verticalLayout_20.setStretch(0, 1)
        self.verticalLayout_20.setStretch(2, 1)
        self.verticalLayout_8.addWidget(self.frame_8)
        self.cropButton_1 = QtWidgets.QPushButton(parent=self.photoTab)
        self.cropButton_1.setMinimumSize(QtCore.QSize(0, 24))
        self.cropButton_1.setMaximumSize(QtCore.QSize(16_777_215, 24))
        self.cropButton_1.setText("")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("resources/icons/crop.svg"), QtGui.QIcon.Mode.Normal,
                        QtGui.QIcon.State.Off)
        self.cropButton_1.setIcon(icon2)
        self.cropButton_1.setObjectName("cropButton_1")
        self.verticalLayout_8.addWidget(self.cropButton_1)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.destinationLineEdit_1 = custom_widgets.PathLineEdit('folder', parent=self.photoTab)
        self.destinationLineEdit_1.setMinimumSize(QtCore.QSize(0, 24))
        self.destinationLineEdit_1.setMaximumSize(QtCore.QSize(16_777_215, 24))
        self.destinationLineEdit_1.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhUrlCharactersOnly)
        self.destinationLineEdit_1.setObjectName("destinationLineEdit_1")
        self.horizontalLayout_6.addWidget(self.destinationLineEdit_1)
        self.destinationButton_1 = QtWidgets.QPushButton(parent=self.photoTab)
        self.destinationButton_1.setMinimumSize(QtCore.QSize(0, 24))
        self.destinationButton_1.setMaximumSize(QtCore.QSize(16_777_215, 24))
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap("resources/icons/folder.svg"), QtGui.QIcon.Mode.Normal,
                        QtGui.QIcon.State.Off)
        self.destinationButton_1.setIcon(icon3)
        self.destinationButton_1.setObjectName("destinationButton_1")
        self.horizontalLayout_6.addWidget(self.destinationButton_1)
        self.verticalLayout_8.addLayout(self.horizontalLayout_6)
        self.verticalLayout_8.setStretch(0, 1)
        self.verticalLayout_8.setStretch(1, 17)
        self.verticalLayout_8.setStretch(2, 1)
        self.verticalLayout_8.setStretch(3, 1)
        self.function_tabWidget.addTab(self.photoTab, icon1, "")
        self.folder_Tab = QtWidgets.QWidget()
        self.folder_Tab.setObjectName("folder_Tab")
        self.verticalLayout_15 = QtWidgets.QVBoxLayout(self.folder_Tab)
        self.verticalLayout_15.setObjectName("verticalLayout_15")
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.folderLineEdit_1 = custom_widgets.PathLineEdit('folder', parent=self.folder_Tab)
        self.folderLineEdit_1.setMinimumSize(QtCore.QSize(0, 24))
        self.folderLineEdit_1.setMaximumSize(QtCore.QSize(16_777_215, 24))
        self.folderLineEdit_1.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhUrlCharactersOnly)
        self.folderLineEdit_1.setObjectName("folderLineEdit_1")
        self.horizontalLayout_11.addWidget(self.folderLineEdit_1)
        self.folderButton_1 = QtWidgets.QPushButton(parent=self.folder_Tab)
        self.folderButton_1.setMinimumSize(QtCore.QSize(0, 24))
        self.folderButton_1.setMaximumSize(QtCore.QSize(16_777_215, 24))
        self.folderButton_1.setIcon(icon3)
        self.folderButton_1.setObjectName("folderButton_1")
        self.horizontalLayout_11.addWidget(self.folderButton_1)
        self.verticalLayout_15.addLayout(self.horizontalLayout_11)
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        self.verticalLayout_16 = QtWidgets.QVBoxLayout()
        self.verticalLayout_16.setSpacing(0)
        self.verticalLayout_16.setObjectName("verticalLayout_16")
        self.frame_3 = QtWidgets.QFrame(parent=self.folder_Tab)
        self.frame_3.setMinimumSize(QtCore.QSize(0, 40))
        self.frame_3.setMaximumSize(QtCore.QSize(16_777_215, 40))
        self.frame_3.setStyleSheet("background: #1f2c33")
        self.frame_3.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.frame_3.setObjectName("frame_3")
        self.horizontalLayout_15 = QtWidgets.QHBoxLayout(self.frame_3)
        self.horizontalLayout_15.setObjectName("horizontalLayout_15")
        self.exposureCheckBox_2 = QtWidgets.QCheckBox(parent=self.frame_3)
        self.exposureCheckBox_2.setStyleSheet(self.exposure_string)
        self.exposureCheckBox_2.setObjectName("exposureCheckBox_2")
        self.horizontalLayout_15.addWidget(self.exposureCheckBox_2, 0, QtCore.Qt.AlignmentFlag.AlignRight)
        self.verticalLayout_16.addWidget(self.frame_3)
        self.frame_12 = QtWidgets.QFrame(parent=self.folder_Tab)
        self.frame_12.setStyleSheet("background: #1f2c33")
        self.frame_12.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.frame_12.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.frame_12.setObjectName("frame_12")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.frame_12)
        self.verticalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_7.setSpacing(0)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.folderWidget = custom_widgets.ImageWidget(parent=self.frame_12)
        self.folderWidget.setStyleSheet("")
        self.folderWidget.setObjectName("folderWidget")
        self.verticalLayout_7.addWidget(self.folderWidget)
        self.verticalLayout_16.addWidget(self.frame_12)
        self.frame_4 = QtWidgets.QFrame(parent=self.folder_Tab)
        self.frame_4.setMinimumSize(QtCore.QSize(0, 40))
        self.frame_4.setMaximumSize(QtCore.QSize(16_777_215, 40))
        self.frame_4.setStyleSheet("background: #1f2c33")
        self.frame_4.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.frame_4.setObjectName("frame_4")
        self.horizontalLayout_16 = QtWidgets.QHBoxLayout(self.frame_4)
        self.horizontalLayout_16.setObjectName("horizontalLayout_16")
        self.progressBar_1 = QtWidgets.QProgressBar(parent=self.frame_4)
        self.progressBar_1.setMinimumSize(QtCore.QSize(0, 12))
        self.progressBar_1.setMaximumSize(QtCore.QSize(16_777_215, 12))
        self.progressBar_1.setProperty("value", 0)
        self.progressBar_1.setTextVisible(False)
        self.progressBar_1.setObjectName("progressBar_1")
        self.horizontalLayout_16.addWidget(self.progressBar_1)
        self.verticalLayout_16.addWidget(self.frame_4)
        self.verticalLayout_16.setStretch(0, 1)
        self.verticalLayout_16.setStretch(1, 10)
        self.verticalLayout_16.setStretch(2, 1)
        self.horizontalLayout_12.addLayout(self.verticalLayout_16)
        self.treeView = QtWidgets.QTreeView(parent=self.folder_Tab)
        self.treeView.setObjectName("treeView")
        self.treeView.setModel(self.file_model)
        self.horizontalLayout_12.addWidget(self.treeView)
        self.horizontalLayout_12.setStretch(0, 4)
        self.horizontalLayout_12.setStretch(1, 3)
        self.verticalLayout_15.addLayout(self.horizontalLayout_12)
        self.horizontalLayout_13 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_13.setObjectName("horizontalLayout_13")
        self.cropButton_2 = QtWidgets.QPushButton(parent=self.folder_Tab)
        self.cropButton_2.setMinimumSize(QtCore.QSize(0, 24))
        self.cropButton_2.setMaximumSize(QtCore.QSize(16_777_215, 24))
        self.cropButton_2.setText("")
        self.cropButton_2.setIcon(icon2)
        self.cropButton_2.setObjectName("cropButton_2")
        self.horizontalLayout_13.addWidget(self.cropButton_2)
        self.cancelButton_1 = QtWidgets.QPushButton(parent=self.folder_Tab)
        self.cancelButton_1.setMinimumSize(QtCore.QSize(0, 24))
        self.cancelButton_1.setMaximumSize(QtCore.QSize(16_777_215, 24))
        self.cancelButton_1.setText("")
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap("resources/icons/cancel.svg"), QtGui.QIcon.Mode.Normal,
                        QtGui.QIcon.State.Off)
        self.cancelButton_1.setIcon(icon4)
        self.cancelButton_1.setObjectName("cancelButton_1")
        self.horizontalLayout_13.addWidget(self.cancelButton_1)
        self.verticalLayout_15.addLayout(self.horizontalLayout_13)
        self.horizontalLayout_14 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_14.setObjectName("horizontalLayout_14")
        self.destinationLineEdit_2 = custom_widgets.PathLineEdit('folder', parent=self.folder_Tab)
        self.destinationLineEdit_2.setMinimumSize(QtCore.QSize(0, 24))
        self.destinationLineEdit_2.setMaximumSize(QtCore.QSize(16_777_215, 24))
        self.destinationLineEdit_2.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhUrlCharactersOnly)
        self.destinationLineEdit_2.setObjectName("destinationLineEdit_2")
        self.horizontalLayout_14.addWidget(self.destinationLineEdit_2)
        self.destinationButton_2 = QtWidgets.QPushButton(parent=self.folder_Tab)
        self.destinationButton_2.setMinimumSize(QtCore.QSize(0, 24))
        self.destinationButton_2.setMaximumSize(QtCore.QSize(16_777_215, 24))
        self.destinationButton_2.setIcon(icon3)
        self.destinationButton_2.setObjectName("destinationButton_2")
        self.horizontalLayout_14.addWidget(self.destinationButton_2)
        self.verticalLayout_15.addLayout(self.horizontalLayout_14)
        self.verticalLayout_15.setStretch(0, 1)
        self.verticalLayout_15.setStretch(1, 17)
        self.verticalLayout_15.setStretch(2, 1)
        self.verticalLayout_15.setStretch(3, 1)
        self.function_tabWidget.addTab(self.folder_Tab, icon3, "")
        self.mappingTab = QtWidgets.QWidget()
        self.mappingTab.setObjectName("mappingTab")
        self.verticalLayout_17 = QtWidgets.QVBoxLayout(self.mappingTab)
        self.verticalLayout_17.setObjectName("verticalLayout_17")
        self.horizontalLayout_17 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_17.setObjectName("horizontalLayout_17")
        self.horizontalLayout_22 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_22.setObjectName("horizontalLayout_22")
        self.horizontalLayout_17.addLayout(self.horizontalLayout_22)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.tableLineEdit = custom_widgets.PathLineEdit('table', parent=self.mappingTab)
        self.tableLineEdit.setMinimumSize(QtCore.QSize(0, 24))
        self.tableLineEdit.setMaximumSize(QtCore.QSize(16_777_215, 24))
        self.tableLineEdit.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhUrlCharactersOnly)
        self.tableLineEdit.setObjectName("tableLineEdit")
        self.gridLayout.addWidget(self.tableLineEdit, 1, 0, 1, 1)
        self.folderLineEdit_2 = custom_widgets.PathLineEdit('folder', parent=self.mappingTab)
        self.folderLineEdit_2.setMinimumSize(QtCore.QSize(0, 24))
        self.folderLineEdit_2.setMaximumSize(QtCore.QSize(16_777_215, 24))
        self.folderLineEdit_2.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhUrlCharactersOnly)
        self.folderLineEdit_2.setObjectName("folderLineEdit_2")
        self.gridLayout.addWidget(self.folderLineEdit_2, 0, 0, 1, 1)
        self.folderButton_2 = QtWidgets.QPushButton(parent=self.mappingTab)
        self.folderButton_2.setMinimumSize(QtCore.QSize(0, 24))
        self.folderButton_2.setMaximumSize(QtCore.QSize(16_777_215, 24))
        self.folderButton_2.setIcon(icon3)
        self.folderButton_2.setObjectName("folderButton_2")
        self.gridLayout.addWidget(self.folderButton_2, 0, 1, 1, 1)
        self.tableButton = QtWidgets.QPushButton(parent=self.mappingTab)
        self.tableButton.setMinimumSize(QtCore.QSize(0, 24))
        self.tableButton.setMaximumSize(QtCore.QSize(16_777_215, 24))
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap("resources/icons/excel.svg"), QtGui.QIcon.Mode.Normal,
                        QtGui.QIcon.State.Off)
        self.tableButton.setIcon(icon5)
        self.tableButton.setObjectName("tableButton")
        self.gridLayout.addWidget(self.tableButton, 1, 1, 1, 1)
        self.horizontalLayout_17.addLayout(self.gridLayout)
        self.verticalLayout_17.addLayout(self.horizontalLayout_17)
        self.horizontalLayout_18 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_18.setObjectName("horizontalLayout_18")
        self.verticalLayout_18 = QtWidgets.QVBoxLayout()
        self.verticalLayout_18.setSpacing(0)
        self.verticalLayout_18.setObjectName("verticalLayout_18")
        self.frame_5 = QtWidgets.QFrame(parent=self.mappingTab)
        self.frame_5.setStyleSheet("background: #1f2c33")
        self.frame_5.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.frame_5.setObjectName("frame_5")
        self.horizontalLayout_23 = QtWidgets.QHBoxLayout(self.frame_5)
        self.horizontalLayout_23.setObjectName("horizontalLayout_23")
        self.exposureCheckBox_3 = QtWidgets.QCheckBox(parent=self.frame_5)
        self.exposureCheckBox_3.setStyleSheet(self.exposure_string)
        self.exposureCheckBox_3.setObjectName("exposureCheckBox_3")
        self.horizontalLayout_23.addWidget(self.exposureCheckBox_3, 0, QtCore.Qt.AlignmentFlag.AlignRight)
        self.verticalLayout_18.addWidget(self.frame_5)
        self.frame_6 = QtWidgets.QFrame(parent=self.mappingTab)
        self.frame_6.setStyleSheet("background: #1f2c33")
        self.frame_6.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.frame_6.setObjectName("frame_6")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.frame_6)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.mappingWidget = custom_widgets.ImageWidget(parent=self.frame_6)
        self.mappingWidget.setStyleSheet("")
        self.mappingWidget.setObjectName("mappingWidget")
        self.verticalLayout.addWidget(self.mappingWidget)
        self.verticalLayout_18.addWidget(self.frame_6)
        self.frame_7 = QtWidgets.QFrame(parent=self.mappingTab)
        self.frame_7.setStyleSheet("background: #1f2c33")
        self.frame_7.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.frame_7.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.frame_7.setObjectName("frame_7")
        self.horizontalLayout_21 = QtWidgets.QHBoxLayout(self.frame_7)
        self.horizontalLayout_21.setObjectName("horizontalLayout_21")
        self.progressBar_2 = QtWidgets.QProgressBar(parent=self.frame_7)
        self.progressBar_2.setMinimumSize(QtCore.QSize(0, 12))
        self.progressBar_2.setMaximumSize(QtCore.QSize(16_777_215, 12))
        self.progressBar_2.setProperty("value", 0)
        self.progressBar_2.setTextVisible(False)
        self.progressBar_2.setObjectName("progressBar_2")
        self.horizontalLayout_21.addWidget(self.progressBar_2)
        self.verticalLayout_18.addWidget(self.frame_7)
        self.verticalLayout_18.setStretch(0, 1)
        self.verticalLayout_18.setStretch(1, 10)
        self.verticalLayout_18.setStretch(2, 1)
        self.horizontalLayout_18.addLayout(self.verticalLayout_18)
        self.verticalLayout_19 = QtWidgets.QVBoxLayout()
        self.verticalLayout_19.setObjectName("verticalLayout_19")
        self.tableView = QtWidgets.QTableView(parent=self.mappingTab)
        self.tableView.setObjectName("tableView")
        self.verticalLayout_19.addWidget(self.tableView)
        self.horizontalLayout_24 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_24.setObjectName("horizontalLayout_24")
        self.comboBox = QtWidgets.QComboBox(parent=self.mappingTab)
        self.comboBox.setMinimumSize(QtCore.QSize(0, 22))
        self.comboBox.setMaximumSize(QtCore.QSize(16_777_215, 22))
        self.comboBox.setObjectName("comboBox")
        self.horizontalLayout_24.addWidget(self.comboBox)
        self.comboBox_2 = QtWidgets.QComboBox(parent=self.mappingTab)
        self.comboBox_2.setMinimumSize(QtCore.QSize(0, 22))
        self.comboBox_2.setMaximumSize(QtCore.QSize(16_777_215, 22))
        self.comboBox_2.setObjectName("comboBox_2")
        self.horizontalLayout_24.addWidget(self.comboBox_2)
        self.verticalLayout_19.addLayout(self.horizontalLayout_24)
        self.horizontalLayout_18.addLayout(self.verticalLayout_19)
        self.horizontalLayout_18.setStretch(0, 1)
        self.horizontalLayout_18.setStretch(1, 2)
        self.verticalLayout_17.addLayout(self.horizontalLayout_18)
        self.horizontalLayout_19 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_19.setObjectName("horizontalLayout_19")
        self.cropButton_3 = QtWidgets.QPushButton(parent=self.mappingTab)
        self.cropButton_3.setMinimumSize(QtCore.QSize(0, 24))
        self.cropButton_3.setMaximumSize(QtCore.QSize(16_777_215, 24))
        self.cropButton_3.setText("")
        self.cropButton_3.setIcon(icon2)
        self.cropButton_3.setObjectName("cropButton_3")
        self.horizontalLayout_19.addWidget(self.cropButton_3)
        self.cancelButton_2 = QtWidgets.QPushButton(parent=self.mappingTab)
        self.cancelButton_2.setMinimumSize(QtCore.QSize(0, 24))
        self.cancelButton_2.setMaximumSize(QtCore.QSize(16_777_215, 24))
        self.cancelButton_2.setText("")
        self.cancelButton_2.setIcon(icon4)
        self.cancelButton_2.setObjectName("cancelButton_2")
        self.horizontalLayout_19.addWidget(self.cancelButton_2)
        self.verticalLayout_17.addLayout(self.horizontalLayout_19)
        self.horizontalLayout_20 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_20.setObjectName("horizontalLayout_20")
        self.destinationLineEdit_3 = custom_widgets.PathLineEdit('folder', parent=self.mappingTab)
        self.destinationLineEdit_3.setMinimumSize(QtCore.QSize(0, 24))
        self.destinationLineEdit_3.setMaximumSize(QtCore.QSize(16_777_215, 24))
        self.destinationLineEdit_3.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhUrlCharactersOnly)
        self.destinationLineEdit_3.setObjectName("destinationLineEdit_3")
        self.horizontalLayout_20.addWidget(self.destinationLineEdit_3)
        self.destinationButton_3 = QtWidgets.QPushButton(parent=self.mappingTab)
        self.destinationButton_3.setMinimumSize(QtCore.QSize(0, 24))
        self.destinationButton_3.setMaximumSize(QtCore.QSize(16_777_215, 24))
        self.destinationButton_3.setIcon(icon3)
        self.destinationButton_3.setObjectName("destinationButton_3")
        self.horizontalLayout_20.addWidget(self.destinationButton_3)
        self.verticalLayout_17.addLayout(self.horizontalLayout_20)
        self.verticalLayout_17.setStretch(0, 2)
        self.verticalLayout_17.setStretch(1, 15)
        self.verticalLayout_17.setStretch(2, 1)
        self.verticalLayout_17.setStretch(3, 1)
        self.function_tabWidget.addTab(self.mappingTab, icon5, "")
        self.videoTab = QtWidgets.QWidget()
        self.videoTab.setObjectName("videoTab")
        self.verticalLayout_22 = QtWidgets.QVBoxLayout(self.videoTab)
        self.verticalLayout_22.setObjectName("verticalLayout_22")
        self.horizontalLayout_25 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_25.setObjectName("horizontalLayout_25")
        self.videoLineEdit = custom_widgets.PathLineEdit('video', parent=self.videoTab)
        self.videoLineEdit.setMinimumSize(QtCore.QSize(0, 24))
        self.videoLineEdit.setMaximumSize(QtCore.QSize(16_777_215, 24))
        self.videoLineEdit.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhUrlCharactersOnly)
        self.videoLineEdit.setObjectName("videoLineEdit")
        self.horizontalLayout_25.addWidget(self.videoLineEdit)
        self.videoButton = QtWidgets.QPushButton(parent=self.videoTab)
        self.videoButton.setMaximumSize(QtCore.QSize(16_777_215, 24))
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap("resources/icons/clapperboard.svg"), QtGui.QIcon.Mode.Normal,
                        QtGui.QIcon.State.Off)
        self.videoButton.setIcon(icon6)
        self.videoButton.setObjectName("videoButton")
        self.horizontalLayout_25.addWidget(self.videoButton)
        self.verticalLayout_22.addLayout(self.horizontalLayout_25)
        self.frame_9 = QtWidgets.QFrame(parent=self.videoTab)
        self.frame_9.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.frame_9.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.frame_9.setObjectName("frame_9")
        self.verticalLayout_23 = QtWidgets.QVBoxLayout(self.frame_9)
        self.verticalLayout_23.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_23.setSpacing(0)
        self.verticalLayout_23.setObjectName("verticalLayout_23")
        self.frame_10 = QtWidgets.QFrame(parent=self.frame_9)
        self.frame_10.setMaximumSize(QtCore.QSize(16_777_215, 42))
        self.frame_10.setStyleSheet("background: #1f2c33")
        self.frame_10.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.frame_10.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.frame_10.setObjectName("frame_10")
        self.horizontalLayout_32 = QtWidgets.QHBoxLayout(self.frame_10)
        self.horizontalLayout_32.setObjectName("horizontalLayout_32")
        self.muteButton = QtWidgets.QPushButton(parent=self.frame_10)
        self.muteButton.setText("")
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap("resources/icons/multimedia_mute.svg"), QtGui.QIcon.Mode.Normal,
                        QtGui.QIcon.State.Off)
        self.muteButton.setIcon(icon7)
        self.muteButton.setObjectName("muteButton")
        self.horizontalLayout_32.addWidget(self.muteButton)
        self.volumeSlider = QtWidgets.QSlider(parent=self.frame_10)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.volumeSlider.sizePolicy().hasHeightForWidth())
        self.volumeSlider.setSizePolicy(sizePolicy)
        self.volumeSlider.setMinimum(-1)
        self.volumeSlider.setMaximum(100)
        self.volumeSlider.setProperty("value", 70)
        self.volumeSlider.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.volumeSlider.setObjectName("volumeSlider")
        self.horizontalLayout_32.addWidget(self.volumeSlider)
        self.positionLabel = QtWidgets.QLabel(parent=self.frame_10)
        self.positionLabel.setObjectName("positionLabel")
        self.horizontalLayout_32.addWidget(self.positionLabel)
        self.timelineSlider = QtWidgets.QSlider(parent=self.frame_10)
        # self.timelineSlider = utils.ClickableSlider(QtCore.Qt.Orientation.Horizontal, parent=self.frame_10)
        self.timelineSlider.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.timelineSlider.setObjectName("timelineSlider")
        
        self.horizontalLayout_32.addWidget(self.timelineSlider)
        self.durationLabel = QtWidgets.QLabel(parent=self.frame_10)
        self.durationLabel.setObjectName("durationLabel")
        self.horizontalLayout_32.addWidget(self.durationLabel)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Fixed,
                                           QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout_32.addItem(spacerItem)
        self.exposureCheckBox_4 = QtWidgets.QCheckBox(parent=self.frame_10)
        self.exposureCheckBox_4.setStyleSheet(self.exposure_string)
        self.exposureCheckBox_4.setObjectName("exposureCheckBox_4")
        self.horizontalLayout_32.addWidget(self.exposureCheckBox_4, 0, QtCore.Qt.AlignmentFlag.AlignRight)
        self.verticalLayout_23.addWidget(self.frame_10)
        self.videoWidget = QVideoWidget(parent=self.frame_9)
        self.videoWidget.setStyleSheet("background: #1f2c33")
        self.videoWidget.setObjectName("videoWidget")
        self.verticalLayout_23.addWidget(self.videoWidget)
        self.frame_11 = QtWidgets.QFrame(parent=self.frame_9)
        self.frame_11.setMaximumSize(QtCore.QSize(16_777_215, 34))
        self.frame_11.setStyleSheet("background: #1f2c33")
        self.frame_11.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.frame_11.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.frame_11.setObjectName("frame_11")
        self.horizontalLayout_31 = QtWidgets.QHBoxLayout(self.frame_11)
        self.horizontalLayout_31.setObjectName("horizontalLayout_31")
        self.progressBar_3 = QtWidgets.QProgressBar(parent=self.frame_11)
        self.progressBar_3.setMinimumSize(QtCore.QSize(0, 12))
        self.progressBar_3.setMaximumSize(QtCore.QSize(16_777_215, 12))
        self.progressBar_3.setProperty("value", 0)
        self.progressBar_3.setTextVisible(False)
        self.progressBar_3.setObjectName("progressBar_3")
        self.horizontalLayout_31.addWidget(self.progressBar_3)
        self.verticalLayout_23.addWidget(self.frame_11)
        self.verticalLayout_23.setStretch(0, 1)
        self.verticalLayout_23.setStretch(1, 10)
        self.verticalLayout_23.setStretch(2, 1)
        self.verticalLayout_22.addWidget(self.frame_9)
        self.horizontalLayout_26 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_26.setObjectName("horizontalLayout_26")
        self.cropButton_4 = QtWidgets.QPushButton(parent=self.videoTab)
        self.cropButton_4.setMinimumSize(QtCore.QSize(0, 24))
        self.cropButton_4.setMaximumSize(QtCore.QSize(16_777_215, 24))
        self.cropButton_4.setText("")
        self.cropButton_4.setIcon(icon2)
        self.cropButton_4.setObjectName("cropButton_4")
        self.horizontalLayout_26.addWidget(self.cropButton_4)
        self.videocropButton = QtWidgets.QPushButton(parent=self.videoTab)
        self.videocropButton.setMinimumSize(QtCore.QSize(0, 24))
        self.videocropButton.setMaximumSize(QtCore.QSize(16_777_215, 24))
        self.videocropButton.setText("")
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap("resources/icons/crop_video.svg"), QtGui.QIcon.Mode.Normal,
                        QtGui.QIcon.State.Off)
        self.videocropButton.setIcon(icon8)
        self.videocropButton.setObjectName("videocropButton")
        self.horizontalLayout_26.addWidget(self.videocropButton)
        self.cancelButton_3 = QtWidgets.QPushButton(parent=self.videoTab)
        self.cancelButton_3.setMinimumSize(QtCore.QSize(0, 24))
        self.cancelButton_3.setMaximumSize(QtCore.QSize(16_777_215, 24))
        self.cancelButton_3.setText("")
        self.cancelButton_3.setIcon(icon4)
        self.cancelButton_3.setObjectName("cancelButton_3")
        self.horizontalLayout_26.addWidget(self.cancelButton_3)
        self.verticalLayout_22.addLayout(self.horizontalLayout_26)
        self.horizontalLayout_30 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_30.setObjectName("horizontalLayout_30")
        self.destinationLineEdit_4 = custom_widgets.PathLineEdit('folder', parent=self.videoTab)
        self.destinationLineEdit_4.setMinimumSize(QtCore.QSize(0, 24))
        self.destinationLineEdit_4.setMaximumSize(QtCore.QSize(16_777_215, 24))
        self.destinationLineEdit_4.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhUrlCharactersOnly)
        self.destinationLineEdit_4.setObjectName("destinationLineEdit_4")
        self.horizontalLayout_30.addWidget(self.destinationLineEdit_4)
        self.destinationButton_4 = QtWidgets.QPushButton(parent=self.videoTab)
        self.destinationButton_4.setMinimumSize(QtCore.QSize(0, 24))
        self.destinationButton_4.setMaximumSize(QtCore.QSize(16_777_215, 24))
        self.destinationButton_4.setIcon(icon3)
        self.destinationButton_4.setObjectName("destinationButton_4")
        self.horizontalLayout_30.addWidget(self.destinationButton_4)
        self.verticalLayout_22.addLayout(self.horizontalLayout_30)
        self.horizontalLayout_28 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_28.setObjectName("horizontalLayout_28")
        self.horizontalLayout_29 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_29.setObjectName("horizontalLayout_29")
        self.playButton = QtWidgets.QPushButton(parent=self.videoTab)
        self.playButton.setEnabled(True)
        self.playButton.setMaximumSize(QtCore.QSize(48, 48))
        self.playButton.setText("")
        icon9 = QtGui.QIcon()
        icon9.addPixmap(QtGui.QPixmap("resources/icons/multimedia_play.svg"), QtGui.QIcon.Mode.Normal,
                        QtGui.QIcon.State.Off)
        self.playButton.setIcon(icon9)
        self.playButton.setIconSize(QtCore.QSize(32, 32))
        self.playButton.setObjectName("playButton")
        self.horizontalLayout_29.addWidget(self.playButton)
        self.stopButton = QtWidgets.QPushButton(parent=self.videoTab)
        self.stopButton.setMinimumSize(QtCore.QSize(48, 48))
        self.stopButton.setMaximumSize(QtCore.QSize(48, 48))
        self.stopButton.setText("")
        icon10 = QtGui.QIcon()
        icon10.addPixmap(QtGui.QPixmap("resources/icons/multimedia_stop.svg"), QtGui.QIcon.Mode.Normal,
                         QtGui.QIcon.State.Off)
        self.stopButton.setIcon(icon10)
        self.stopButton.setIconSize(QtCore.QSize(32, 32))
        self.stopButton.setObjectName("stopButton")
        self.horizontalLayout_29.addWidget(self.stopButton)
        self.stepbackButton = QtWidgets.QPushButton(parent=self.videoTab)
        self.stepbackButton.setMinimumSize(QtCore.QSize(48, 48))
        self.stepbackButton.setMaximumSize(QtCore.QSize(48, 48))
        self.stepbackButton.setText("")
        icon11 = QtGui.QIcon()
        icon11.addPixmap(QtGui.QPixmap("resources/icons/multimedia_left.svg"), QtGui.QIcon.Mode.Normal,
                         QtGui.QIcon.State.Off)
        self.stepbackButton.setIcon(icon11)
        self.stepbackButton.setIconSize(QtCore.QSize(32, 32))
        self.stepbackButton.setObjectName("stepbackButton")
        self.horizontalLayout_29.addWidget(self.stepbackButton)
        self.stepfwdButton = QtWidgets.QPushButton(parent=self.videoTab)
        self.stepfwdButton.setMinimumSize(QtCore.QSize(48, 48))
        self.stepfwdButton.setMaximumSize(QtCore.QSize(48, 48))
        self.stepfwdButton.setText("")
        icon12 = QtGui.QIcon()
        icon12.addPixmap(QtGui.QPixmap("resources/icons/multimedia_right.svg"), QtGui.QIcon.Mode.Normal,
                         QtGui.QIcon.State.Off)
        self.stepfwdButton.setIcon(icon12)
        self.stepfwdButton.setIconSize(QtCore.QSize(32, 32))
        self.stepfwdButton.setObjectName("stepfwdButton")
        self.horizontalLayout_29.addWidget(self.stepfwdButton)
        self.rewindButton = QtWidgets.QPushButton(parent=self.videoTab)
        self.rewindButton.setMinimumSize(QtCore.QSize(48, 48))
        self.rewindButton.setMaximumSize(QtCore.QSize(48, 48))
        self.rewindButton.setText("")
        icon13 = QtGui.QIcon()
        icon13.addPixmap(QtGui.QPixmap("resources/icons/multimedia_rewind.svg"), QtGui.QIcon.Mode.Normal,
                         QtGui.QIcon.State.Off)
        self.rewindButton.setIcon(icon13)
        self.rewindButton.setIconSize(QtCore.QSize(32, 32))
        self.rewindButton.setObjectName("rewindButton")
        self.horizontalLayout_29.addWidget(self.rewindButton)
        self.fastfwdButton = QtWidgets.QPushButton(parent=self.videoTab)
        self.fastfwdButton.setMinimumSize(QtCore.QSize(48, 48))
        self.fastfwdButton.setMaximumSize(QtCore.QSize(48, 48))
        self.fastfwdButton.setText("")
        icon14 = QtGui.QIcon()
        icon14.addPixmap(QtGui.QPixmap("resources/icons/multimedia_fastfwd.svg"), QtGui.QIcon.Mode.Normal,
                         QtGui.QIcon.State.Off)
        self.fastfwdButton.setIcon(icon14)
        self.fastfwdButton.setIconSize(QtCore.QSize(32, 32))
        self.fastfwdButton.setObjectName("fastfwdButton")
        self.horizontalLayout_29.addWidget(self.fastfwdButton)
        self.pushButton_26 = QtWidgets.QPushButton(parent=self.videoTab)
        self.pushButton_26.setMinimumSize(QtCore.QSize(48, 48))
        self.pushButton_26.setMaximumSize(QtCore.QSize(48, 48))
        self.pushButton_26.setText("")
        icon15 = QtGui.QIcon()
        icon15.addPixmap(QtGui.QPixmap("resources/icons/multimedia_begining.svg"), QtGui.QIcon.Mode.Normal,
                         QtGui.QIcon.State.Off)
        self.pushButton_26.setIcon(icon15)
        self.pushButton_26.setIconSize(QtCore.QSize(32, 32))
        self.pushButton_26.setObjectName("pushButton_26")
        self.horizontalLayout_29.addWidget(self.pushButton_26)
        self.pushButton_27 = QtWidgets.QPushButton(parent=self.videoTab)
        self.pushButton_27.setMinimumSize(QtCore.QSize(48, 48))
        self.pushButton_27.setMaximumSize(QtCore.QSize(48, 48))
        self.pushButton_27.setText("")
        icon16 = QtGui.QIcon()
        icon16.addPixmap(QtGui.QPixmap("resources/icons/multimedia_end.svg"), QtGui.QIcon.Mode.Normal,
                         QtGui.QIcon.State.Off)
        self.pushButton_27.setIcon(icon16)
        self.pushButton_27.setIconSize(QtCore.QSize(32, 32))
        self.pushButton_27.setObjectName("pushButton_27")
        self.horizontalLayout_29.addWidget(self.pushButton_27)
        self.startmarkerButton = QtWidgets.QPushButton(parent=self.videoTab)
        self.startmarkerButton.setMinimumSize(QtCore.QSize(48, 48))
        self.startmarkerButton.setMaximumSize(QtCore.QSize(48, 48))
        self.startmarkerButton.setText("")
        icon17 = QtGui.QIcon()
        icon17.addPixmap(QtGui.QPixmap("resources/icons/multimedia_leftmarker.svg"), QtGui.QIcon.Mode.Normal,
                         QtGui.QIcon.State.Off)
        self.startmarkerButton.setIcon(icon17)
        self.startmarkerButton.setIconSize(QtCore.QSize(32, 32))
        self.startmarkerButton.setObjectName("startmarkerButton")
        self.horizontalLayout_29.addWidget(self.startmarkerButton)
        self.endmarkerButton = QtWidgets.QPushButton(parent=self.videoTab)
        self.endmarkerButton.setMinimumSize(QtCore.QSize(48, 48))
        self.endmarkerButton.setMaximumSize(QtCore.QSize(48, 48))
        self.endmarkerButton.setText("")
        icon18 = QtGui.QIcon()
        icon18.addPixmap(QtGui.QPixmap("resources/icons/multimedia_rightmarker.svg"), QtGui.QIcon.Mode.Normal,
                         QtGui.QIcon.State.Off)
        self.endmarkerButton.setIcon(icon18)
        self.endmarkerButton.setIconSize(QtCore.QSize(32, 32))
        self.endmarkerButton.setObjectName("endmarkerButton")
        self.horizontalLayout_29.addWidget(self.endmarkerButton)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding,
                                            QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout_29.addItem(spacerItem1)
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.selectStartMarkerButton = QtWidgets.QPushButton(parent=self.videoTab)
        self.selectStartMarkerButton.setMinimumSize(QtCore.QSize(150, 0))
        self.selectStartMarkerButton.setObjectName("selectStartMarkerButton")
        self.gridLayout_2.addWidget(self.selectStartMarkerButton, 0, 1, 1, 1)
        self.label_15 = QtWidgets.QLabel(parent=self.videoTab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                                           QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_15.sizePolicy().hasHeightForWidth())
        self.label_15.setSizePolicy(sizePolicy)
        self.label_15.setMaximumSize(QtCore.QSize(14, 14))
        self.label_15.setText("")
        self.label_15.setPixmap(QtGui.QPixmap("resources/icons/marker_label_a.svg"))
        self.label_15.setScaledContents(True)
        self.label_15.setObjectName("label_15")
        self.gridLayout_2.addWidget(self.label_15, 0, 0, 1, 1)
        self.selectEndMarkerButton = QtWidgets.QPushButton(parent=self.videoTab)
        self.selectEndMarkerButton.setMinimumSize(QtCore.QSize(150, 0))
        self.selectEndMarkerButton.setObjectName("selectEndMarkerButton")
        self.gridLayout_2.addWidget(self.selectEndMarkerButton, 1, 1, 1, 1)
        self.label_16 = QtWidgets.QLabel(parent=self.videoTab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                                           QtWidgets.QSizePolicy.Policy.Preferred)
        self.video = Video(self.audio, self.videoWidget, self.player, self.timelineSlider, self.volumeSlider, 
                           self.positionLabel, self.durationLabel, self.selectEndMarkerButton)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_16.sizePolicy().hasHeightForWidth())
        self.label_16.setSizePolicy(sizePolicy)
        self.label_16.setMaximumSize(QtCore.QSize(14, 14))
        self.label_16.setText("")
        self.label_16.setPixmap(QtGui.QPixmap("resources/icons/marker_label_b.svg"))
        self.label_16.setScaledContents(True)
        self.label_16.setObjectName("label_16")
        self.gridLayout_2.addWidget(self.label_16, 1, 0, 1, 1)
        self.horizontalLayout_29.addLayout(self.gridLayout_2)
        self.horizontalLayout_28.addLayout(self.horizontalLayout_29)
        self.horizontalLayout_28.setStretch(0, 1)
        self.verticalLayout_22.addLayout(self.horizontalLayout_28)
        self.verticalLayout_22.setStretch(0, 1)
        self.verticalLayout_22.setStretch(1, 19)
        self.verticalLayout_22.setStretch(2, 1)
        self.verticalLayout_22.setStretch(4, 2)
        self.function_tabWidget.addTab(self.videoTab, icon6, "")
        self.verticalLayout_2.addWidget(self.function_tabWidget)
        self.settings_tabWidget = QtWidgets.QTabWidget(parent=self.centralwidget)
        self.settings_tabWidget.setTabShape(QtWidgets.QTabWidget.TabShape.Rounded)
        self.settings_tabWidget.setTabsClosable(False)
        self.settings_tabWidget.setMovable(True)
        self.settings_tabWidget.setTabBarAutoHide(False)
        self.settings_tabWidget.setObjectName("settings_tabWidget")
        self.settingsTab = QtWidgets.QWidget()
        self.settingsTab.setObjectName("settingsTab")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.settingsTab)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.gammaDial = custom_widgets.CustomQDial(parent=self.settingsTab)
        self.gammaDial.setMinimum(1)
        self.gammaDial.setMaximum(2_000)
        self.gammaDial.setSingleStep(5)
        self.gammaDial.setPageStep(100)
        self.gammaDial.setProperty("value", 1_000)
        self.gammaDial.setSliderPosition(1_000)
        self.gammaDial.setInvertedAppearance(True)
        self.gammaDial.setInvertedControls(False)
        self.gammaDial.setWrapping(False)
        self.gammaDial.setNotchesVisible(True)
        self.gammaDial.setObjectName("gammaDial")
        self.verticalLayout_3.addWidget(self.gammaDial)
        self.label = QtWidgets.QLabel(parent=self.settingsTab)
        self.label.setObjectName("label")
        self.verticalLayout_3.addWidget(self.label, 0, QtCore.Qt.AlignmentFlag.AlignHCenter)
        self.verticalLayout_3.setStretch(0, 7)
        self.verticalLayout_3.setStretch(1, 1)
        self.horizontalLayout_3.addLayout(self.verticalLayout_3)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.faceDial = custom_widgets.CustomQDial(parent=self.settingsTab)
        self.faceDial.setMaximum(100)
        self.faceDial.setProperty("value", 62)
        self.faceDial.setNotchesVisible(True)
        self.faceDial.setObjectName("faceDial")
        self.verticalLayout_4.addWidget(self.faceDial)
        self.label_2 = QtWidgets.QLabel(parent=self.settingsTab)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_4.addWidget(self.label_2, 0, QtCore.Qt.AlignmentFlag.AlignHCenter)
        self.verticalLayout_4.setStretch(0, 7)
        self.verticalLayout_4.setStretch(1, 1)
        self.horizontalLayout_3.addLayout(self.verticalLayout_4)
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.sensitivityDial = custom_widgets.CustomQDial(parent=self.settingsTab)
        self.sensitivityDial.setMaximum(100)
        self.sensitivityDial.setProperty("value", 50)
        self.sensitivityDial.setNotchesVisible(True)
        self.sensitivityDial.setObjectName("sensitivityDial")
        self.verticalLayout_5.addWidget(self.sensitivityDial)
        self.label_3 = QtWidgets.QLabel(parent=self.settingsTab)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_5.addWidget(self.label_3, 0, QtCore.Qt.AlignmentFlag.AlignHCenter)
        self.verticalLayout_5.setStretch(0, 7)
        self.verticalLayout_5.setStretch(1, 1)
        self.horizontalLayout_3.addLayout(self.verticalLayout_5)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding,
                                            QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem2)
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout()
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.label_4 = QtWidgets.QLabel(parent=self.settingsTab)
        self.label_4.setObjectName("label_4")
        self.verticalLayout_9.addWidget(self.label_4)
        self.widthLineEdit = custom_widgets.NumberLineEdit(parent=self.settingsTab)
        self.widthLineEdit.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhDigitsOnly)
        self.widthLineEdit.setText("")
        self.widthLineEdit.setValidator(self.validator)
        self.widthLineEdit.setObjectName("widthLineEdit")
        self.verticalLayout_9.addWidget(self.widthLineEdit)
        self.horizontalLayout_9.addLayout(self.verticalLayout_9)
        self.verticalLayout_10 = QtWidgets.QVBoxLayout()
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.label_5 = QtWidgets.QLabel(parent=self.settingsTab)
        self.label_5.setObjectName("label_5")
        self.verticalLayout_10.addWidget(self.label_5)
        self.heightLineEdit = custom_widgets.NumberLineEdit(parent=self.settingsTab)
        self.heightLineEdit.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhDigitsOnly)
        self.heightLineEdit.setText("")
        self.heightLineEdit.setValidator(self.validator)
        self.heightLineEdit.setObjectName("heightLineEdit")
        self.verticalLayout_10.addWidget(self.heightLineEdit)
        self.horizontalLayout_9.addLayout(self.verticalLayout_10)
        self.verticalLayout_6.addLayout(self.horizontalLayout_9)
        self.label_7 = QtWidgets.QLabel(parent=self.settingsTab)
        self.label_7.setObjectName("label_7")
        self.verticalLayout_6.addWidget(self.label_7, 0, QtCore.Qt.AlignmentFlag.AlignHCenter)
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.verticalLayout_14 = QtWidgets.QVBoxLayout()
        self.verticalLayout_14.setObjectName("verticalLayout_14")
        self.topDial = custom_widgets.CustomQDial(parent=self.settingsTab)
        self.topDial.setMaximum(30)
        self.topDial.setNotchesVisible(True)
        self.topDial.setObjectName("topDial")
        self.verticalLayout_14.addWidget(self.topDial)
        self.label_8 = QtWidgets.QLabel(parent=self.settingsTab)
        self.label_8.setObjectName("label_8")
        self.verticalLayout_14.addWidget(self.label_8, 0, QtCore.Qt.AlignmentFlag.AlignHCenter)
        self.horizontalLayout_10.addLayout(self.verticalLayout_14)
        self.verticalLayout_13 = QtWidgets.QVBoxLayout()
        self.verticalLayout_13.setObjectName("verticalLayout_13")
        self.bottomDial = custom_widgets.CustomQDial(parent=self.settingsTab)
        self.bottomDial.setMaximum(30)
        self.bottomDial.setNotchesVisible(True)
        self.bottomDial.setObjectName("bottomDial")
        self.verticalLayout_13.addWidget(self.bottomDial)
        self.label_9 = QtWidgets.QLabel(parent=self.settingsTab)
        self.label_9.setObjectName("label_9")
        self.verticalLayout_13.addWidget(self.label_9, 0, QtCore.Qt.AlignmentFlag.AlignHCenter)
        self.horizontalLayout_10.addLayout(self.verticalLayout_13)
        self.verticalLayout_12 = QtWidgets.QVBoxLayout()
        self.verticalLayout_12.setObjectName("verticalLayout_12")
        self.leftDial = custom_widgets.CustomQDial(parent=self.settingsTab)
        self.leftDial.setMaximum(30)
        self.leftDial.setNotchesVisible(True)
        self.leftDial.setObjectName("leftDial")
        self.verticalLayout_12.addWidget(self.leftDial)
        self.label_10 = QtWidgets.QLabel(parent=self.settingsTab)
        self.label_10.setObjectName("label_10")
        self.verticalLayout_12.addWidget(self.label_10, 0, QtCore.Qt.AlignmentFlag.AlignHCenter)
        self.horizontalLayout_10.addLayout(self.verticalLayout_12)
        self.verticalLayout_11 = QtWidgets.QVBoxLayout()
        self.verticalLayout_11.setObjectName("verticalLayout_11")
        self.rightDial = custom_widgets.CustomQDial(parent=self.settingsTab)
        self.rightDial.setMaximum(30)
        self.rightDial.setNotchesVisible(True)
        self.rightDial.setObjectName("rightDial")
        self.verticalLayout_11.addWidget(self.rightDial)
        self.label_11 = QtWidgets.QLabel(parent=self.settingsTab)
        self.label_11.setObjectName("label_11")
        self.verticalLayout_11.addWidget(self.label_11, 0, QtCore.Qt.AlignmentFlag.AlignHCenter)
        self.horizontalLayout_10.addLayout(self.verticalLayout_11)
        self.verticalLayout_6.addLayout(self.horizontalLayout_10)
        self.horizontalLayout_3.addLayout(self.verticalLayout_6)
        self.horizontalLayout_3.setStretch(0, 1)
        self.horizontalLayout_3.setStretch(1, 1)
        self.horizontalLayout_3.setStretch(2, 1)
        self.horizontalLayout_3.setStretch(4, 2)
        icon19 = QtGui.QIcon()
        icon19.addPixmap(QtGui.QPixmap("resources/icons/settings.svg"), QtGui.QIcon.Mode.Normal,
                         QtGui.QIcon.State.Off)
        self.settings_tabWidget.addTab(self.settingsTab, icon19, "")
        self.formatTab = QtWidgets.QWidget()
        self.formatTab.setObjectName("formatTab")
        self.horizontalLayout_27 = QtWidgets.QHBoxLayout(self.formatTab)
        self.horizontalLayout_27.setObjectName("horizontalLayout_27")
        self.verticalLayout_21 = QtWidgets.QVBoxLayout()
        self.verticalLayout_21.setObjectName("verticalLayout_21")
        self.radioButton_1 = QtWidgets.QRadioButton(parent=self.formatTab)
        self.radioButton_1.setStyleSheet(re.sub('_string', '_no', self.radioButton_string))
        self.radioButton_1.setText("")
        self.radioButton_1.setIconSize(QtCore.QSize(64, 64))
        self.radioButton_1.setChecked(True)
        self.radioButton_1.setObjectName("radioButton_1")
        self.verticalLayout_21.addWidget(self.radioButton_1, 0, QtCore.Qt.AlignmentFlag.AlignHCenter)
        self.horizontalLayout_27.addLayout(self.verticalLayout_21)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding,
                                            QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout_27.addItem(spacerItem3)
        self.verticalLayout_24 = QtWidgets.QVBoxLayout()
        self.verticalLayout_24.setObjectName("verticalLayout_24")
        self.radioButton_2 = QtWidgets.QRadioButton(parent=self.formatTab)
        self.radioButton_2.setStyleSheet(re.sub('_string', '_bmp', self.radioButton_string))
        self.radioButton_2.setText("")
        self.radioButton_2.setObjectName("radioButton_2")
        self.verticalLayout_24.addWidget(self.radioButton_2, 0, QtCore.Qt.AlignmentFlag.AlignHCenter)
        self.horizontalLayout_27.addLayout(self.verticalLayout_24)
        self.verticalLayout_25 = QtWidgets.QVBoxLayout()
        self.verticalLayout_25.setObjectName("verticalLayout_25")
        self.radioButton_3 = QtWidgets.QRadioButton(parent=self.formatTab)
        self.radioButton_3.setStyleSheet(re.sub('_string', '_jpg', self.radioButton_string))
        self.radioButton_3.setText("")
        self.radioButton_3.setObjectName("radioButton_3")
        self.verticalLayout_25.addWidget(self.radioButton_3, 0, QtCore.Qt.AlignmentFlag.AlignHCenter)
        self.horizontalLayout_27.addLayout(self.verticalLayout_25)
        self.verticalLayout_26 = QtWidgets.QVBoxLayout()
        self.verticalLayout_26.setObjectName("verticalLayout_26")
        self.radioButton_4 = QtWidgets.QRadioButton(parent=self.formatTab)
        self.radioButton_4.setStyleSheet(re.sub('_string', '_png', self.radioButton_string))
        self.radioButton_4.setText("")
        self.radioButton_4.setObjectName("radioButton_4")
        self.verticalLayout_26.addWidget(self.radioButton_4, 0, QtCore.Qt.AlignmentFlag.AlignHCenter)
        self.horizontalLayout_27.addLayout(self.verticalLayout_26)
        self.verticalLayout_27 = QtWidgets.QVBoxLayout()
        self.verticalLayout_27.setObjectName("verticalLayout_27")
        self.radioButton_5 = QtWidgets.QRadioButton(parent=self.formatTab)
        self.radioButton_5.setStyleSheet(re.sub('_string', '_tiff', self.radioButton_string))
        self.radioButton_5.setText("")
        self.radioButton_5.setObjectName("radioButton_5")
        self.verticalLayout_27.addWidget(self.radioButton_5, 0, QtCore.Qt.AlignmentFlag.AlignHCenter)
        self.horizontalLayout_27.addLayout(self.verticalLayout_27)
        self.verticalLayout_28 = QtWidgets.QVBoxLayout()
        self.verticalLayout_28.setObjectName("verticalLayout_28")
        self.radioButton_6 = QtWidgets.QRadioButton(parent=self.formatTab)
        self.radioButton_6.setStyleSheet(re.sub('_string', '_webp', self.radioButton_string))
        self.radioButton_6.setText("")
        self.radioButton_6.setObjectName("radioButton_6")
        self.verticalLayout_28.addWidget(self.radioButton_6, 0, QtCore.Qt.AlignmentFlag.AlignHCenter)
        self.horizontalLayout_27.addLayout(self.verticalLayout_28)
        icon20 = QtGui.QIcon()
        icon20.addPixmap(QtGui.QPixmap("resources/icons/memory_card.svg"), QtGui.QIcon.Mode.Normal,
                         QtGui.QIcon.State.Off)
        self.settings_tabWidget.addTab(self.formatTab, icon20, "")
        self.verticalLayout_2.addWidget(self.settings_tabWidget)
        self.verticalLayout_2.setStretch(0, 13)
        self.verticalLayout_2.setStretch(1, 3)
        self.horizontalLayout_2.addLayout(self.verticalLayout_2)
        self.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(parent=self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1348, 22))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(parent=self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuTools = QtWidgets.QMenu(parent=self.menubar)
        self.menuTools.setObjectName("menuTools")
        self.menuInfo = QtWidgets.QMenu(parent=self.menubar)
        self.menuInfo.setObjectName("menuInfo")
        self.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(parent=self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)
        self.actionAbout_Face_Cropper = QtGui.QAction(parent=self)
        self.actionAbout_Face_Cropper.setObjectName("actionAbout_Face_Cropper")
        self.actionUse_Mapping = QtGui.QAction(parent=self)
        self.actionUse_Mapping.setIcon(icon5)
        self.actionUse_Mapping.setObjectName("actionUse_Mapping")
        self.actionCrop_File = QtGui.QAction(parent=self)
        self.actionCrop_File.setIcon(icon1)
        self.actionCrop_File.setObjectName("actionCrop_File")
        self.actionCrop_Folder = QtGui.QAction(parent=self)
        self.actionCrop_Folder.setIcon(icon3)
        self.actionCrop_Folder.setObjectName("actionCrop_Folder")
        self.actionSquare = QtGui.QAction(parent=self)
        self.actionSquare.setObjectName("actionSquare")
        self.actionGolden_Ratio = QtGui.QAction(parent=self)
        self.actionGolden_Ratio.setObjectName("actionGolden_Ratio")
        self.action2_3_Ratio = QtGui.QAction(parent=self)
        self.action2_3_Ratio.setObjectName("action2_3_Ratio")
        self.action3_4_Ratio = QtGui.QAction(parent=self)
        self.action3_4_Ratio.setObjectName("action3_4_Ratio")
        self.action4_5_Ratio = QtGui.QAction(parent=self)
        self.action4_5_Ratio.setObjectName("action4_5_Ratio")
        self.actionCrop_Video = QtGui.QAction(parent=self)
        self.actionCrop_Video.setIcon(icon6)
        self.actionCrop_Video.setObjectName("actionCrop_Video")
        self.menuFile.addAction(self.actionSquare)
        self.menuFile.addAction(self.actionGolden_Ratio)
        self.menuFile.addAction(self.action2_3_Ratio)
        self.menuFile.addAction(self.action3_4_Ratio)
        self.menuFile.addAction(self.action4_5_Ratio)
        self.menuTools.addAction(self.actionUse_Mapping)
        self.menuTools.addAction(self.actionCrop_File)
        self.menuTools.addAction(self.actionCrop_Folder)
        self.menuTools.addAction(self.actionCrop_Video)
        self.menuInfo.addAction(self.actionAbout_Face_Cropper)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuTools.menuAction())
        self.menubar.addAction(self.menuInfo.menuAction())

        # CONNECTIONS
        self.actionAbout_Face_Cropper.triggered.connect(lambda: self.load_about_form())
        self.actionGolden_Ratio.triggered.connect(lambda: self.load_preset(0.5 * (1 + 5 ** 0.5)))
        self.action2_3_Ratio.triggered.connect(lambda: self.load_preset(1.5))
        self.action3_4_Ratio.triggered.connect(lambda: self.load_preset(4 / 3))
        self.action4_5_Ratio.triggered.connect(lambda: self.load_preset(1.25))
        self.actionSquare.triggered.connect(lambda: self.load_preset(1))

        self.actionCrop_File.triggered.connect(lambda: self.function_tabWidget.setCurrentIndex(0))
        self.actionCrop_Folder.triggered.connect(lambda: self.function_tabWidget.setCurrentIndex(1))
        self.actionUse_Mapping.triggered.connect(lambda: self.function_tabWidget.setCurrentIndex(2))
        self.actionCrop_Video.triggered.connect(lambda: self.function_tabWidget.setCurrentIndex(3))

        self.photoButton.clicked.connect(lambda: self.open_folder(self.photoLineEdit, self.photoWidget))
        self.folderButton_1.clicked.connect(lambda: self.open_folder(self.folderLineEdit_1, self.folderWidget,
                                                                     self.file_model))
        self.folderButton_2.clicked.connect(lambda: self.open_folder(self.folderLineEdit_2, self.mappingWidget))

        self.tableButton.clicked.connect(lambda: self.open_table())

        self.videoButton.clicked.connect(
            lambda: self.video.open_video(self, self.videoLineEdit, self.playButton, self.cropButton_4))

        self.destinationButton_1.clicked.connect(lambda: self.open_folder(self.destinationLineEdit_1))
        self.destinationButton_2.clicked.connect(lambda: self.open_folder(self.destinationLineEdit_2))
        self.destinationButton_3.clicked.connect(lambda: self.open_folder(self.destinationLineEdit_3))
        self.destinationButton_4.clicked.connect(lambda: self.open_folder(self.destinationLineEdit_4))

        self.cropButton_1.clicked.connect(
            lambda: self.cropper.crop(Path(self.photoLineEdit.text()), Path(self.destinationLineEdit_1.text()),
                               self.widthLineEdit, self.heightLineEdit, self.exposureCheckBox_1,
                               self.sensitivityDial, self.faceDial, self.gammaDial,
                               self.topDial, self.bottomDial, self.leftDial,
                               self.rightDial, self.radio_choices[self.radio][0], self.radio_choices))

        self.cropButton_2.clicked.connect(lambda: self.folder_process())
        self.cropButton_3.clicked.connect(lambda: self.mapping_process())
        self.cropButton_4.clicked.connect(
            lambda: self.cropper.crop_frame(self.videoLineEdit, self.destinationLineEdit_4, self.widthLineEdit,
                                            self.heightLineEdit, self.exposureCheckBox_4, self.sensitivityDial,
                                            self.faceDial, self.gammaDial, self.topDial, self.bottomDial, self.leftDial,
                                            self.rightDial, self.positionLabel, self.timelineSlider,
                                            self.radio_choices[self.radio][0], self.radio_choices))
        self.videocropButton.clicked.connect(lambda: self.video_process())

        self.cancelButton_1.clicked.connect(lambda: self.terminate(self.cropper))
        self.cancelButton_2.clicked.connect(lambda: self.terminate(self.cropper))
        self.cancelButton_3.clicked.connect(lambda: self.terminate(self.cropper))

        self.playButton.clicked.connect(lambda: self.video.play_video(self.playButton))
        self.playButton.clicked.connect(
            lambda: self.change_widget_state(True, self.stopButton, self.stepbackButton,  self.stepfwdButton,
                                             self.fastfwdButton, self.rewindButton, self.pushButton_26,
                                             self.pushButton_27, self.startmarkerButton, self.endmarkerButton,
                                             self.selectEndMarkerButton, self.selectStartMarkerButton))
        self.stopButton.clicked.connect(lambda: self.video.stop_btn())
        self.stepbackButton.clicked.connect(lambda: self.video.stepback())
        self.stepfwdButton.clicked.connect(lambda: self.video.stepfwd())
        self.fastfwdButton.clicked.connect(lambda: self.video.fastfwd())
        self.rewindButton.clicked.connect(lambda: self.video.rewind())
        self.pushButton_26.clicked.connect(lambda: self.video.goto_begining())
        self.pushButton_27.clicked.connect(lambda: self.video.goto_end())
        self.startmarkerButton.clicked.connect(
            lambda: self.video.set_startPosition(self.selectStartMarkerButton, self.timelineSlider))
        self.endmarkerButton.clicked.connect(
            lambda: self.video.set_stopPosition(self.selectEndMarkerButton, self.timelineSlider))
        self.selectStartMarkerButton.clicked.connect(lambda: self.video.goto(self.selectStartMarkerButton))
        self.selectEndMarkerButton.clicked.connect(lambda: self.video.goto(self.selectEndMarkerButton))
        self.muteButton.clicked.connect(lambda: self.video.volume_mute(self.volumeSlider, self.muteButton))

        self.connect_lineEdits_buttons(self.photoLineEdit, self.folderLineEdit_1, self.folderLineEdit_2,
                                       self.tableLineEdit, self.videoLineEdit, self.destinationLineEdit_1,
                                       self.destinationLineEdit_2, self.destinationLineEdit_3,
                                       self.destinationLineEdit_4, self.widthLineEdit, self.heightLineEdit)

        self.photoLineEdit.textChanged.connect(lambda: self.load_data(self.photoLineEdit, self.photoWidget))
        self.folderLineEdit_1.textChanged.connect(lambda: self.load_data(self.folderLineEdit_1, self.folderWidget))
        self.folderLineEdit_2.textChanged.connect(lambda: self.load_data(self.folderLineEdit_2, self.mappingWidget))

        self.connect_input_widgets(
            self.exposureCheckBox_1, self.exposureCheckBox_2, self.exposureCheckBox_3, self.exposureCheckBox_4,
            self.gammaDial, self.faceDial, self.sensitivityDial, self.topDial, self.bottomDial, self.leftDial,
            self.rightDial, self.widthLineEdit, self.heightLineEdit)

        self.treeView.selectionModel().selectionChanged.connect(lambda: self.reload_widgets())

        self.connect_radio_buttons(self.radioButton_1, self.radioButton_2, self.radioButton_3,
                                   self.radioButton_4, self.radioButton_5, self.radioButton_6,)

        # Folder start connection
        self.cropper.folder_started.connect(
            lambda: self.disable_widget(self.widthLineEdit, self.heightLineEdit, self.exposureCheckBox_2,
                                        self.sensitivityDial, self.faceDial, self.gammaDial, self.topDial,
                                        self.bottomDial, self.leftDial, self.rightDial, self.folderLineEdit_1,
                                        self.destinationLineEdit_2, self.destinationButton_2, self.folderButton_1,
                                        self.radioButton_1, self.radioButton_2, self.radioButton_3, self.radioButton_4,
                                        self.radioButton_5, self.radioButton_6, self.cropButton_2))
        self.cropper.folder_started.connect(lambda: self.enable_widget(self.cancelButton_1))

        # Maping start connection
        self.cropper.mapping_started.connect(
            lambda: self.disable_widget(self.widthLineEdit, self.heightLineEdit, self.exposureCheckBox_3,
                                        self.sensitivityDial, self.faceDial, self.gammaDial, self.topDial,
                                        self.bottomDial, self.leftDial, self.rightDial, self.folderLineEdit_2,
                                        self.destinationLineEdit_3, self.destinationButton_3, self.folderButton_2,
                                        self.tableLineEdit, self.comboBox, self.comboBox_2, self.radioButton_1,
                                        self.radioButton_2, self.radioButton_3, self.radioButton_4, self.radioButton_5,
                                        self.radioButton_6, self.cropButton_3))
        self.cropper.mapping_started.connect(lambda: self.enable_widget(self.cancelButton_2))

        # Video start connection
        self.cropper.video_started.connect(
            lambda: self.disable_widget(self.widthLineEdit, self.heightLineEdit, self.exposureCheckBox_4,
                                        self.sensitivityDial, self.faceDial, self.gammaDial, self.topDial,
                                        self.bottomDial, self.leftDial, self.rightDial, self.videoLineEdit,
                                        self.destinationLineEdit_4, self.destinationButton_4, self.radioButton_1,
                                        self.radioButton_2, self.radioButton_3, self.radioButton_4, self.radioButton_5,
                                        self.radioButton_6, self.cropButton_4, self.videocropButton))
        self.cropper.video_started.connect(lambda: self.enable_widget(self.cancelButton_3))

        # Folder end connection
        self.cropper.folder_finished.connect(
            lambda: self.enable_widget(self.widthLineEdit, self.heightLineEdit, self.exposureCheckBox_2,
                                       self.sensitivityDial, self.faceDial, self.gammaDial, self.topDial,
                                       self.bottomDial, self.leftDial, self.rightDial, self.folderLineEdit_1,
                                       self.destinationLineEdit_2, self.destinationButton_2, self.folderButton_1,
                                       self.radioButton_1, self.radioButton_2, self.radioButton_3, self.radioButton_4,
                                       self.radioButton_5, self.radioButton_6, self.cropButton_2))
        self.cropper.folder_finished.connect(lambda: self.disable_widget(self.cancelButton_1))
        self.cropper.folder_finished.connect(lambda: self.show_message_box(self.destinationLineEdit_2))

        # Maping end connection
        self.cropper.mapping_finished.connect(
            lambda: self.enable_widget(self.widthLineEdit, self.heightLineEdit, self.exposureCheckBox_3,
                                       self.sensitivityDial, self.faceDial, self.gammaDial, self.topDial,
                                       self.bottomDial, self.leftDial, self.rightDial, self.folderLineEdit_2,
                                       self.destinationLineEdit_3, self.destinationButton_3, self.folderButton_2,
                                       self.tableLineEdit, self.comboBox, self.comboBox_2, self.radioButton_1,
                                       self.radioButton_2, self.radioButton_3, self.radioButton_4, self.radioButton_5,
                                       self.radioButton_6, self.cropButton_3))
        self.cropper.mapping_finished.connect(lambda: self.disable_widget(self.cancelButton_2))
        self.cropper.mapping_finished.connect(lambda: self.show_message_box(self.destinationLineEdit_3))

        # Video end connection
        self.cropper.video_finished.connect(
            lambda: self.enable_widget(self.widthLineEdit, self.heightLineEdit, self.exposureCheckBox_4,
                                       self.sensitivityDial, self.faceDial, self.gammaDial, self.topDial,
                                       self.bottomDial, self.leftDial, self.rightDial, self.videoLineEdit,
                                       self.destinationLineEdit_4, self.destinationButton_4, self.radioButton_1,
                                       self.radioButton_2, self.radioButton_3, self.radioButton_4, self.radioButton_5,
                                       self.radioButton_6, self.cropButton_4, self.videocropButton))
        self.cropper.video_finished.connect(lambda: self.disable_widget(self.cancelButton_3))
        self.cropper.video_finished.connect(lambda: self.show_message_box(self.destinationLineEdit_4))

        self.cropper.folder_progress.connect(self.update_progress_1)
        self.cropper.mapping_progress.connect(self.update_progress_2)
        self.cropper.video_progress.connect(self.update_progress_3)

        self.radio_choices = np.array(['No', '.bmp', '.jpg', '.png', '.tiff', '.webp'])
        self.radio_buttons = [self.radioButton_1, self.radioButton_2, self.radioButton_3,
                              self.radioButton_4, self.radioButton_5, self.radioButton_6]
        self.radio = np.array([r.isChecked() for r in self.radio_buttons])

        self.retranslateUi()
        self.disable_buttons()
        self.change_widget_state(False, self.cropButton_1, self.cropButton_2, self.cropButton_3, self.cropButton_4,
                                 self.videocropButton, self.cancelButton_1, self.cancelButton_2, self.cancelButton_3,
                                 self.playButton, self.stopButton, self.stepbackButton, self.stepfwdButton,
                                 self.rewindButton, self.fastfwdButton, self.pushButton_26, self.pushButton_27,
                                 self.startmarkerButton, self.endmarkerButton, self.selectStartMarkerButton,
                                 self.selectEndMarkerButton, self.timelineSlider)
        self.function_tabWidget.setCurrentIndex(0)
        self.settings_tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(self)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.photoLineEdit.setPlaceholderText(_translate("MainWindow", "Choose the image you want to crop"))
        self.photoButton.setText(_translate("MainWindow", "PushButton"))
        self.exposureCheckBox_1.setText(_translate("MainWindow", "Autocorrect Exposure"))
        self.destinationLineEdit_1.setPlaceholderText(
            _translate("MainWindow", "Choose where you want to save the cropped image"))
        self.destinationButton_1.setText(_translate("MainWindow", "Destination Folder"))
        self.function_tabWidget.setTabText(
            self.function_tabWidget.indexOf(self.photoTab), _translate("MainWindow", "Photo Crop"))
        self.folderLineEdit_1.setPlaceholderText(_translate("MainWindow", "Choose the folder you want to crop"))
        self.folderButton_1.setText(_translate("MainWindow", "Select Folder"))
        self.exposureCheckBox_2.setText(_translate("MainWindow", "Autocorrect Exposure"))
        self.destinationLineEdit_2.setPlaceholderText(
            _translate("MainWindow", "Choose where you want to save the cropped images"))
        self.destinationButton_2.setText(_translate("MainWindow", "Destination Folder"))
        self.function_tabWidget.setTabText(
            self.function_tabWidget.indexOf(self.folder_Tab), _translate("MainWindow", "Folder Crop"))
        self.tableLineEdit.setPlaceholderText(_translate("MainWindow", "Choose the Excel or CSV file with the mapping"))
        self.folderLineEdit_2.setPlaceholderText(_translate("MainWindow", "Choose the folder you want to crop"))
        self.folderButton_2.setText(_translate("MainWindow", "Select Folder"))
        self.tableButton.setText(_translate("MainWindow", "Open File"))
        self.exposureCheckBox_3.setText(_translate("MainWindow", "Autocorrect Exposure"))
        self.comboBox.setPlaceholderText(_translate("MainWindow", "Filename column"))
        self.comboBox_2.setPlaceholderText(_translate("MainWindow", "Mapping column"))
        self.destinationLineEdit_3.setPlaceholderText(
            _translate("MainWindow", "Choose where you want to save the cropped images"))
        self.destinationButton_3.setText(_translate("MainWindow", "Destination Folder"))
        self.function_tabWidget.setTabText(
            self.function_tabWidget.indexOf(self.mappingTab), _translate("MainWindow", "Mapping Crop"))
        self.videoLineEdit.setPlaceholderText(_translate("MainWindow", "Choose the video you want to crop"))
        self.videoButton.setText(_translate("MainWindow", "Open Video"))
        self.positionLabel.setText(_translate("MainWindow", "00:00:00"))
        self.durationLabel.setText(_translate("MainWindow", "00:00:00"))
        self.exposureCheckBox_4.setText(_translate("MainWindow", "Autocorrect Exposure"))
        self.destinationLineEdit_4.setPlaceholderText(
            _translate("MainWindow", "Choose where you want to save the cropped images"))
        self.destinationButton_4.setText(_translate("MainWindow", "Destination Folder"))
        self.selectStartMarkerButton.setText(_translate("MainWindow", "00:00:00"))
        self.selectEndMarkerButton.setText(_translate("MainWindow", "00:00:00"))
        self.function_tabWidget.setTabText(
            self.function_tabWidget.indexOf(self.videoTab), _translate("MainWindow", "Video Crop"))
        self.label.setText(_translate("MainWindow", "Gamma"))
        self.label_2.setText(_translate("MainWindow", "Face %"))
        self.label_3.setText(_translate("MainWindow", "Sensitivity"))
        self.label_4.setText(_translate("MainWindow", "Width (px)"))
        self.widthLineEdit.setPlaceholderText(_translate("MainWindow", "400"))
        self.label_5.setText(_translate("MainWindow", "Height (px)"))
        self.heightLineEdit.setPlaceholderText(_translate("MainWindow", "500"))
        self.label_7.setText(_translate("MainWindow", "Padding"))
        self.label_8.setText(_translate("MainWindow", "Top"))
        self.label_9.setText(_translate("MainWindow", "Bottom"))
        self.label_10.setText(_translate("MainWindow", "Left"))
        self.label_11.setText(_translate("MainWindow", "Right"))
        self.settings_tabWidget.setTabText(
            self.settings_tabWidget.indexOf(self.settingsTab), _translate("MainWindow", "Settings"))
        self.settings_tabWidget.setTabText(
            self.settings_tabWidget.indexOf(self.formatTab), _translate("MainWindow", "Format Conversion"))
        self.menuFile.setTitle(_translate("MainWindow", "Presets"))
        self.menuTools.setTitle(_translate("MainWindow", "Tools"))
        self.menuInfo.setTitle(_translate("MainWindow", "Info"))
        self.actionAbout_Face_Cropper.setText(_translate("MainWindow", "About Face Cropper"))
        self.actionUse_Mapping.setText(_translate("MainWindow", "Use Mapping"))
        self.actionCrop_File.setText(_translate("MainWindow", "Crop File"))
        self.actionCrop_Folder.setText(_translate("MainWindow", "Crop Folder"))
        self.actionSquare.setText(_translate("MainWindow", "Square"))
        self.actionGolden_Ratio.setText(_translate("MainWindow", "Golden Ratio"))
        self.action2_3_Ratio.setText(_translate("MainWindow", "2:3 Ratio"))
        self.action3_4_Ratio.setText(_translate("MainWindow", "3:4 Ratio"))
        self.action4_5_Ratio.setText(_translate("MainWindow", "4:5 Ratio"))
        self.actionCrop_Video.setText(_translate("MainWindow", "Crop Video"))

    def connect_radio_buttons(self, *buttons: QtWidgets.QRadioButton):
        for radioButton in buttons:
            radioButton.clicked.connect(self.radio_logic)

    def connect_lineEdits_buttons(self, *lineEdits: QtWidgets.QLineEdit):
        for lineEdit in lineEdits:
            lineEdit.textChanged.connect(lambda: self.disable_buttons())

    def connect_input_widgets(self, *input_widgets: QtWidgets.QWidget):
        for input_widget in input_widgets:
            if isinstance(input_widget, QtWidgets.QLineEdit):
                input_widget.textChanged.connect(lambda: self.reload_widgets())
            elif isinstance(input_widget, QtWidgets.QDial):
                input_widget.valueChanged.connect(lambda: self.reload_widgets())
            elif isinstance(input_widget, QtWidgets.QCheckBox):
                input_widget.stateChanged.connect(lambda: self.reload_widgets())

    def folder_image_select(self) -> None:
        if not self.widthLineEdit.text() or not self.heightLineEdit.text():
            return None
        
        if not Path(self.folderLineEdit_1.text()).as_posix():
            return None
        self.cropper.display_crop(Path(self.file_model.filePath(self.treeView.currentIndex())), self.exposureCheckBox_2,
                                  self.widthLineEdit, self.heightLineEdit, self.sensitivityDial, self.faceDial,
                                  self.gammaDial, self.topDial, self.bottomDial, self.leftDial, self.rightDial,
                                  self.folderWidget)

    def reload_widgets(self) -> None:
        common_widgets = (self.widthLineEdit, self.heightLineEdit, self.sensitivityDial, self.faceDial, 
                          self.gammaDial, self.topDial, self.bottomDial, self.leftDial, self.rightDial)
        if not self.widthLineEdit.text() or not self.heightLineEdit.text():
            return None
        if self.function_tabWidget.currentIndex() == 0:
            f_name = Path(self.photoLineEdit.text())
            if not f_name.as_posix():
                return None
            self.cropper.display_crop(f_name, self.exposureCheckBox_1, *common_widgets, self.photoWidget)
        elif self.function_tabWidget.currentIndex() == 1:
            if self.treeView.currentIndex().isValid():
                folder = Path(self.file_model.filePath(self.treeView.currentIndex()))
            else:
                folder = Path(self.folderLineEdit_1.text())

            if not folder.as_posix():
                return None
            self.cropper.display_crop(folder, self.exposureCheckBox_2, *common_widgets, self.folderWidget,
                                      IMAGE_TYPES)
        elif self.function_tabWidget.currentIndex() == 2:
            folder = Path(self.folderLineEdit_2.text())
            if not folder.as_posix():
                return None
            self.cropper.display_crop(folder, self.exposureCheckBox_3, *common_widgets, self.mappingWidget,
                                      IMAGE_TYPES)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if not event.mimeData().hasUrls():
            event.ignore()
            return

        event.setDropAction(QtCore.Qt.DropAction.CopyAction)
        file_path = Path(event.mimeData().urls()[0].toLocalFile())
        common_widgets = (self.widthLineEdit, self.heightLineEdit, self.sensitivityDial, self.faceDial,
                          self.gammaDial, self.topDial, self.bottomDial, self.leftDial, self.rightDial)

        if file_path.is_dir():
            self.handle_directory(file_path, common_widgets)
        elif file_path.is_file():
            self.handle_file(file_path, common_widgets)

        event.accept()

    def handle_directory(self, file_path, common_widgets):
        self.function_tabWidget.setCurrentIndex(1)
        self.folderLineEdit_1.setText(file_path.as_posix())
        if self.widthLineEdit.text() and self.heightLineEdit.text():
            self.cropper.display_crop(file_path, self.exposureCheckBox_2, *common_widgets, self.folderWidget)

    def handle_file(self, file_path, common_widgets):
        if file_path.suffix in IMAGE_TYPES:
            self.handle_image_file(file_path, common_widgets)
        elif file_path.suffix in VIDEO_TYPES:
            self.handle_video_file(file_path)
        elif file_path.suffix in PANDAS_TYPES:
            self.handle_pandas_file(file_path)

    def handle_image_file(self, file_path, common_widgets):
        self.function_tabWidget.setCurrentIndex(0)
        self.photoLineEdit.setText(file_path.as_posix())
        if self.widthLineEdit.text() and self.heightLineEdit.text():
            self.cropper.display_crop(file_path, self.exposureCheckBox_1, *common_widgets, self.photoWidget)

    def handle_video_file(self, file_path):
        self.function_tabWidget.setCurrentIndex(3)
        self.videoLineEdit.setText(file_path.as_posix())

    def handle_pandas_file(self, file_path):
        self.function_tabWidget.setCurrentIndex(2)
        self.tableLineEdit.setText(file_path.as_posix())
        data = utils.open_file(file_path)
        assert isinstance(data, pd.DataFrame)

        self.data_frame = data
        if self.data_frame is None:
            return

        self.model = custom_widgets.DataFrameModel(self.data_frame)
        self.tableView.setModel(self.model)

        self.comboBox.addItems(self.data_frame.columns.to_numpy())
        self.comboBox_2.addItems(self.data_frame.columns.to_numpy())


    def load_preset(self, phi: Union[int, float]) -> None:
        if phi == 1:
            if int(self.widthLineEdit.text()) > int(self.heightLineEdit.text()):
                self.heightLineEdit.setText(self.widthLineEdit.text())
            elif int(self.widthLineEdit.text()) < int(self.heightLineEdit.text()):
                self.widthLineEdit.setText(self.heightLineEdit.text())
        elif int(self.widthLineEdit.text()) >= int(self.heightLineEdit.text()):
            self.heightLineEdit.setText(str(int(float(self.widthLineEdit.text()) * phi)))
        elif int(self.widthLineEdit.text()) < int(self.heightLineEdit.text()):
            self.widthLineEdit.setText(str(int(float(self.heightLineEdit.text()) / phi)))

    def load_data(self, line_edit: Union[QtWidgets.QLineEdit, str], image_widget: custom_widgets.ImageWidget) -> None:
        common_widgets = (self.widthLineEdit, self.heightLineEdit, self.sensitivityDial, self.faceDial,
                          self.gammaDial, self.topDial, self.bottomDial, self.leftDial, self.rightDial,
                          image_widget)
        f_name = line_edit.text() if isinstance(line_edit, QtWidgets.QLineEdit) else line_edit
        path = Path(f_name)
        try:
            if line_edit is self.photoLineEdit:
                self.cropper.display_crop(path, self.exposureCheckBox_1, *common_widgets)
            elif line_edit is self.folderLineEdit_1:
                self.file_model.setRootPath(f_name)
                self.treeView.setRootIndex(self.file_model.index(f_name))
                self.cropper.display_crop(path, self.exposureCheckBox_2, *common_widgets, IMAGE_TYPES)
            elif line_edit is self.folderLineEdit_2:
                self.cropper.display_crop(path, self.exposureCheckBox_3, *common_widgets, IMAGE_TYPES)
        except (IndexError, FileNotFoundError, ValueError, AttributeError):
            return None

    def open_folder(self, line_edit: QtWidgets.QLineEdit, image_widget: Optional[custom_widgets.ImageWidget] = None,
                    file_model: Optional[QtGui.QFileSystemModel] = None) -> None:

        if line_edit in {self.folderLineEdit_1, self.folderLineEdit_2, self.destinationLineEdit_1, 
                         self.destinationLineEdit_2, self.destinationLineEdit_3, self.destinationLineEdit_4}:
            f_name = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Directory', Photo().default_directory)
            line_edit.setText(f_name)

            if image_widget is None:
                return

            if isinstance(file_model, QtGui.QFileSystemModel):
                self.load_data(line_edit, image_widget)

        if line_edit is self.photoLineEdit:
            f_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File', Photo().default_directory,
                                                              Photo().type_string())
            line_edit.setText(f_name)
            if image_widget is None:
                return
            self.load_data(line_edit, image_widget)

    def open_table(self) -> None:
        type_string = "All Files (*);;" + ";;".join(f"{_} Files (*{_})" for _ in np.sort(utils.PANDAS_TYPES))
        f_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File', Photo().default_directory, type_string)
        if f_name is None:
            return None
        self.tableLineEdit.setText(f_name)

        data = utils.open_file(f_name)
        try:
            assert isinstance(data, pd.DataFrame)
        except AssertionError:
            return None

        self.data_frame = data
        if self.data_frame is None:
            return None
        self.model = custom_widgets.DataFrameModel(self.data_frame)
        self.tableView.setModel(self.model)

        self.comboBox.addItems(self.data_frame.columns.to_numpy())
        self.comboBox_2.addItems(self.data_frame.columns.to_numpy())

    @staticmethod
    def load_about_form() -> None:
        about_ui = custom_widgets.UiDialog()
        about_ui.exec()

    def radio_logic(self) -> None:
        self.radio = np.array([r.isChecked() for r in self.radio_buttons])

    @staticmethod
    def show_message_box(destination: QtWidgets.QLineEdit):
        def message_button(answer):
            if answer.text() == '&Yes':
                startfile(destination.text())

        def helper_function(msg_box: QtWidgets.QMessageBox):
            msg_box.setWindowTitle('Open Destination Folder')
            msg_box.setText('Open destination folder?')
            msg_box.setIcon(QtWidgets.QMessageBox.Icon.Question)
            msg_box.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No)
            msg_box.buttonClicked.connect(message_button)
            x = msg_box.exec()

        msg = QtWidgets.QMessageBox()
        helper_function(msg)

    @staticmethod
    def disable_widget(*args: QtWidgets.QWidget) -> None:
        for arg in args:
            arg.setDisabled(True)

    @staticmethod
    def enable_widget(*args: QtWidgets.QWidget) -> None:
        for arg in args:
            arg.setEnabled(True)

    @staticmethod
    def change_widget_state(boolean: bool, *args: QtWidgets.QWidget) -> None:
        for arg in args:
            if boolean:
                arg.setEnabled(boolean)
            else:
                arg.setDisabled(not boolean)

    @staticmethod
    def terminate(cropper: Cropper):
        cropper.end_task = True

    def disable_buttons(self) -> None:
        def all_filled(*line_edits: Union[QtWidgets.QLineEdit, QtWidgets.QComboBox]) -> bool:
            x = all(edit.text() for edit in line_edits if isinstance(edit, QtWidgets.QLineEdit))
            y = all(edit.currentText() for edit in line_edits if isinstance(edit, QtWidgets.QComboBox))
            return x and y

        def update_widget_state(condition: bool, *widgets) -> None:
            for widget in widgets:
                self.change_widget_state(condition, widget)

        common_line_edits = (self.widthLineEdit, self.heightLineEdit)

        # Photo logic
        update_widget_state(
            all_filled(self.photoLineEdit, self.destinationLineEdit_1, *common_line_edits), self.cropButton_1)
        # Folder logic
        update_widget_state(
            all_filled(self.folderLineEdit_1, self.destinationLineEdit_2, *common_line_edits), self.cropButton_2)
        # Mapping logic
        update_widget_state(
            all_filled(self.folderLineEdit_2, self.tableLineEdit, self.destinationLineEdit_3, self.comboBox,
                       self.comboBox_2, *common_line_edits), self.cropButton_3)
        # Video logic
        update_widget_state(
            all_filled(self.videoLineEdit, self.destinationLineEdit_4, *common_line_edits), self.cropButton_4,
            self.videocropButton)

    def folder_process(self) -> None:
        self.cropper.reset()
        file_list = np.fromiter(Path(self.folderLineEdit_1.text()).iterdir(), Path)
        photo_files = np.array([pic.suffix.lower() in IMAGE_TYPES for pic in file_list])
        process = Process(target=self.cropper.crop_dir, daemon=True,
                          args=(file_list[photo_files], Path(self.destinationLineEdit_2.text()), self.widthLineEdit,
                                self.heightLineEdit, self.exposureCheckBox_2, self.sensitivityDial, self.faceDial,
                                self.gammaDial, self.topDial, self.bottomDial, self.leftDial, self.rightDial,
                                self.radio_choices[self.radio][0], self.folderLineEdit_1, self.radio_choices))
        process.run()

    def mapping_process(self) -> None:
        self.cropper.reset()
        process = Process(target=self.cropper.mapping_crop, daemon=True,
                          args=(Path(self.folderLineEdit_2.text()), self.data_frame, self.comboBox, self.comboBox_2,
                                Path(self.destinationLineEdit_3.text()), self.widthLineEdit, self.heightLineEdit,
                                self.exposureCheckBox_3, self.sensitivityDial, self.faceDial, self.gammaDial,
                                self.topDial, self.bottomDial, self.leftDial, self.rightDial,
                                self.radio_choices[self.radio][0], self.radio_choices))
        process.run()

    def video_process(self) -> None:
        self.cropper.reset()
        process = Process(target=self.cropper.extract_frames, daemon=True,
                          args=(self.videoLineEdit, self.video.start_position, self.video.stop_position,
                                self.destinationLineEdit_4, self.widthLineEdit, self.heightLineEdit,
                                self.exposureCheckBox_4, self.sensitivityDial, self.faceDial, self.gammaDial,
                                self.topDial, self.bottomDial, self.leftDial, self.rightDial,
                                self.radio_choices[self.radio][0], self.radio_choices))
        process.run()

    def update_progress_1(self, value: int) -> None:
        self.progressBar_1.setValue(value)
        QtWidgets.QApplication.processEvents()

    def update_progress_2(self, value: int) -> None:
        self.progressBar_2.setValue(value)
        QtWidgets.QApplication.processEvents()

    def update_progress_3(self, value: int) -> None:
        self.progressBar_3.setValue(value)
        QtWidgets.QApplication.processEvents()
