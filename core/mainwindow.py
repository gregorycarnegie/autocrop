from multiprocessing import Process
from pathlib import Path
from typing import Callable, Optional, Union, Tuple, Dict

import numpy as np
import pandas as pd
from PyQt6 import QtCore, QtGui, QtWidgets, QtMultimedia, QtMultimediaWidgets

from .cropper import Cropper
from .custom_widgets import DataFrameModel, ImageWidget, PathLineEdit, NumberLineEdit
from .file_types import Photo, Video, IMAGE_TYPES, VIDEO_TYPES, PANDAS_TYPES
from .job import Job
from .utils import open_file
from .window_functions import setup_frame, setup_progress_bar, setup_dial, setup_lcd, setup_radio_button, \
    setup_dial_area, setup_combo, enable_widget, disable_widget, change_widget_state, uncheck_boxes, terminate, \
    load_about_form, show_message_box


class UiMainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super(UiMainWindow, self).__init__()
        self.setAcceptDrops(True)
        self.model: Optional[DataFrameModel] = None
        self.data_frame: Optional[pd.DataFrame] = None
        self.validator = QtGui.QIntValidator(100, 10_000)
        self.file_model = QtGui.QFileSystemModel(self)
        self.file_model.setFilter(QtCore.QDir.Filter.NoDotAndDotDot | QtCore.QDir.Filter.Files)
        self.file_model.setNameFilters(Photo().file_filter())
        self.cropper = Cropper()
        self.player = QtMultimedia.QMediaPlayer()
        self.audio = QtMultimedia.QAudioOutput()
        self.setObjectName("MainWindow")
        self.resize(1_348, 896)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("resources/logos/logo.ico"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
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
        self.photoLineEdit = self.setup_line_edit(self.photoTab, "photoLineEdit", self.horizontalLayout_5, path_type='image')
        self.photoButton, icon1 = self.setup_button_icon(self.photoTab, 'picture', 1, self.horizontalLayout_5)
        self.verticalLayout_8.addLayout(self.horizontalLayout_5)
        self.frame_8 = setup_frame(self.photoTab, "frame_8")
        self.verticalLayout_20 = QtWidgets.QVBoxLayout(self.frame_8)
        self.verticalLayout_20.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_20.setSpacing(0)
        self.verticalLayout_20.setObjectName("verticalLayout_20")
        self.frame = setup_frame(self.photoTab, "frame", set_size=True)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout(self.frame)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.mfaceCheckBox_1, self.tiltCheckBox_1, self.exposureCheckBox_1 = self.setup_checkboxes(self.frame, self.horizontalLayout_8, 1)
        self.verticalLayout_20.addWidget(self.frame)
        self.frame_13 = setup_frame(self.photoTab, "frame_13")
        self.verticalLayout_29 = QtWidgets.QVBoxLayout(self.frame_13)
        self.verticalLayout_29.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_29.setSpacing(0)
        self.verticalLayout_29.setObjectName("verticalLayout_29")
        self.photoWidget = ImageWidget(parent=self.frame_13)
        self.photoWidget.setStyleSheet("")
        self.photoWidget.setObjectName("photoWidget")
        self.verticalLayout_29.addWidget(self.photoWidget)
        self.verticalLayout_20.addWidget(self.frame_13)
        self.frame_2 = setup_frame(self.photoTab, "frame_2", set_size=True)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(self.frame_2)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.cropButton_1 = self.setup_button(self.frame_2, 'crop', 1, self.horizontalLayout_7)
        self.verticalLayout_20.addWidget(self.frame_2)
        self.verticalLayout_20.setStretch(2, 1)
        self.verticalLayout_8.addWidget(self.frame_8)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.destinationLineEdit_1 = self.setup_line_edit(self.photoTab, "destinationLineEdit_1", self.horizontalLayout_6, path_type='folder')
        self.destinationButton_1, icon3 = self.setup_button_icon(self.photoTab, 'destination', 1, self.horizontalLayout_6)
        self.verticalLayout_8.addLayout(self.horizontalLayout_6)
        self.verticalLayout_8.setStretch(0, 1)
        self.verticalLayout_8.setStretch(1, 17)
        self.verticalLayout_8.setStretch(2, 1)
        self.function_tabWidget.addTab(self.photoTab, icon1, "")
        self.folder_Tab = QtWidgets.QWidget()
        self.folder_Tab.setObjectName("folder_Tab")
        self.verticalLayout_15 = QtWidgets.QVBoxLayout(self.folder_Tab)
        self.verticalLayout_15.setObjectName("verticalLayout_15")
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.folderLineEdit_1 = self.setup_line_edit(self.folder_Tab, "folderLineEdit_1", self.horizontalLayout_11, path_type='folder')
        self.folderButton_1 = self.setup_button(self.folder_Tab, 'folder', 1, self.horizontalLayout_11)
        self.verticalLayout_15.addLayout(self.horizontalLayout_11)
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        self.verticalLayout_16 = QtWidgets.QVBoxLayout()
        self.verticalLayout_16.setSpacing(0)
        self.verticalLayout_16.setObjectName("verticalLayout_16")
        self.frame_3 = setup_frame(self.folder_Tab, "frame_3", set_size=True)
        self.horizontalLayout_15 = QtWidgets.QHBoxLayout(self.frame_3)
        self.horizontalLayout_15.setObjectName("horizontalLayout_15")
        self.mfaceCheckBox_2, self.tiltCheckBox_2, self.exposureCheckBox_2 = self.setup_checkboxes(self.frame_3, self.horizontalLayout_15, 2)
        self.verticalLayout_16.addWidget(self.frame_3)
        self.frame_12 = setup_frame(self.folder_Tab, "frame_12")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.frame_12)
        self.verticalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_7.setSpacing(0)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.folderWidget = ImageWidget(parent=self.frame_12)
        self.folderWidget.setStyleSheet("")
        self.folderWidget.setObjectName("folderWidget")
        self.verticalLayout_7.addWidget(self.folderWidget)
        self.verticalLayout_16.addWidget(self.frame_12)
        self.frame_14 = setup_frame(self.folder_Tab, "frame_14")
        self.horizontalLayout_16 = QtWidgets.QHBoxLayout(self.frame_14)
        self.horizontalLayout_16.setContentsMargins(-1, 9, -1, 0)
        self.horizontalLayout_16.setObjectName("horizontalLayout_16")
        self.cropButton_2 = self.setup_button(self.frame_14, 'crop', 2, self.horizontalLayout_16)
        self.cancelButton_1 = self.setup_button(self.frame_14, 'cancel', 1, self.horizontalLayout_16)
        self.verticalLayout_16.addWidget(self.frame_14)
        self.frame_4 = setup_frame(self.folder_Tab, "frame_4", set_size=True)
        self.verticalLayout_30 = QtWidgets.QVBoxLayout(self.frame_4)
        self.verticalLayout_30.setContentsMargins(-1, 9, -1, -1)
        self.verticalLayout_30.setObjectName("verticalLayout_30")
        self.progressBar_1 = setup_progress_bar(self.frame_4, "progressBar_1", self.verticalLayout_30)
        self.verticalLayout_16.addWidget(self.frame_4)
        self.verticalLayout_16.setStretch(0, 1)
        self.verticalLayout_16.setStretch(1, 10)
        self.verticalLayout_16.setStretch(2, 1)
        self.verticalLayout_16.setStretch(3, 1)
        self.horizontalLayout_12.addLayout(self.verticalLayout_16)
        self.treeView = QtWidgets.QTreeView(parent=self.folder_Tab)
        self.treeView.setObjectName("treeView")
        self.treeView.setModel(self.file_model)
        self.horizontalLayout_12.addWidget(self.treeView)
        self.horizontalLayout_12.setStretch(0, 4)
        self.horizontalLayout_12.setStretch(1, 3)
        self.verticalLayout_15.addLayout(self.horizontalLayout_12)
        self.horizontalLayout_14 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_14.setObjectName("horizontalLayout_14")
        self.destinationLineEdit_2 = self.setup_line_edit(self.folder_Tab, "destinationLineEdit_2", self.horizontalLayout_14, path_type='folder')
        self.destinationButton_2 = self.setup_button(self.folder_Tab, 'destination', 2, self.horizontalLayout_14)
        self.verticalLayout_15.addLayout(self.horizontalLayout_14)
        self.verticalLayout_15.setStretch(0, 1)
        self.verticalLayout_15.setStretch(1, 17)
        self.verticalLayout_15.setStretch(2, 1)
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
        self.tableLineEdit = self.setup_line_edit(self.mappingTab, "tableLineEdit", path_type='table')
        self.gridLayout.addWidget(self.tableLineEdit, 1, 0, 1, 1)
        self.folderLineEdit_2 = self.setup_line_edit(self.mappingTab, "folderLineEdit_2", path_type='folder')
        self.gridLayout.addWidget(self.folderLineEdit_2, 0, 0, 1, 1)
        self.folderButton_2 = self.setup_button(self.mappingTab, 'folder', 2)
        self.gridLayout.addWidget(self.folderButton_2, 0, 1, 1, 1)
        self.tableButton, icon5 = self.setup_button_icon(self.mappingTab, 'data', 2)
        self.gridLayout.addWidget(self.tableButton, 1, 1, 1, 1)
        self.horizontalLayout_17.addLayout(self.gridLayout)
        self.verticalLayout_17.addLayout(self.horizontalLayout_17)
        self.horizontalLayout_18 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_18.setObjectName("horizontalLayout_18")
        self.verticalLayout_18 = QtWidgets.QVBoxLayout()
        self.verticalLayout_18.setSpacing(0)
        self.verticalLayout_18.setObjectName("verticalLayout_18")
        self.frame_5 = setup_frame(self.mappingTab, "frame_5")
        self.horizontalLayout_23 = QtWidgets.QHBoxLayout(self.frame_5)
        self.horizontalLayout_23.setObjectName("horizontalLayout_23")
        self.mfaceCheckBox_3, self.tiltCheckBox_3, self.exposureCheckBox_3 = self.setup_checkboxes(self.frame_5, self.horizontalLayout_23, 3)
        self.verticalLayout_18.addWidget(self.frame_5)
        self.frame_6 = setup_frame(self.mappingTab, "frame_6")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.frame_6)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.mappingWidget = ImageWidget(parent=self.frame_6)
        self.mappingWidget.setStyleSheet("")
        self.mappingWidget.setObjectName("mappingWidget")
        self.verticalLayout.addWidget(self.mappingWidget)
        self.verticalLayout_18.addWidget(self.frame_6)
        self.frame_15 = setup_frame(self.mappingTab, "frame_15")
        self.horizontalLayout_13 = QtWidgets.QHBoxLayout(self.frame_15)
        self.horizontalLayout_13.setContentsMargins(-1, 9, -1, 0)
        self.horizontalLayout_13.setObjectName("horizontalLayout_13")
        self.cropButton_3 = self.setup_button(self.frame_15, 'crop', 3, self.horizontalLayout_13)
        self.cancelButton_2 = self.setup_button(self.frame_15, 'cancel', 2, self.horizontalLayout_13)
        self.verticalLayout_18.addWidget(self.frame_15)
        self.frame_7 = setup_frame(self.mappingTab, "frame_7")
        self.horizontalLayout_21 = QtWidgets.QHBoxLayout(self.frame_7)
        self.horizontalLayout_21.setContentsMargins(-1, 9, -1, -1)
        self.horizontalLayout_21.setObjectName("horizontalLayout_21")
        self.progressBar_2 = setup_progress_bar(self.frame_7, "progressBar_2", self.horizontalLayout_21)
        self.verticalLayout_18.addWidget(self.frame_7)
        self.verticalLayout_18.setStretch(0, 1)
        self.verticalLayout_18.setStretch(1, 10)
        self.verticalLayout_18.setStretch(2, 1)
        self.verticalLayout_18.setStretch(3, 1)
        self.horizontalLayout_18.addLayout(self.verticalLayout_18)
        self.verticalLayout_19 = QtWidgets.QVBoxLayout()
        self.verticalLayout_19.setObjectName("verticalLayout_19")
        self.tableView = QtWidgets.QTableView(parent=self.mappingTab)
        self.tableView.setObjectName("tableView")
        self.verticalLayout_19.addWidget(self.tableView)
        self.horizontalLayout_24 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_24.setObjectName("horizontalLayout_24")
        self.comboBox_1 = setup_combo(self.mappingTab, 'comboBox_1', self.horizontalLayout_24)
        self.comboBox_2 = setup_combo(self.mappingTab, 'comboBox_2', self.horizontalLayout_24)
        self.verticalLayout_19.addLayout(self.horizontalLayout_24)
        self.horizontalLayout_18.addLayout(self.verticalLayout_19)
        self.horizontalLayout_18.setStretch(0, 1)
        self.horizontalLayout_18.setStretch(1, 2)
        self.verticalLayout_17.addLayout(self.horizontalLayout_18)
        self.horizontalLayout_20 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_20.setObjectName("horizontalLayout_20")
        self.destinationLineEdit_3 = self.setup_line_edit(self.mappingTab, "destinationLineEdit_3", self.horizontalLayout_20, path_type='folder')
        self.destinationButton_3 = self.setup_button(self.mappingTab, 'destination', 3, self.horizontalLayout_20)
        self.verticalLayout_17.addLayout(self.horizontalLayout_20)
        self.verticalLayout_17.setStretch(0, 2)
        self.verticalLayout_17.setStretch(1, 15)
        self.verticalLayout_17.setStretch(2, 1)
        self.function_tabWidget.addTab(self.mappingTab, icon5, "")
        self.videoTab = QtWidgets.QWidget()
        self.videoTab.setObjectName("videoTab")
        self.verticalLayout_22 = QtWidgets.QVBoxLayout(self.videoTab)
        self.verticalLayout_22.setObjectName("verticalLayout_22")
        self.horizontalLayout_25 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_25.setObjectName("horizontalLayout_25")
        self.videoLineEdit = self.setup_line_edit(self.videoTab, "videoLineEdit", self.horizontalLayout_25, path_type='video')
        self.videoButton, icon6 = self.setup_button_icon(self.videoTab, 'video', 1, self.horizontalLayout_25)
        self.verticalLayout_22.addLayout(self.horizontalLayout_25)
        self.frame_9 = setup_frame(self.videoTab, "frame_9")
        self.verticalLayout_23 = QtWidgets.QVBoxLayout(self.frame_9)
        self.verticalLayout_23.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_23.setSpacing(0)
        self.verticalLayout_23.setObjectName("verticalLayout_23")
        self.frame_10 = setup_frame(self.frame_9, "frame_10", set_size=True)
        self.horizontalLayout_32 = QtWidgets.QHBoxLayout(self.frame_10)
        self.horizontalLayout_32.setObjectName("horizontalLayout_32")
        self.muteButton = QtWidgets.QPushButton(parent=self.frame_10)
        self.muteButton.setText("")
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap("resources/icons/multimedia_mute.svg"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
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
        self.timelineSlider.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.timelineSlider.setObjectName("timelineSlider")
        self.horizontalLayout_32.addWidget(self.timelineSlider)
        self.durationLabel = QtWidgets.QLabel(parent=self.frame_10)
        self.durationLabel.setObjectName("durationLabel")
        self.horizontalLayout_32.addWidget(self.durationLabel)
        self.mfaceCheckBox_4, self.tiltCheckBox_4, self.exposureCheckBox_4 = self.setup_checkboxes(self.frame_10, self.horizontalLayout_32, 4, True)
        self.verticalLayout_23.addWidget(self.frame_10)
        self.videoWidget = QtMultimediaWidgets.QVideoWidget(parent=self.frame_9)
        self.videoWidget.setStyleSheet("background: #1f2c33")
        self.videoWidget.setObjectName("videoWidget")
        self.verticalLayout_23.addWidget(self.videoWidget)
        self.frame_16 = setup_frame(self.frame_9, "frame_16")
        self.horizontalLayout_19 = QtWidgets.QHBoxLayout(self.frame_16)
        self.horizontalLayout_19.setContentsMargins(-1, 9, -1, 0)
        self.horizontalLayout_19.setObjectName("horizontalLayout_19")
        self.cropButton_4 = self.setup_button(self.frame_16, 'crop', 4, self.horizontalLayout_19)
        self.videocropButton = self.setup_button(self.frame_16, 'vcrop', 1, self.horizontalLayout_19)
        self.cancelButton_3 = self.setup_button(self.frame_16, 'cancel', 3, self.horizontalLayout_19)
        self.verticalLayout_23.addWidget(self.frame_16)
        self.frame_11 = setup_frame(self.frame_9, "frame_11", set_size=True)
        self.horizontalLayout_31 = QtWidgets.QHBoxLayout(self.frame_11)
        self.horizontalLayout_31.setObjectName("horizontalLayout_31")
        self.progressBar_3 = setup_progress_bar(self.frame_11, "progressBar_3", self.horizontalLayout_31)
        self.verticalLayout_23.addWidget(self.frame_11)
        self.verticalLayout_23.setStretch(0, 1)
        self.verticalLayout_23.setStretch(1, 10)
        self.verticalLayout_23.setStretch(2, 1)
        self.verticalLayout_23.setStretch(3, 1)
        self.verticalLayout_22.addWidget(self.frame_9)
        self.horizontalLayout_30 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_30.setObjectName("horizontalLayout_30")
        self.destinationLineEdit_4 = self.setup_line_edit(self.videoTab, "destinationLineEdit_4", self.horizontalLayout_30, path_type='folder')
        self.destinationButton_4 = self.setup_button(self.videoTab, 'destination', 4, self.horizontalLayout_30)
        self.verticalLayout_22.addLayout(self.horizontalLayout_30)
        self.horizontalLayout_28 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_28.setObjectName("horizontalLayout_28")
        self.horizontalLayout_29 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_29.setObjectName("horizontalLayout_29")
        self.playButton = self.setup_button(self.videoTab, 'play', 1, self.horizontalLayout_29, normal=False)
        self.stopButton = self.setup_button(self.videoTab, 'stop', 1, self.horizontalLayout_29, normal=False)
        self.stepbackButton = self.setup_button(self.videoTab, 'stepback', 1, self.horizontalLayout_29, normal=False)
        self.stepfwdButton = self.setup_button(self.videoTab, 'stepfwd', 1, self.horizontalLayout_29, normal=False)
        self.rewindButton = self.setup_button(self.videoTab, 'rewind', 1, self.horizontalLayout_29, normal=False)
        self.fastfwdButton = self.setup_button(self.videoTab, 'fastfwd', 1, self.horizontalLayout_29, normal=False)
        self.goto_beginingButton = self.setup_button(self.videoTab, 'goto_begining', 1, self.horizontalLayout_29, normal=False)
        self.goto_endButton = self.setup_button(self.videoTab, 'goto_end', 1, self.horizontalLayout_29, normal=False)
        self.startmarkerButton = self.setup_button(self.videoTab, 'startmarker', 1, self.horizontalLayout_29, normal=False)
        self.endmarkerButton = self.setup_button(self.videoTab, 'endmarker', 1, self.horizontalLayout_29, normal=False)
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout_29.addItem(spacerItem4)
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.selectStartMarkerButton = QtWidgets.QPushButton(parent=self.videoTab)
        self.selectStartMarkerButton.setMinimumSize(QtCore.QSize(150, 0))
        self.selectStartMarkerButton.setObjectName("selectStartMarkerButton")
        self.gridLayout_2.addWidget(self.selectStartMarkerButton, 0, 1, 1, 1)
        self.label_15 = QtWidgets.QLabel(parent=self.videoTab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
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
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_16.sizePolicy().hasHeightForWidth())
        self.video = Video(self.audio, self.videoWidget, self.player, self.timelineSlider, self.volumeSlider, 
                           self.positionLabel, self.durationLabel, self.selectEndMarkerButton)
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
        self.verticalLayout_22.setStretch(3, 2)
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
        self.gammaDial = setup_dial(
            self.settingsTab, 1, 2_000, 5, 100, 1_000, 1_000, True, False, False, True, "gammaDial", self.verticalLayout_3
            )
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout.addItem(spacerItem5)
        self.label = QtWidgets.QLabel(parent=self.settingsTab)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.gammaLCDNumber = setup_lcd(self.settingsTab, "gammaLCDNumber", self.horizontalLayout, 1_000)
        spacerItem6 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout.addItem(spacerItem6)
        self.verticalLayout_3.addLayout(self.horizontalLayout)
        self.verticalLayout_3.setStretch(0, 7)
        self.horizontalLayout_3.addLayout(self.verticalLayout_3)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.faceDial = setup_dial(
            self.settingsTab, max_=100, dval=62, notchvis=True, name="faceDial", layout=self.verticalLayout_4
            )
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        spacerItem7 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem7)
        self.label_2 = QtWidgets.QLabel(parent=self.settingsTab)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_4.addWidget(self.label_2)
        self.faceLCDNumber = setup_lcd(self.settingsTab, "faceLCDNumber", self.horizontalLayout_4, 62)
        spacerItem8 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem8)
        self.verticalLayout_4.addLayout(self.horizontalLayout_4)
        self.verticalLayout_4.setStretch(0, 7)
        self.horizontalLayout_3.addLayout(self.verticalLayout_4)
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.sensitivityDial = setup_dial(
            self.settingsTab, 0, 100, dval=50, invapp=False, invctrl=False, notchvis=True, name="sensitivityDial", layout=self.verticalLayout_5
        )
        self.horizontalLayout_33 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_33.setObjectName("horizontalLayout_33")
        spacerItem9 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout_33.addItem(spacerItem9)
        self.label_3 = QtWidgets.QLabel(parent=self.settingsTab)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_33.addWidget(self.label_3)
        self.sensitivityLCDNumber = setup_lcd(self.settingsTab, "sensitivityLCDNumber", self.horizontalLayout_33, 50)
        spacerItem10 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout_33.addItem(spacerItem10)
        self.verticalLayout_5.addLayout(self.horizontalLayout_33)
        self.verticalLayout_5.setStretch(0, 7)
        self.horizontalLayout_3.addLayout(self.verticalLayout_5)
        spacerItem11 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem11)
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout()
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.label_4 = QtWidgets.QLabel(parent=self.settingsTab)
        self.label_4.setObjectName("label_4")
        self.verticalLayout_9.addWidget(self.label_4)
        self.widthLineEdit = self.setup_line_edit(self.settingsTab, "widthLineEdit", self.verticalLayout_9)
        self.horizontalLayout_9.addLayout(self.verticalLayout_9)
        self.verticalLayout_10 = QtWidgets.QVBoxLayout()
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.label_5 = QtWidgets.QLabel(parent=self.settingsTab)
        self.label_5.setObjectName("label_5")
        self.verticalLayout_10.addWidget(self.label_5)
        self.heightLineEdit = self.setup_line_edit(self.settingsTab, "heightLineEdit", self.verticalLayout_10)
        self.horizontalLayout_9.addLayout(self.verticalLayout_10)
        self.verticalLayout_6.addLayout(self.horizontalLayout_9)
        self.label_7 = QtWidgets.QLabel(parent=self.settingsTab)
        self.label_7.setObjectName("label_7")
        self.verticalLayout_6.addWidget(self.label_7, 0, QtCore.Qt.AlignmentFlag.AlignHCenter)
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.topDial, self.topLCDNumber, self.label_8 = setup_dial_area(
            self.settingsTab, "top", "label_8", "verticalLayout_14", self.horizontalLayout_10
            )
        self.bottomDial, self.bottomLCDNumber, self.label_9 = setup_dial_area(
            self.settingsTab, "bottom", "label_9", "verticalLayout_13", self.horizontalLayout_10
            )
        self.leftDial, self.leftLCDNumber, self.label_10 = setup_dial_area(
            self.settingsTab, "left", "label_10", "verticalLayout_12", self.horizontalLayout_10
            )
        self.rightDial, self.rightLCDNumber, self.label_11 = setup_dial_area(
            self.settingsTab, "right", "label_11", "verticalLayout_11", self.horizontalLayout_10
            )
        self.verticalLayout_6.addLayout(self.horizontalLayout_10)
        self.horizontalLayout_3.addLayout(self.verticalLayout_6)
        self.horizontalLayout_3.setStretch(0, 1)
        self.horizontalLayout_3.setStretch(1, 1)
        self.horizontalLayout_3.setStretch(2, 1)
        self.horizontalLayout_3.setStretch(4, 2)
        icon19 = QtGui.QIcon()
        icon19.addPixmap(QtGui.QPixmap("resources/icons/settings.svg"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.settings_tabWidget.addTab(self.settingsTab, icon19, "")
        self.formatTab = QtWidgets.QWidget()
        self.formatTab.setObjectName("formatTab")
        self.horizontalLayout_27 = QtWidgets.QHBoxLayout(self.formatTab)
        self.horizontalLayout_27.setObjectName("horizontalLayout_27")
        self.radioButton_1 = setup_radio_button(self.formatTab, self.horizontalLayout_27, '_no', 1, checked=True, spacer=True)
        self.radioButton_2 = setup_radio_button(self.formatTab, self.horizontalLayout_27, '_bmp', 2)
        self.radioButton_3 = setup_radio_button(self.formatTab, self.horizontalLayout_27, '_jpg', 3)
        self.radioButton_4 = setup_radio_button(self.formatTab, self.horizontalLayout_27, '_png', 4)
        self.radioButton_5 = setup_radio_button(self.formatTab, self.horizontalLayout_27, '_tiff', 5)
        self.radioButton_6 = setup_radio_button(self.formatTab, self.horizontalLayout_27, '_webp', 6)
        icon20 = QtGui.QIcon()
        icon20.addPixmap(QtGui.QPixmap("resources/icons/memory_card.svg"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
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
        self.gammaDial.valueChanged['int'].connect(self.gammaLCDNumber.display) # type: ignore
        self.faceDial.valueChanged['int'].connect(self.faceLCDNumber.display) # type: ignore
        self.sensitivityDial.valueChanged['int'].connect(self.sensitivityLCDNumber.display) # type: ignore
        self.topDial.valueChanged['int'].connect(self.topLCDNumber.display) # type: ignore
        self.bottomDial.valueChanged['int'].connect(self.bottomLCDNumber.display) # type: ignore
        self.leftDial.valueChanged['int'].connect(self.leftLCDNumber.display) # type: ignore
        self.rightDial.valueChanged['int'].connect(self.rightLCDNumber.display) # type: ignore

        self.actionAbout_Face_Cropper.triggered.connect(lambda: load_about_form())
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
        self.folderButton_1.clicked.connect(
            lambda: self.open_folder(self.folderLineEdit_1, self.folderWidget, self.file_model))
        self.folderButton_2.clicked.connect(lambda: self.open_folder(self.folderLineEdit_2, self.mappingWidget))

        self.tableButton.clicked.connect(lambda: self.open_table())

        self.videoButton.clicked.connect(
            lambda: self.video.open_video(self, self.videoLineEdit, self.playButton, self.cropButton_4))

        self.destinationButton_1.clicked.connect(lambda: self.open_folder(self.destinationLineEdit_1))
        self.destinationButton_2.clicked.connect(lambda: self.open_folder(self.destinationLineEdit_2))
        self.destinationButton_3.clicked.connect(lambda: self.open_folder(self.destinationLineEdit_3))
        self.destinationButton_4.clicked.connect(lambda: self.open_folder(self.destinationLineEdit_4))
        
        self.cropButton_1.clicked.connect(lambda: self.crop_photo())
        self.cropButton_2.clicked.connect(lambda: self.folder_process())
        self.cropButton_3.clicked.connect(lambda: self.mapping_process())
        self.cropButton_4.clicked.connect(lambda: self.crop_frame())

        self.videocropButton.clicked.connect(lambda: self.video_process())

        self.playButton.clicked.connect(lambda: self.video.play_video(self.playButton))
        self.playButton.clicked.connect(
            lambda: change_widget_state(True, self.stopButton, self.stepbackButton,  self.stepfwdButton,
                                             self.fastfwdButton, self.rewindButton, self.goto_beginingButton,
                                             self.goto_endButton, self.startmarkerButton, self.endmarkerButton,
                                             self.selectEndMarkerButton, self.selectStartMarkerButton))
        self.stopButton.clicked.connect(lambda: self.video.stop_btn())
        self.stepbackButton.clicked.connect(lambda: self.video.stepback())
        self.stepfwdButton.clicked.connect(lambda: self.video.stepfwd())
        self.fastfwdButton.clicked.connect(lambda: self.video.fastfwd())
        self.rewindButton.clicked.connect(lambda: self.video.rewind())
        self.goto_beginingButton.clicked.connect(lambda: self.video.goto_begining())
        self.goto_endButton.clicked.connect(lambda: self.video.goto_end())
        self.startmarkerButton.clicked.connect(
            lambda: self.video.set_startPosition(self.selectStartMarkerButton, self.timelineSlider))
        self.endmarkerButton.clicked.connect(
            lambda: self.video.set_stopPosition(self.selectEndMarkerButton, self.timelineSlider))
        self.selectStartMarkerButton.clicked.connect(lambda: self.video.goto(self.selectStartMarkerButton))
        self.selectEndMarkerButton.clicked.connect(lambda: self.video.goto(self.selectEndMarkerButton))
        self.muteButton.clicked.connect(lambda: self.video.volume_mute(self.volumeSlider, self.muteButton))

        self.photoLineEdit.textChanged.connect(lambda: self.load_data(self.photoLineEdit, self.photoWidget))
        self.folderLineEdit_1.textChanged.connect(lambda: self.load_data(self.folderLineEdit_1, self.folderWidget))
        self.folderLineEdit_2.textChanged.connect(lambda: self.load_data(self.folderLineEdit_2, self.mappingWidget))

        self.connect_input_widgets(
            self.exposureCheckBox_1, self.exposureCheckBox_2, self.exposureCheckBox_3, self.exposureCheckBox_4,
            self.gammaDial, self.faceDial, self.sensitivityDial, self.topDial, self.bottomDial, self.leftDial,
            self.rightDial, self.widthLineEdit, self.heightLineEdit)

        self.treeView.selectionModel().selectionChanged.connect(lambda: self.reload_widgets())

        # Folder start connection
        self.cropper.folder_started.connect(
            lambda: disable_widget(self.widthLineEdit, self.heightLineEdit, self.exposureCheckBox_2,
                                        self.sensitivityDial, self.faceDial, self.gammaDial, self.topDial,
                                        self.bottomDial, self.leftDial, self.rightDial, self.folderLineEdit_1,
                                        self.destinationLineEdit_2, self.destinationButton_2, self.folderButton_1,
                                        self.radioButton_1, self.radioButton_2, self.radioButton_3, self.radioButton_4,
                                        self.radioButton_5, self.radioButton_6, self.cropButton_2,
                                        self.exposureCheckBox_2, self.mfaceCheckBox_2, self.tiltCheckBox_2))
        self.cropper.folder_started.connect(lambda: enable_widget(self.cancelButton_1))

        # Maping start connection
        self.cropper.mapping_started.connect(
            lambda: disable_widget(self.widthLineEdit, self.heightLineEdit, self.exposureCheckBox_3,
                                        self.sensitivityDial, self.faceDial, self.gammaDial, self.topDial,
                                        self.bottomDial, self.leftDial, self.rightDial, self.folderLineEdit_2,
                                        self.destinationLineEdit_3, self.destinationButton_3, self.folderButton_2,
                                        self.tableLineEdit, self.comboBox_1, self.comboBox_2, self.radioButton_1,
                                        self.radioButton_2, self.radioButton_3, self.radioButton_4, self.radioButton_5,
                                        self.radioButton_6, self.cropButton_3,
                                        self.exposureCheckBox_3, self.mfaceCheckBox_3, self.tiltCheckBox_3))
        self.cropper.mapping_started.connect(lambda: enable_widget(self.cancelButton_2))

        # Video start connection
        self.cropper.video_started.connect(
            lambda: disable_widget(self.widthLineEdit, self.heightLineEdit, self.exposureCheckBox_4,
                                        self.sensitivityDial, self.faceDial, self.gammaDial, self.topDial,
                                        self.bottomDial, self.leftDial, self.rightDial, self.videoLineEdit,
                                        self.destinationLineEdit_4, self.destinationButton_4, self.radioButton_1,
                                        self.radioButton_2, self.radioButton_3, self.radioButton_4, self.radioButton_5,
                                        self.radioButton_6, self.cropButton_4, self.videocropButton,
                                        self.exposureCheckBox_4, self.mfaceCheckBox_4, self.tiltCheckBox_4))
        self.cropper.video_started.connect(lambda: enable_widget(self.cancelButton_3))

        # Folder end connection
        self.cropper.folder_finished.connect(
            lambda: enable_widget(self.widthLineEdit, self.heightLineEdit, self.exposureCheckBox_2,
                                       self.sensitivityDial, self.faceDial, self.gammaDial, self.topDial,
                                       self.bottomDial, self.leftDial, self.rightDial, self.folderLineEdit_1,
                                       self.destinationLineEdit_2, self.destinationButton_2, self.folderButton_1,
                                       self.radioButton_1, self.radioButton_2, self.radioButton_3, self.radioButton_4,
                                       self.radioButton_5, self.radioButton_6, self.cropButton_2,
                                       self.exposureCheckBox_2, self.mfaceCheckBox_2, self.tiltCheckBox_2))
        self.cropper.folder_finished.connect(lambda: disable_widget(self.cancelButton_1))
        self.cropper.folder_finished.connect(lambda: show_message_box(self.destinationLineEdit_2))

        # Maping end connection
        self.cropper.mapping_finished.connect(
            lambda: enable_widget(self.widthLineEdit, self.heightLineEdit, self.exposureCheckBox_3,
                                       self.sensitivityDial, self.faceDial, self.gammaDial, self.topDial,
                                       self.bottomDial, self.leftDial, self.rightDial, self.folderLineEdit_2,
                                       self.destinationLineEdit_3, self.destinationButton_3, self.folderButton_2,
                                       self.tableLineEdit, self.comboBox_1, self.comboBox_2, self.radioButton_1,
                                       self.radioButton_2, self.radioButton_3, self.radioButton_4, self.radioButton_5,
                                       self.radioButton_6, self.cropButton_3,
                                       self.exposureCheckBox_3, self.mfaceCheckBox_3, self.tiltCheckBox_3))
        self.cropper.mapping_finished.connect(lambda: disable_widget(self.cancelButton_2))
        self.cropper.mapping_finished.connect(lambda: show_message_box(self.destinationLineEdit_3))

        # Video end connection
        self.cropper.video_finished.connect(
            lambda: enable_widget(self.widthLineEdit, self.heightLineEdit, self.exposureCheckBox_4,
                                       self.sensitivityDial, self.faceDial, self.gammaDial, self.topDial,
                                       self.bottomDial, self.leftDial, self.rightDial, self.videoLineEdit,
                                       self.destinationLineEdit_4, self.destinationButton_4, self.radioButton_1,
                                       self.radioButton_2, self.radioButton_3, self.radioButton_4, self.radioButton_5,
                                       self.radioButton_6, self.cropButton_4, self.videocropButton,
                                       self.exposureCheckBox_4, self.mfaceCheckBox_4, self.tiltCheckBox_4))
        self.cropper.video_finished.connect(lambda: disable_widget(self.cancelButton_3))
        self.cropper.video_finished.connect(lambda: show_message_box(self.destinationLineEdit_4))

        self.cropper.folder_progress.connect(self.update_progress_1)
        self.cropper.mapping_progress.connect(self.update_progress_2)
        self.cropper.video_progress.connect(self.update_progress_3)

        self.retranslateUi()
        self.function_tabWidget.setCurrentIndex(0)
        self.settings_tabWidget.setCurrentIndex(0)
        self.disable_buttons()
        change_widget_state(False, self.cropButton_1, self.cropButton_2, self.cropButton_3, self.cropButton_4,
                                 self.videocropButton, self.cancelButton_1, self.cancelButton_2, self.cancelButton_3,
                                 self.playButton, self.stopButton, self.stepbackButton, self.stepfwdButton,
                                 self.rewindButton, self.fastfwdButton, self.goto_beginingButton, self.goto_endButton,
                                 self.startmarkerButton, self.endmarkerButton, self.selectStartMarkerButton,
                                 self.selectEndMarkerButton, self.timelineSlider)
        QtCore.QMetaObject.connectSlotsByName(self)

    def retranslateUi(self) -> None:
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.photoLineEdit.setPlaceholderText(_translate("MainWindow", "Choose the image you want to crop"))
        self.photoButton.setText(_translate("MainWindow", "PushButton"))
        self.mfaceCheckBox_1.setText(_translate("MainWindow", "Multi-Face"))
        self.tiltCheckBox_1.setText(_translate("MainWindow", "Autotilt"))
        self.exposureCheckBox_1.setText(_translate("MainWindow", "Autocorrect"))
        self.destinationLineEdit_1.setPlaceholderText(_translate("MainWindow", "Choose where you want to save the cropped image"))
        self.destinationButton_1.setText(_translate("MainWindow", "Destination Folder"))
        self.function_tabWidget.setTabText(self.function_tabWidget.indexOf(self.photoTab), _translate("MainWindow", "Photo Crop"))
        self.folderLineEdit_1.setPlaceholderText(_translate("MainWindow", "Choose the folder you want to crop"))
        self.folderButton_1.setText(_translate("MainWindow", "Select Folder"))
        self.mfaceCheckBox_2.setText(_translate("MainWindow", "Multi-Face"))
        self.tiltCheckBox_2.setText(_translate("MainWindow", "Autotilt"))
        self.exposureCheckBox_2.setText(_translate("MainWindow", "Autocorrect"))
        self.destinationLineEdit_2.setPlaceholderText(_translate("MainWindow", "Choose where you want to save the cropped images"))
        self.destinationButton_2.setText(_translate("MainWindow", "Destination Folder"))
        self.function_tabWidget.setTabText(self.function_tabWidget.indexOf(self.folder_Tab), _translate("MainWindow", "Folder Crop"))
        self.tableLineEdit.setPlaceholderText(_translate("MainWindow", "Choose the Excel or CSV file with the mapping"))
        self.folderLineEdit_2.setPlaceholderText(_translate("MainWindow", "Choose the folder you want to crop"))
        self.folderButton_2.setText(_translate("MainWindow", "Select Folder"))
        self.tableButton.setText(_translate("MainWindow", "Open File"))
        self.mfaceCheckBox_3.setText(_translate("MainWindow", "Multi-Face"))
        self.tiltCheckBox_3.setText(_translate("MainWindow", "Autotilt"))
        self.exposureCheckBox_3.setText(_translate("MainWindow", "Autocorrect"))
        self.comboBox_1.setPlaceholderText(_translate("MainWindow", "Filename column"))
        self.comboBox_2.setPlaceholderText(_translate("MainWindow", "Mapping column"))
        self.destinationLineEdit_3.setPlaceholderText(_translate("MainWindow", "Choose where you want to save the cropped images"))
        self.destinationButton_3.setText(_translate("MainWindow", "Destination Folder"))
        self.function_tabWidget.setTabText(self.function_tabWidget.indexOf(self.mappingTab), _translate("MainWindow", "Mapping Crop"))
        self.videoLineEdit.setPlaceholderText(_translate("MainWindow", "Choose the video you want to crop"))
        self.videoButton.setText(_translate("MainWindow", "Open Video"))
        self.positionLabel.setText(_translate("MainWindow", "00:00:00"))
        self.durationLabel.setText(_translate("MainWindow", "00:00:00"))
        self.mfaceCheckBox_4.setText(_translate("MainWindow", "Multi-Face"))
        self.tiltCheckBox_4.setText(_translate("MainWindow", "Autotilt"))
        self.exposureCheckBox_4.setText(_translate("MainWindow", "Autocorrect"))
        self.destinationLineEdit_4.setPlaceholderText(_translate("MainWindow", "Choose where you want to save the cropped images"))
        self.destinationButton_4.setText(_translate("MainWindow", "Destination Folder"))
        self.selectStartMarkerButton.setText(_translate("MainWindow", "00:00:00"))
        self.selectEndMarkerButton.setText(_translate("MainWindow", "00:00:00"))
        self.function_tabWidget.setTabText(self.function_tabWidget.indexOf(self.videoTab), _translate("MainWindow", "Video Crop"))
        self.label.setText(_translate("MainWindow", "Gamma:"))
        self.label_2.setText(_translate("MainWindow", "Face %:"))
        self.label_3.setText(_translate("MainWindow", "Sensitivity:"))
        self.label_4.setText(_translate("MainWindow", "Width (px)"))
        self.widthLineEdit.setPlaceholderText(_translate("MainWindow", "Try typing a number e.g. 400"))
        self.label_5.setText(_translate("MainWindow", "Height (px)"))
        self.heightLineEdit.setPlaceholderText(_translate("MainWindow", "Try typing a number e.g. 400"))
        self.label_7.setText(_translate("MainWindow", "Padding"))
        self.label_8.setText(_translate("MainWindow", "Top:"))
        self.label_9.setText(_translate("MainWindow", "Bottom:"))
        self.label_10.setText(_translate("MainWindow", "Left:"))
        self.label_11.setText(_translate("MainWindow", "Right:"))
        self.settings_tabWidget.setTabText(self.settings_tabWidget.indexOf(self.settingsTab), _translate("MainWindow", "Settings"))
        self.settings_tabWidget.setTabText(self.settings_tabWidget.indexOf(self.formatTab), _translate("MainWindow", "Format Conversion"))
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

    def setup_checkboxes(self, parent: QtWidgets.QWidget, layout: QtWidgets.QHBoxLayout, series: int,
                         fix_spacer: Optional[bool] = False) -> Tuple[QtWidgets.QCheckBox, ...]:
        stylesheet = """QCheckBox:unchecked{color: red}
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
        
        mfaceCheckBox, tiltCheckBox, exposureCheckBox = (QtWidgets.QCheckBox(parent=parent) for _ in range(3))

        if fix_spacer:
            spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Minimum)
        else:
            spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        layout.addItem(spacerItem)
        mfaceCheckBox.setStyleSheet(stylesheet)
        mfaceCheckBox.setObjectName(f"mfaceCheckBox_{series}")
        layout.addWidget(mfaceCheckBox)
        tiltCheckBox.setStyleSheet(stylesheet)
        tiltCheckBox.setObjectName(f"tiltCheckBox_{series}")
        layout.addWidget(tiltCheckBox)
        exposureCheckBox.setStyleSheet(stylesheet)
        exposureCheckBox.setObjectName(f"exposureCheckBox_{series}")
        layout.addWidget(exposureCheckBox, 0, QtCore.Qt.AlignmentFlag.AlignRight)
        # Connect Checkboxes
        mfaceCheckBox.clicked.connect(lambda: self.reload_widgets())
        exposureCheckBox.clicked.connect(lambda: self.reload_widgets())
        tiltCheckBox.clicked.connect(lambda: self.reload_widgets())
        mfaceCheckBox.clicked.connect(lambda: uncheck_boxes(exposureCheckBox, tiltCheckBox))
        exposureCheckBox.clicked.connect(lambda: uncheck_boxes(mfaceCheckBox))
        tiltCheckBox.clicked.connect(lambda: uncheck_boxes(mfaceCheckBox))
        return mfaceCheckBox, tiltCheckBox, exposureCheckBox

    def setup_line_edit(self, parent: QtWidgets.QWidget, name: str,
                        layout: Optional[Union[QtWidgets.QHBoxLayout, QtWidgets.QVBoxLayout]] = None,
                        path_type: Optional[str] = None) -> QtWidgets.QLineEdit:
        line_edit: Union[NumberLineEdit, PathLineEdit]
        if path_type is None:
            line_edit = NumberLineEdit(parent=parent)
            line_edit.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhDigitsOnly)
        else:
            line_edit = PathLineEdit(path_type, parent=parent)
            line_edit.setMinimumSize(QtCore.QSize(0, 24))
            line_edit.setMaximumSize(QtCore.QSize(16_777_215, 24))
            line_edit.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhUrlCharactersOnly)
    
        line_edit.setObjectName(name)
        line_edit.textChanged.connect(lambda: self.disable_buttons())
        if layout is None:
            return line_edit
        layout.addWidget(line_edit)
        return line_edit

    def initialize_button(self,
                          parent: QtWidgets.QWidget,
                          icon_: str,
                          series: int,
                          normal: Optional[bool] = True) -> Tuple[QtWidgets.QPushButton, QtGui.QIcon]:
        logo_dict: Dict[str, Tuple[QtGui.QPixmap, str]] = {
            'crop': (QtGui.QPixmap("resources/icons/crop.svg"), f"cropButton_{series}"),
            'picture': (QtGui.QPixmap("resources/icons/picture.svg"), "photoButton"),
            'folder': (QtGui.QPixmap("resources/icons/folder.svg"), f"folderButton_{series}"),
            'destination': (QtGui.QPixmap("resources/icons/folder.svg"), f"destinationButton_{series}"),
            'cancel': (QtGui.QPixmap("resources/icons/cancel.svg"), f"cancelButton_{series}"),
            'data': (QtGui.QPixmap("resources/icons/excel.svg"), "tableButton"),
            'video': (QtGui.QPixmap("resources/icons/clapperboard.svg"), "videoButton"),
            'vcrop': (QtGui.QPixmap("resources/icons/crop_video.svg"), "videocropButton"),
            'play': (QtGui.QPixmap("resources/icons/multimedia_play.svg"), 'playButton'),
            'stop': (QtGui.QPixmap("resources/icons/multimedia_stop.svg"), 'stopButton'),
            'stepback': (QtGui.QPixmap("resources/icons/multimedia_left.svg"), 'stepbackButton'),
            'stepfwd': (QtGui.QPixmap("resources/icons/multimedia_right.svg"), 'stepfwdButton'),
            'rewind': (QtGui.QPixmap("resources/icons/multimedia_rewind.svg"), 'rewindButton'),
            'fastfwd': (QtGui.QPixmap("resources/icons/multimedia_fastfwd.svg"), 'fastfwdButton'),
            'goto_begining': (QtGui.QPixmap("resources/icons/multimedia_begining.svg"), 'goto_beginingButton'),
            'goto_end': (QtGui.QPixmap("resources/icons/multimedia_end.svg"), 'goto_endButton'),
            'startmarker': (QtGui.QPixmap("resources/icons/multimedia_leftmarker.svg"), 'startmarkerButton'),
            'endmarker': (QtGui.QPixmap("resources/icons/multimedia_rightmarker.svg"), 'endmarkerButton')
            }
        
        button = QtWidgets.QPushButton(parent=parent)
        if normal:
            x = (0, 24)
            y = (16_777_215, 24)
        else:
            x = y = (48, 48)
        button.setMinimumSize(QtCore.QSize(*x))
        button.setMaximumSize(QtCore.QSize(*y))
        button.setText("")
        icon = QtGui.QIcon()
        pixmap, name = logo_dict[icon_]
        icon.addPixmap(pixmap, QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        button.setIcon(icon)
        if not normal:
            button.setIconSize(QtCore.QSize(32, 32))
        button.setObjectName(name)
        if icon_ == 'cancel':
            button.clicked.connect(lambda: terminate(self.cropper))
        return button, icon

    def setup_button(self,
                     parent: QtWidgets.QWidget,
                     icon_: str,
                     series: int,
                     layout: Optional[Union[QtWidgets.QHBoxLayout, QtWidgets.QVBoxLayout]] = None,
                     normal: Optional[bool] = True) -> QtWidgets.QPushButton:
        button, _ = self.initialize_button(parent, icon_, series, normal)
        if layout is not None:
            layout.addWidget(button)
        return button

    def setup_button_icon(self,
                          parent: QtWidgets.QWidget,
                          icon_: str,
                          series: int,
                          layout: Optional[Union[QtWidgets.QHBoxLayout, QtWidgets.QVBoxLayout]] = None,
                          normal: Optional[bool] = True) -> Tuple[QtWidgets.QPushButton, QtGui.QIcon]:
        button, icon = self.initialize_button(parent, icon_, series, normal)
        if layout is not None:
            layout.addWidget(button)
        return button, icon

    def connect_input_widgets(self, *input_widgets: QtWidgets.QWidget) -> None:
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
        self.display_crop(self.folderWidget, Path(self.file_model.filePath(self.treeView.currentIndex())),
                          self.exposureCheckBox_2, self.mfaceCheckBox_2, self.tiltCheckBox_2)

    def reload_widgets(self) -> None:
        if not self.widthLineEdit.text() or not self.heightLineEdit.text():
            return None
        if self.function_tabWidget.currentIndex() == 0:
            f_name = Path(self.photoLineEdit.text())
            if not f_name.as_posix():
                return None
            self.display_crop(
                self.photoWidget, f_name, self.exposureCheckBox_1, self.mfaceCheckBox_1, self.tiltCheckBox_1
            )
        elif self.function_tabWidget.currentIndex() == 1:
            if self.treeView.currentIndex().isValid():
                folder = Path(self.file_model.filePath(self.treeView.currentIndex()))
            else:
                folder = Path(self.folderLineEdit_1.text())

            if not folder.as_posix():
                return None
            self.display_crop(
                self.folderWidget, folder, self.exposureCheckBox_2, self.mfaceCheckBox_2, self.tiltCheckBox_2
            )
        elif self.function_tabWidget.currentIndex() == 2:
            folder = Path(self.folderLineEdit_2.text())
            if not folder.as_posix():
                return None
            self.display_crop(
                self.mappingWidget, folder, self.exposureCheckBox_3, self.mfaceCheckBox_3, self.tiltCheckBox_3
            )

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:
        if not event.mimeData().hasUrls():
            event.ignore()
            return

        event.setDropAction(QtCore.Qt.DropAction.CopyAction)
        file_path = Path(event.mimeData().urls()[0].toLocalFile())

        if file_path.is_dir():
            self.handle_directory(file_path)
        elif file_path.is_file():
            self.handle_file(file_path)

        event.accept()

    def handle_directory(self, file_path: Path) -> None:
        extentions = {y.suffix.lower() for y in file_path.iterdir()}
        mask = [ext in extentions for ext in PANDAS_TYPES]
        if any(mask):
            self.function_tabWidget.setCurrentIndex(2)
            self.folderLineEdit_2.setText(file_path.as_posix())
            if self.widthLineEdit.text() and self.heightLineEdit.text():
                self.display_crop(self.mappingWidget, file_path, self.exposureCheckBox_3, self.mfaceCheckBox_3,
                                  self.tiltCheckBox_3)  
        else:
            self.function_tabWidget.setCurrentIndex(1)
            self.folderLineEdit_1.setText(file_path.as_posix())
            if self.widthLineEdit.text() and self.heightLineEdit.text():
                self.display_crop(self.folderWidget, file_path, self.exposureCheckBox_2, self.mfaceCheckBox_2,
                                  self.tiltCheckBox_2)

    def handle_file(self, file_path: Path) -> None:
        if file_path.suffix.lower() in IMAGE_TYPES:
            self.handle_image_file(file_path)
        elif file_path.suffix.lower() in VIDEO_TYPES:
            self.handle_video_file(file_path)
        elif file_path.suffix.lower() in PANDAS_TYPES:
            self.handle_pandas_file(file_path)

    def handle_image_file(self, file_path: Path) -> None:
        self.function_tabWidget.setCurrentIndex(0)
        self.photoLineEdit.setText(file_path.as_posix())
        if self.widthLineEdit.text() and self.heightLineEdit.text():
            self.display_crop(self.photoWidget, file_path, self.exposureCheckBox_1, self.mfaceCheckBox_1,
                              self.tiltCheckBox_1)

    def handle_video_file(self, file_path: Path) -> None:
        self.function_tabWidget.setCurrentIndex(3)
        self.videoLineEdit.setText(file_path.as_posix())

    def handle_pandas_file(self, file_path: Path) -> None:
        self.function_tabWidget.setCurrentIndex(2)
        self.tableLineEdit.setText(file_path.as_posix())
        data = open_file(file_path)
        try:
            assert isinstance(data, pd.DataFrame)
        except AssertionError:
            return None

        self.process_data(data)

    def process_data(self, data: pd.DataFrame) -> None:
        self.data_frame = data
        if self.data_frame is None:
            return

        self.model = DataFrameModel(self.data_frame)
        self.tableView.setModel(self.model)
        self.comboBox_1.addItems(self.data_frame.columns.to_numpy())
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

    def load_data(self, line_edit: QtWidgets.QLineEdit, image_widget: ImageWidget) -> None:
        try:
            if line_edit is self.photoLineEdit:
                self.display_crop(image_widget, line_edit, self.exposureCheckBox_1, self.mfaceCheckBox_1,
                                  self.tiltCheckBox_1)
            elif line_edit is self.folderLineEdit_1:
                f_name = line_edit.text()
                self.file_model.setRootPath(f_name)
                self.treeView.setRootIndex(self.file_model.index(f_name))
                self.display_crop(image_widget, line_edit, self.exposureCheckBox_2, self.mfaceCheckBox_2,
                                  self.tiltCheckBox_2)
            elif line_edit is self.folderLineEdit_2:
                self.display_crop(image_widget, line_edit, self.exposureCheckBox_3, self.mfaceCheckBox_3,
                                  self.tiltCheckBox_3)
        except (IndexError, FileNotFoundError, ValueError, AttributeError):
            return None

    def open_folder(self, line_edit: QtWidgets.QLineEdit, image_widget: Optional[ImageWidget] = None,
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
        type_string = 'All Files (*);;' + ';;'.join(f'{_} Files (*{_})' for _ in np.sort(PANDAS_TYPES))
        f_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File', Photo().default_directory, type_string)
        if f_name is None:
            return None
        self.tableLineEdit.setText(f_name)
        data = open_file(f_name)
        try:
            assert isinstance(data, pd.DataFrame)
        except AssertionError:
            return None

        self.process_data(data)

    def disable_buttons(self) -> None:
        def all_filled(*line_edits: Union[QtWidgets.QLineEdit, QtWidgets.QComboBox]) -> bool:
            x = all(edit.text() for edit in line_edits if isinstance(edit, QtWidgets.QLineEdit))
            y = all(edit.currentText() for edit in line_edits if isinstance(edit, QtWidgets.QComboBox))
            return x and y

        def update_widget_state(condition: bool, *widgets) -> None:
            for widget in widgets:
                change_widget_state(condition, widget)

        common_line_edits = (self.widthLineEdit, self.heightLineEdit)

        # Photo logic
        update_widget_state(
            all_filled(self.photoLineEdit, self.destinationLineEdit_1, *common_line_edits), self.cropButton_1)
        # Folder logic
        update_widget_state(
            all_filled(self.folderLineEdit_1, self.destinationLineEdit_2, *common_line_edits), self.cropButton_2)
        # Mapping logic
        update_widget_state(
            all_filled(self.folderLineEdit_2, self.tableLineEdit, self.destinationLineEdit_3, self.comboBox_1,
                       self.comboBox_2, *common_line_edits), self.cropButton_3)
        # Video logic
        update_widget_state(
            all_filled(self.videoLineEdit, self.destinationLineEdit_4, *common_line_edits), self.cropButton_4,
            self.videocropButton)

    def display_crop(self,
                     image_widget: ImageWidget,
                     line_edit: Union[Path, QtWidgets.QLineEdit],
                     exposure: QtWidgets.QCheckBox,
                     multi: QtWidgets.QCheckBox,
                     tilt: QtWidgets.QCheckBox) -> None:
        job = Job(
            self.widthLineEdit,
            self.heightLineEdit,
            exposure,
            multi,
            tilt,
            self.sensitivityDial,
            self.faceDial,
            self.gammaDial, 
            self.topDial, 
            self.bottomDial, 
            self.leftDial, 
            self.rightDial,
            (self.radioButton_1, self.radioButton_2, self.radioButton_3,
             self.radioButton_4, self.radioButton_5, self.radioButton_6),
        )
        self.cropper.display_crop(job, line_edit, image_widget)

    def crop_photo(self) -> None:
        job = Job(
            self.widthLineEdit,
            self.heightLineEdit,
            self.exposureCheckBox_1,
            self.mfaceCheckBox_1,
            self.tiltCheckBox_1,
            self.sensitivityDial,
            self.faceDial,
            self.gammaDial, 
            self.topDial, 
            self.bottomDial, 
            self.leftDial, 
            self.rightDial,
            (self.radioButton_1, self.radioButton_2, self.radioButton_3,
             self.radioButton_4, self.radioButton_5, self.radioButton_6),
             photo_path=self.photoLineEdit,
             destination=self.destinationLineEdit_1,
        )
        self.cropper.crop(Path(self.photoLineEdit.text()), job, self.cropper.face_workers[0][0],
                          self.cropper.face_workers[0][1])

    def run_batch_process(self, function: Callable, job: Job) -> None:
        self.cropper.reset()
        process = Process(target=function, daemon=True, args=(job,))
        process.run()

    def folder_process(self) -> None:
        job = Job(
            self.widthLineEdit,
            self.heightLineEdit,
            self.exposureCheckBox_2,
            self.mfaceCheckBox_2,
            self.tiltCheckBox_2,
            self.sensitivityDial,
            self.faceDial,
            self.gammaDial, 
            self.topDial, 
            self.bottomDial, 
            self.leftDial, 
            self.rightDial,
            (self.radioButton_1, self.radioButton_2, self.radioButton_3,
             self.radioButton_4, self.radioButton_5, self.radioButton_6),
             folder_path=self.folderLineEdit_1,
             destination=self.destinationLineEdit_2,
        )
        self.run_batch_process(self.cropper.crop_dir, job)

    def mapping_process(self) -> None:
        job = Job(
            self.widthLineEdit,
            self.heightLineEdit,
            self.exposureCheckBox_3,
            self.mfaceCheckBox_3,
            self.tiltCheckBox_3,
            self.sensitivityDial,
            self.faceDial,
            self.gammaDial, 
            self.topDial, 
            self.bottomDial, 
            self.leftDial, 
            self.rightDial,
            (self.radioButton_1, self.radioButton_2, self.radioButton_3,
             self.radioButton_4, self.radioButton_5, self.radioButton_6),
             folder_path=self.folderLineEdit_2,
             destination=self.destinationLineEdit_3,
             table=self.data_frame,
             column1=self.comboBox_1,
             column2=self.comboBox_2
        )
        self.run_batch_process(self.cropper.mapping_crop, job)

    def crop_frame(self) -> None:
        job = Job(
            self.widthLineEdit,
            self.heightLineEdit,
            self.exposureCheckBox_4,
            self.mfaceCheckBox_4,
            self.tiltCheckBox_4,
            self.sensitivityDial,
            self.faceDial,
            self.gammaDial, 
            self.topDial, 
            self.bottomDial, 
            self.leftDial, 
            self.rightDial,
            (self.radioButton_1, self.radioButton_2, self.radioButton_3,
             self.radioButton_4, self.radioButton_5, self.radioButton_6),
             video_path=self.videoLineEdit,
             destination=self.destinationLineEdit_4,
        )
        self.cropper.crop_frame(job, self.positionLabel, self.timelineSlider)

    def video_process(self) -> None:
        job = Job(
            self.widthLineEdit,
            self.heightLineEdit,
            self.exposureCheckBox_4,
            self.mfaceCheckBox_4,
            self.tiltCheckBox_4,
            self.sensitivityDial,
            self.faceDial,
            self.gammaDial, 
            self.topDial, 
            self.bottomDial, 
            self.leftDial, 
            self.rightDial,
            (self.radioButton_1, self.radioButton_2, self.radioButton_3,
             self.radioButton_4, self.radioButton_5, self.radioButton_6),
             video_path=self.videoLineEdit,
             destination=self.destinationLineEdit_4,
             start_position=self.video.start_position,
             stop_position=self.video.stop_position
        )
        self.run_batch_process(self.cropper.extract_frames, job)

    def update_progress_1(self, value: int) -> None:
        self.progressBar_1.setValue(value)
        QtWidgets.QApplication.processEvents()

    def update_progress_2(self, value: int) -> None:
        self.progressBar_2.setValue(value)
        QtWidgets.QApplication.processEvents()

    def update_progress_3(self, value: int) -> None:
        self.progressBar_3.setValue(value)
        QtWidgets.QApplication.processEvents()
