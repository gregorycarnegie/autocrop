from pathlib import Path
from threading import Thread
from typing import Literal

import numpy as np
from PyQt6 import QtCore, QtGui, QtMultimedia, QtMultimediaWidgets, QtWidgets

from core import window_functions as wf
from core.croppers import VideoCropper
from core.enums import FunctionType, GuiIcon
from file_types import Video
from line_edits import LineEditState, NumberLineEdit, PathLineEdit, PathType
from .ui_crop_batch_widget import UiCropBatchWidget
from .ui_media_control_widget import UiMediaControlWidget


class UiVideoTabWidget(UiCropBatchWidget):
    def __init__(self, crop_worker: VideoCropper, object_name: str, parent: QtWidgets.QWidget) -> None:
        super().__init__(object_name, parent)
        self.crop_worker = crop_worker
        self.vol_cache = 70
        self.rewind_timer = QtCore.QTimer()
        self.default_directory = Video.default_directory
        self.player = QtMultimedia.QMediaPlayer()
        self.audio = QtMultimedia.QAudioOutput()
        self.start_position, self.stop_position, self.step = .0, .0, 100
        self.speed = 0
        self.reverse = 0

        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                                            QtWidgets.QSizePolicy.Policy.Expanding)
        size_policy.setHorizontalStretch(0)
        size_policy.setVerticalStretch(0)
        size_policy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(size_policy)

        self.toolBox = QtWidgets.QToolBox(self)
        self.toolBox.setObjectName(u"toolBox")
        self.inputLineEdit = self.create_str_line_edit(u"inputLineEdit", PathType.VIDEO)
        self.inputLineEdit.setParent(self.page_1)

        self.horizontalLayout_2.addWidget(self.inputLineEdit)

        self.inputButton.setParent(self.page_1)
        icon = wf.create_button_icon(GuiIcon.CLAPPERBOARD)
        self.inputButton.setIcon(icon)

        self.horizontalLayout_2.addWidget(self.inputButton)
        self.horizontalLayout_2.setStretch(0, 1)

        self.verticalLayout_200.addLayout(self.horizontalLayout_2)

        self.frame_1 = wf.create_frame(u"frame_1", self.page_1, self.size_policy2)
        self.verticalLayout_9 = wf.setup_vbox(u"verticalLayout_9", self.frame_1)
        self.muteButton_1 = QtWidgets.QPushButton(self.frame_1)
        self.muteButton_1.setObjectName(u"muteButton_1")
        self.muteButton_1.setMinimumSize(QtCore.QSize(30, 30))
        self.muteButton_1.setMaximumSize(QtCore.QSize(30, 30))
        self.muteButton_1.setBaseSize(QtCore.QSize(30, 30))
        icon1 = wf.create_button_icon(GuiIcon.MULTIMEDIA_MUTE)
        self.muteButton_1.setIcon(icon1)

        self.horizontalLayout_1.addWidget(self.muteButton_1)

        self.volumeSlider_1 = QtWidgets.QSlider(self.frame_1)
        self.volumeSlider_1.setObjectName(u"volumeSlider_1")
        size_policy3 = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        size_policy3.setHorizontalStretch(0)
        size_policy3.setVerticalStretch(0)
        size_policy3.setHeightForWidth(self.volumeSlider_1.sizePolicy().hasHeightForWidth())
        self.volumeSlider_1.setSizePolicy(size_policy3)
        self.volumeSlider_1.setMinimumSize(QtCore.QSize(0, 30))
        self.volumeSlider_1.setMaximumSize(QtCore.QSize(16_777_215, 30))
        self.volumeSlider_1.setMinimum(-1)
        self.volumeSlider_1.setMaximum(100)
        self.volumeSlider_1.setValue(70)
        self.volumeSlider_1.setOrientation(QtCore.Qt.Orientation.Horizontal)

        self.horizontalLayout_1.addWidget(self.volumeSlider_1)

        self.positionLabel_1 = QtWidgets.QLabel(self.frame_1)
        self.positionLabel_1.setObjectName(u"positionLabel_1")
        size_policy4 = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed)
        size_policy4.setHorizontalStretch(0)
        size_policy4.setVerticalStretch(0)
        size_policy4.setHeightForWidth(self.positionLabel_1.sizePolicy().hasHeightForWidth())
        self.positionLabel_1.setSizePolicy(size_policy4)
        self.positionLabel_1.setMinimumSize(QtCore.QSize(0, 30))
        self.positionLabel_1.setMaximumSize(QtCore.QSize(16_777_215, 30))

        self.horizontalLayout_1.addWidget(self.positionLabel_1)

        self.timelineSlider_1 = QtWidgets.QSlider(self.frame_1)
        self.timelineSlider_1.setObjectName(u"timelineSlider_1")
        size_policy5 = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        size_policy5.setHorizontalStretch(1)
        size_policy5.setVerticalStretch(0)
        size_policy5.setHeightForWidth(self.timelineSlider_1.sizePolicy().hasHeightForWidth())
        self.timelineSlider_1.setSizePolicy(size_policy5)
        self.timelineSlider_1.setMinimumSize(QtCore.QSize(0, 30))
        self.timelineSlider_1.setMaximumSize(QtCore.QSize(16_777_215, 30))
        self.timelineSlider_1.setOrientation(QtCore.Qt.Orientation.Horizontal)

        self.horizontalLayout_1.addWidget(self.timelineSlider_1)

        self.durationLabel_1 = QtWidgets.QLabel(self.frame_1)
        self.durationLabel_1.setObjectName(u"durationLabel_1")
        size_policy4.setHeightForWidth(self.durationLabel_1.sizePolicy().hasHeightForWidth())
        self.durationLabel_1.setSizePolicy(size_policy4)
        self.durationLabel_1.setMinimumSize(QtCore.QSize(0, 30))
        self.durationLabel_1.setMaximumSize(QtCore.QSize(16_777_215, 30))

        self.horizontalLayout_1.addWidget(self.durationLabel_1)

        self.horizontalSpacer_1 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Policy.Fixed,
                                                        QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout_1.addItem(self.horizontalSpacer_1)

        self.mfaceCheckBox.setParent(self.frame_1)
        self.horizontalLayout_1.addWidget(self.mfaceCheckBox)

        self.tiltCheckBox.setParent(self.frame_1)
        self.horizontalLayout_1.addWidget(self.tiltCheckBox)

        self.exposureCheckBox.setParent(self.frame_1)
        self.horizontalLayout_1.addWidget(self.exposureCheckBox)

        self.horizontalLayout_1.setStretch(0, 1)
        self.horizontalLayout_1.setStretch(1, 3)

        self.verticalLayout_9.addLayout(self.horizontalLayout_1)

        self.videoWidget = QtMultimediaWidgets.QVideoWidget(self.frame_1)
        self.videoWidget.setObjectName(u"videoWidget")
        self.size_policy2.setHeightForWidth(self.videoWidget.sizePolicy().hasHeightForWidth())
        self.videoWidget.setSizePolicy(self.size_policy2)
        self.videoWidget.setMinimumSize(QtCore.QSize(200, 200))
        self.videoWidget.setStyleSheet(u"background: #1f2c33")

        self.verticalLayout_9.addWidget(self.videoWidget)

        self.mediacontrolWidget_1 = UiMediaControlWidget(self.frame_1, self.player, self.crop_worker)
        self.mediacontrolWidget_1.setObjectName(u"mediacontrolWidget_1")

        self.verticalLayout_9.addWidget(self.mediacontrolWidget_1)

        self.progressBar.setParent(self.frame_1)

        self.verticalLayout_9.addWidget(self.progressBar)
        self.verticalLayout_9.setStretch(1, 1)

        self.verticalLayout_200.addWidget(self.frame_1)

        self.destinationLineEdit.setParent(self.page_1)

        self.horizontalLayout_3.addWidget(self.destinationLineEdit)

        self.destinationButton.setParent(self.page_1)
        self.destinationButton.setIcon(self.folder_icon)

        self.horizontalLayout_3.addWidget(self.destinationButton)
        self.horizontalLayout_3.setStretch(0, 1)

        self.verticalLayout_200.addLayout(self.horizontalLayout_3)

        self.toolBox.addItem(self.page_1, u"Video Player")
        self.frame_2 = wf.create_frame(u"frame_2", self.page_2, self.size_policy2)
        self.verticalLayout_10 = wf.setup_vbox(u"verticalLayout_10", self.frame_2)
        self.horizontalLayout_5 = wf.setup_hbox(u'horizontalLayout_5')
        self.muteButton_2 = QtWidgets.QPushButton(self.frame_2)
        self.muteButton_2.setObjectName(u"muteButton_2")
        self.muteButton_2.setMinimumSize(QtCore.QSize(30, 30))
        self.muteButton_2.setMaximumSize(QtCore.QSize(30, 30))
        self.muteButton_2.setBaseSize(QtCore.QSize(30, 30))
        self.muteButton_2.setIcon(icon1)

        self.horizontalLayout_5.addWidget(self.muteButton_2)

        self.volumeSlider_2 = QtWidgets.QSlider(self.frame_2)
        self.volumeSlider_2.setObjectName(u"volumeSlider_2")
        size_policy3.setHeightForWidth(self.volumeSlider_2.sizePolicy().hasHeightForWidth())
        self.volumeSlider_2.setSizePolicy(size_policy3)
        self.volumeSlider_2.setMinimumSize(QtCore.QSize(0, 30))
        self.volumeSlider_2.setMaximumSize(QtCore.QSize(16_777_215, 30))
        self.volumeSlider_2.setMinimum(-1)
        self.volumeSlider_2.setMaximum(100)
        self.volumeSlider_2.setValue(70)
        self.volumeSlider_2.setOrientation(QtCore.Qt.Orientation.Horizontal)

        self.horizontalLayout_5.addWidget(self.volumeSlider_2)

        self.positionLabel_2 = QtWidgets.QLabel(self.frame_2)
        self.positionLabel_2.setObjectName(u"positionLabel_2")
        size_policy4.setHeightForWidth(self.positionLabel_2.sizePolicy().hasHeightForWidth())
        self.positionLabel_2.setSizePolicy(size_policy4)
        self.positionLabel_2.setMinimumSize(QtCore.QSize(0, 30))
        self.positionLabel_2.setMaximumSize(QtCore.QSize(16_777_215, 30))

        self.horizontalLayout_5.addWidget(self.positionLabel_2)

        self.timelineSlider_2 = QtWidgets.QSlider(self.frame_2)
        self.timelineSlider_2.setObjectName(u"timelineSlider_2")
        size_policy5.setHeightForWidth(self.timelineSlider_2.sizePolicy().hasHeightForWidth())
        self.timelineSlider_2.setSizePolicy(size_policy5)
        self.timelineSlider_2.setMinimumSize(QtCore.QSize(0, 30))
        self.timelineSlider_2.setMaximumSize(QtCore.QSize(16_777_215, 30))
        self.timelineSlider_2.setOrientation(QtCore.Qt.Orientation.Horizontal)

        self.horizontalLayout_5.addWidget(self.timelineSlider_2)

        self.durationLabel_2 = QtWidgets.QLabel(self.frame_2)
        self.durationLabel_2.setObjectName(u"durationLabel_2")
        size_policy4.setHeightForWidth(self.durationLabel_2.sizePolicy().hasHeightForWidth())
        self.durationLabel_2.setSizePolicy(size_policy4)
        self.durationLabel_2.setMinimumSize(QtCore.QSize(0, 30))
        self.durationLabel_2.setMaximumSize(QtCore.QSize(16_777_215, 30))

        self.horizontalLayout_5.addWidget(self.durationLabel_2)

        self.horizontalLayout_5.setStretch(0, 1)
        self.horizontalLayout_5.setStretch(1, 3)

        self.verticalLayout_10.addLayout(self.horizontalLayout_5)

        self.toggleCheckBox.setParent(self.frame_2)
        self.horizontalLayout_4.addWidget(self.toggleCheckBox)

        self.horizontalSpacer_2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding,
                                                        QtWidgets.QSizePolicy.Policy.Minimum)

        self.horizontalLayout_4.addItem(self.horizontalSpacer_2)

        self.mfaceCheckBox_2 = self.create_checkbox(u"mfaceCheckBox_2")
        self.mfaceCheckBox_2.setParent(self.frame_2)
        self.horizontalLayout_4.addWidget(self.mfaceCheckBox_2)

        self.tiltCheckBox_2 = self.create_checkbox(u"tiltCheckBox_2")
        self.mfaceCheckBox_2.setParent(self.frame_2)
        self.horizontalLayout_4.addWidget(self.tiltCheckBox_2)

        self.exposureCheckBox_2 = self.create_checkbox(u"exposureCheckBox_2")
        self.mfaceCheckBox_2.setParent(self.frame_2)
        self.horizontalLayout_4.addWidget(self.exposureCheckBox_2)

        self.horizontalLayout_4.setStretch(1, 20)

        self.verticalLayout_10.addLayout(self.horizontalLayout_4)

        self.imageWidget.setParent(self.frame_2)

        self.verticalLayout_10.addWidget(self.imageWidget)

        self.mediacontrolWidget_2 = UiMediaControlWidget(self.frame_2, self.player, self.crop_worker)
        self.mediacontrolWidget_2.setObjectName(u"mediacontrolWidget_2")

        self.verticalLayout_10.addWidget(self.mediacontrolWidget_2)

        self.progressBar_2 = self.create_progress_bar(u"progressBar_2", self.frame_2)

        self.verticalLayout_10.addWidget(self.progressBar_2)

        self.verticalLayout_300.addWidget(self.frame_2)

        self.toolBox.addItem(self.page_2, u"Crop View")

        self.verticalLayout_100.addWidget(self.toolBox)

        # Connections
        self.exposureCheckBox.toggled.connect(self.exposureCheckBox_2.setChecked)
        self.tiltCheckBox.toggled.connect(self.tiltCheckBox_2.setChecked)
        self.mfaceCheckBox.toggled.connect(self.mfaceCheckBox_2.setChecked)
        self.exposureCheckBox_2.toggled.connect(self.exposureCheckBox.setChecked)
        self.tiltCheckBox_2.toggled.connect(self.tiltCheckBox.setChecked)
        self.mfaceCheckBox_2.toggled.connect(self.mfaceCheckBox.setChecked)

        self.mediacontrolWidget_1.startmarkerButton.clicked.connect(lambda: self.set_start_position(self.mediacontrolWidget_1.selectStartMarkerButton))
        self.mediacontrolWidget_1.endmarkerButton.clicked.connect(lambda: self.set_stop_position(self.mediacontrolWidget_1.selectEndMarkerButton))
        self.mediacontrolWidget_1.startmarkerButton.clicked.connect(lambda: self.set_start_position(self.mediacontrolWidget_2.selectStartMarkerButton))
        self.mediacontrolWidget_1.endmarkerButton.clicked.connect(lambda: self.set_stop_position(self.mediacontrolWidget_2.selectEndMarkerButton))
        self.mediacontrolWidget_2.startmarkerButton.clicked.connect(lambda: self.set_start_position(self.mediacontrolWidget_1.selectStartMarkerButton))
        self.mediacontrolWidget_2.endmarkerButton.clicked.connect(lambda: self.set_stop_position(self.mediacontrolWidget_1.selectEndMarkerButton))
        self.mediacontrolWidget_2.startmarkerButton.clicked.connect(lambda: self.set_start_position(self.mediacontrolWidget_2.selectStartMarkerButton))
        self.mediacontrolWidget_2.endmarkerButton.clicked.connect(lambda: self.set_stop_position(self.mediacontrolWidget_2.selectEndMarkerButton))

        self.crop_worker.progress.connect(self.update_progress)
        self.volumeSlider_1.sliderMoved.connect(self.volume_slider_changed)
        self.volumeSlider_2.sliderMoved.connect(self.volume_slider_changed)
        self.timelineSlider_1.sliderMoved.connect(self.player_slider_changed)
        self.timelineSlider_2.sliderMoved.connect(self.player_slider_changed)
        self.timelineSlider_1.sliderMoved.connect(self.timelineSlider_2.setSliderPosition)
        self.timelineSlider_2.sliderMoved.connect(self.timelineSlider_1.setSliderPosition)
        self.inputButton.clicked.connect(lambda: self.open_video())
        self.destinationButton.clicked.connect(lambda: self.open_path(self.destinationLineEdit))

        for control in [self.mediacontrolWidget_1, self.mediacontrolWidget_2]:
            control.cropButton.clicked.connect(lambda: self.crop_frame())
            control.videocropButton.clicked.connect(lambda: self.video_process())
            control.cancelButton.clicked.connect(lambda: self.crop_worker.terminate())
            control.cancelButton.clicked.connect(
                lambda: self.cancel_button_operation(control.cancelButton,
                                                     control.videocropButton,
                                                     control.cropButton))

            control.playButton.clicked.connect(lambda: self.change_playback_state())
            control.stopButton.clicked.connect(lambda: self.stop_playback())
            control.stepbackButton.clicked.connect(lambda: self.step_back())
            control.stepfwdButton.clicked.connect(lambda: self.step_forward())
            control.fastfwdButton.clicked.connect(lambda: self.fast_forward())
            control.goto_beginingButton.clicked.connect(lambda: self.goto_beginning())
            control.goto_endButton.clicked.connect(lambda: self.goto_end())
            control.selectStartMarkerButton.clicked.connect(
                lambda: self.goto(control.selectStartMarkerButton))
            control.selectEndMarkerButton.clicked.connect(
                lambda: self.goto(control.selectEndMarkerButton))

        self.muteButton_1.clicked.connect(lambda: self.volume_mute())
        self.muteButton_2.clicked.connect(lambda: self.volume_mute())

        self.connect_input_widgets(self.controlWidget.widthLineEdit, self.controlWidget.heightLineEdit,
                                   self.destinationLineEdit, self.inputLineEdit,
                                   self.exposureCheckBox, self.mfaceCheckBox, self.tiltCheckBox)

        # Media connections
        self.audio.mutedChanged.connect(lambda: self.change_audio_icon())

        # self.player.playbackStateChanged.connect(lambda: self.change_playback_icons())
        self.player.positionChanged.connect(self.position_changed)
        self.player.durationChanged.connect(self.duration_changed)
        self.player.errorOccurred.connect(self.player_error_occurred) #??????

        # Connect crop worker
        self.connect_crop_worker()

        self.toggleCheckBox.toggled.connect(self.controlWidget.setVisible)

        # Connect crop worker
        self.connect_crop_worker()

        self.retranslateUi()

        self.toolBox.setCurrentIndex(0)

        QtCore.QMetaObject.connectSlotsByName(self)

    # setupUi

    def retranslateUi(self):
        self.setWindowTitle(QtCore.QCoreApplication.translate("self", u"Form", None))
        self.inputLineEdit.setPlaceholderText(
            QtCore.QCoreApplication.translate("self", u"Choose the video you want to crop", None))
        self.inputButton.setText(QtCore.QCoreApplication.translate("self", u"Open Video", None))
        self.muteButton_1.setText("")
        self.positionLabel_1.setText(QtCore.QCoreApplication.translate("self", u"00:00:00", None))
        self.durationLabel_1.setText(QtCore.QCoreApplication.translate("self", u"00:00:00", None))
        self.mfaceCheckBox.setText(QtCore.QCoreApplication.translate("self", u"Multi-Face", None))
        self.tiltCheckBox.setText(QtCore.QCoreApplication.translate("self", u"Autotilt", None))
        self.exposureCheckBox.setText(QtCore.QCoreApplication.translate("self", u"Autocorrect", None))
        self.destinationLineEdit.setPlaceholderText(
            QtCore.QCoreApplication.translate("self", u"Choose where you want to save the cropped images", None))
        self.destinationButton.setText(QtCore.QCoreApplication.translate("self", u"Destination Folder", None))
        self.toolBox.setItemText(self.toolBox.indexOf(self.page_1),
                                 QtCore.QCoreApplication.translate("self", u"Video Player", None))
        self.muteButton_2.setText("")
        self.positionLabel_2.setText(QtCore.QCoreApplication.translate("self", u"00:00:00", None))
        self.durationLabel_2.setText(QtCore.QCoreApplication.translate("self", u"00:00:00", None))
        self.toggleCheckBox.setText(QtCore.QCoreApplication.translate("self", u"Toggle Settings", None))
        self.mfaceCheckBox_2.setText(QtCore.QCoreApplication.translate("self", u"Multi-Face", None))
        self.tiltCheckBox_2.setText(QtCore.QCoreApplication.translate("self", u"Autotilt", None))
        self.exposureCheckBox_2.setText(QtCore.QCoreApplication.translate("self", u"Autocorrect", None))
        self.toolBox.setItemText(self.toolBox.indexOf(self.page_2),
                                 QtCore.QCoreApplication.translate("self", u"Crop View", None))

    def update_progress(self, x: int, y:int) -> None:
        """Only sublasses of the CropBatchWidget class should implement this method"""
        self.progressBar.setValue(int(self.PROGRESSBAR_STEPS * x / y))
        self.progressBar_2.setValue(int(self.PROGRESSBAR_STEPS * x / y))
        QtWidgets.QApplication.processEvents()

    def connect_crop_worker(self) -> None:
        widget_list = (self.controlWidget.widthLineEdit, self.controlWidget.heightLineEdit,
                       self.controlWidget.sensitivityDial, self.controlWidget.fpctDial, self.controlWidget.gammaDial,
                       self.controlWidget.topDial, self.controlWidget.bottomDial, self.controlWidget.leftDial,
                       self.controlWidget.rightDial, self.inputLineEdit, self.destinationLineEdit,
                       self.destinationButton, self.controlWidget.radioButton_none, self.controlWidget.radioButton_bmp,
                       self.controlWidget.radioButton_jpg, self.controlWidget.radioButton_png,
                       self.controlWidget.radioButton_tiff, self.controlWidget.radioButton_webp, self.exposureCheckBox,
                       self.mfaceCheckBox, self.tiltCheckBox, self.exposureCheckBox_2, self.mfaceCheckBox_2,
                       self.tiltCheckBox_2)

        self.crop_worker.started.connect(lambda: wf.disable_widget(*widget_list))  # Video start connection
        self.crop_worker.finished.connect(lambda: wf.enable_widget(*widget_list))  # Video end connection

        self.crop_worker.finished.connect(lambda: wf.show_message_box(self.destination))
        self.crop_worker.progress.connect(self.update_progress)

    def player_error_occurred(self, error: QtMultimedia.QMediaPlayer.Error) -> None:
        match error:
            case QtMultimedia.QMediaPlayer.Error.NoError:
                pass
            case _:
                self.stop_playback()
                wf.show_error_box(f'{error.name} occurred while loading the video. ',
                                  'Please check the video file path and try again.')

    def setup_label(self,
                    name: GuiIcon = Literal[GuiIcon.MULTIMEDIA_LABEL_A, GuiIcon.MULTIMEDIA_LABEL_B]) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(parent=self)
        size_policy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        size_policy.setHorizontalStretch(0)
        size_policy.setVerticalStretch(0)
        size_policy.setHeightForWidth(label.sizePolicy().hasHeightForWidth())
        label.setSizePolicy(size_policy)
        label.setMaximumSize(QtCore.QSize(14, 14))
        label.setText('')
        label.setPixmap(QtGui.QPixmap(name))
        label.setScaledContents(True)
        match name:
            case GuiIcon.MULTIMEDIA_LABEL_A:
                label.setObjectName(u'label_A')
            case GuiIcon.MULTIMEDIA_LABEL_B:
                label.setObjectName(u'label_B')
        return label

    def open_path(self, line_edit: PathLineEdit) -> None:
        self.player.pause()
        super().open_path(line_edit)
        self.player.play()

    def open_video(self) -> None:
        self.check_playback_state()
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open Video', self.default_directory,
                                                             Video.type_string())
        self.inputLineEdit.setText(file_name)
        if self.inputLineEdit.state is LineEditState.INVALID_INPUT:
            return
        self.player.setSource(QtCore.QUrl.fromLocalFile(file_name))
        self.reset_video_widgets()

    def open_dropped_video(self) -> None:
        self.player.setSource(QtCore.QUrl.fromLocalFile(self.inputLineEdit.text()))
        self.reset_video_widgets()

    def reset_video_widgets(self):
        self.create_media_player()
        self.mediacontrolWidget_1.playButton.setEnabled(True)
        self.mediacontrolWidget_2.playButton.setEnabled(True)
        self.timelineSlider_1.setEnabled(True)
        self.timelineSlider_2.setEnabled(True)

    def change_audio_icon(self) -> None:
        if self.audio.isMuted():
            self.muteButton_1.setIcon(QtGui.QIcon(GuiIcon.MULTIMEDIA_UNMUTE))
            self.muteButton_2.setIcon(QtGui.QIcon(GuiIcon.MULTIMEDIA_UNMUTE))
        else:
            self.muteButton_1.setIcon(QtGui.QIcon(GuiIcon.MULTIMEDIA_MUTE))
            self.muteButton_2.setIcon(QtGui.QIcon(GuiIcon.MULTIMEDIA_MUTE))

    def playback_bool(self,
                      a0: QtMultimedia.QMediaPlayer.PlaybackState = QtMultimedia.QMediaPlayer.PlaybackState.PausedState,
                      a1: QtMultimedia.QMediaPlayer.PlaybackState = QtMultimedia.QMediaPlayer.PlaybackState.StoppedState) -> \
            tuple[bool, bool]:
        """Returns a tuple of bools comparing the playback state to the Class attributes of
        PyQt6.QtMultimedia.QMediaPlayer.PlaybackState"""
        return self.player.playbackState() == a0, self.player.playbackState() == a1

    def check_playback_state(self) -> None:
        """Stops playback if in the paused state or playing state"""
        x, y = self.playback_bool(a1=QtMultimedia.QMediaPlayer.PlaybackState.PlayingState)
        if x ^ y:
            self.stop_playback()

    def change_playback_state(self):
        match self.player.playbackState():
            case QtMultimedia.QMediaPlayer.PlaybackState.PlayingState:
                self.player.pause()
                self.speed = 0
            case _:
                self.player.play()

    def create_media_player(self) -> None:
        self.player.setAudioOutput(self.audio)
        self.player.setVideoOutput(self.videoWidget)

    def stop_playback(self) -> None:
        self.timelineSlider_1.setDisabled(True)
        self.timelineSlider_2.setDisabled(True)
        # Check paused or playing
        x, y = self.playback_bool(a1=QtMultimedia.QMediaPlayer.PlaybackState.PlayingState)
        if x ^ y:
            self.player.stop()

    def fast_forward(self) -> None:
        # Check paused or stopped
        x, y = self.playback_bool()
        if x ^ y:
            return
        video_speeds = np.array([1.0, 1.25, 1.5, 1.75, 2.0])
        self.reverse = 0
        self.speed += 1
        if self.speed > video_speeds.size - 1:
            self.player.setPlaybackRate(float(video_speeds[-1]))
        else:
            self.player.setPlaybackRate(float(video_speeds[self.speed]))

    def step_forward(self):
        if (new_position := self.player.position() + 10_000) >= self.player.duration():
            self.player.setPosition(self.player.duration())
        else:
            self.player.setPosition(new_position)

    def step_back(self):
        if (new_position := self.player.position() - 10_000) <= 0:
            self.player.setPosition(0)
        else:
            self.player.setPosition(new_position)

    def position_changed(self, position: int) -> None:
        def callback():
            for slider, label in zip((self.timelineSlider_1, self.timelineSlider_2),(self.positionLabel_1, self.positionLabel_2)):
                if slider.maximum() != self.player.duration():
                    slider.setMaximum(self.player.duration())

                slider.blockSignals(True)
                slider.setValue(position)
                slider.blockSignals(False)
                label.setText(wf.get_qtime(position).toString())

        thread = Thread(target=callback)
        thread.start()

    def duration_changed(self, duration: int) -> None:
        self.timelineSlider_1.setMaximum(duration)
        self.timelineSlider_2.setMaximum(duration)
        if duration >= 0:
            qtime = wf.get_qtime(self.player.duration()).toString()
            self.durationLabel_1.setText(qtime)
            self.durationLabel_2.setText(qtime)
            self.mediacontrolWidget_1.selectEndMarkerButton.setText(qtime)
            self.mediacontrolWidget_2.selectEndMarkerButton.setText(qtime)

    def player_slider_changed(self, position: int) -> None:
        self.player.setPosition(position)

    def volume_slider_changed(self, position: int) -> None:
        self.audio.setVolume(position)
        self.vol_cache = position

    def volume_mute(self) -> None:
        if self.audio.isMuted():
            self.audio.setMuted(False)
            self.volumeSlider_1.setValue(self.vol_cache)
            self.volumeSlider_2.setValue(self.vol_cache)
        else:
            self.audio.setMuted(True)
            self.volumeSlider_2.setValue(0)
            self.volumeSlider_2.setValue(0)

    def goto_beginning(self) -> None:
        self.player.setPosition(0)

    def goto_end(self) -> None:
        self.player.setPosition(self.player.duration())

    @staticmethod
    def set_marker_time(button: QtWidgets.QPushButton, flag: bool, time_value: float, position: float) -> None:
        if flag:
            position = time_value
            wf.set_marker_time(button, position)     
    
    def set_start_position(self, button: QtWidgets.QPushButton) -> None:
        x = (time_value := self.timelineSlider_1.value() * .001) < self.stop_position
        y = self.start_position == .0 and self.stop_position == .0
        self.set_marker_time(button, x | y, time_value, self.start_position)

    def set_stop_position(self, button: QtWidgets.QPushButton) -> None:
        x = (time_value := self.timelineSlider_1.value() * .001) > self.start_position
        y = self.start_position == .0 and self.stop_position == .0
        self.set_marker_time(button, x | y, time_value, self.stop_position)

    def goto(self, marker_button: QtWidgets.QPushButton) -> None:
        m = np.array(marker_button.text().split(':')).astype(int)
        if (x := np.sum([60 ** (2 - i) * m[i] for i in np.arange(m.size)]) * 1_000) >= self.player.duration():
            self.player.setPosition(self.player.duration())
        elif x == 0:
            self.player.setPosition(0)
        else:
            self.player.setPosition(int(x))

    def connect_input_widgets(self, *input_widgets: QtWidgets.QWidget) -> None:
        for input_widget in input_widgets:
            match input_widget:
                case NumberLineEdit() | PathLineEdit():
                    input_widget.textChanged.connect(lambda: self.disable_buttons())
                case QtWidgets.QCheckBox():
                    self.connect_checkbox(input_widget)
                case _:
                    pass

    def disable_buttons(self) -> None:
        wf.change_widget_state(
            wf.all_filled(self.inputLineEdit, self.destinationLineEdit, self.controlWidget.widthLineEdit,
                          self.controlWidget.heightLineEdit),
            self.mediacontrolWidget_1.cropButton, self.mediacontrolWidget_1.videocropButton,
            self.mediacontrolWidget_2.cropButton, self.mediacontrolWidget_2.videocropButton)

    def crop_frame(self) -> None:
        def callback():
            self.player.pause()
            job = self.create_job(FunctionType.FRAME,
                                  video_path=Path(self.inputLineEdit.text()),
                                  destination=Path(self.destinationLineEdit.text()))
            self.crop_worker.crop_frame(job, self.positionLabel_1, self.timelineSlider_1)

        if Path(self.inputLineEdit.text()).parent == Path(self.destinationLineEdit.text()):
            match wf.show_warning(FunctionType.FRAME):
                case QtWidgets.QMessageBox.StandardButton.Yes:
                    callback()
                case _:
                    return
        callback()

    def video_process(self) -> None:
        x = self.mediacontrolWidget_1.selectStartMarkerButton.text()
        y = self.mediacontrolWidget_1.selectEndMarkerButton.text()

        def callback():
            self.player.pause()
            job = self.create_job(FunctionType.VIDEO,
                                  video_path=Path(self.inputLineEdit.text()),
                                  destination=Path(self.destinationLineEdit.text()),
                                  start_position=wf.pos_from_marker(x),
                                  stop_position=wf.pos_from_marker(y))
            self.run_batch_process(job, function=self.crop_worker.extract_frames,
                                   reset_worker_func=lambda: self.crop_worker.reset_task())

        if Path(self.inputLineEdit.text()).parent == Path(self.destinationLineEdit.text()):
            match wf.show_warning(FunctionType.VIDEO):
                case QtWidgets.QMessageBox.StandardButton.Yes:
                    callback()
                case _:
                    return
        callback()
