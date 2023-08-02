from pathlib import Path
from threading import Thread
from typing import Optional, Tuple, Union

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets, QtMultimedia
from PyQt6.QtMultimediaWidgets import QVideoWidget

from line_edits import PathLineEdit, PathType, NumberLineEdit, LineEditState
from .cropper import Cropper
from .custom_crop_widget import CustomCropWidget
from .custom_dial_widget import CustomDialWidget
from .enums import FunctionType
from .ext_widget import ExtWidget
from .f_type_photo import Photo
from .window_functions import change_widget_state, disable_widget, enable_widget, show_message_box, uncheck_boxes, \
    create_media_button


class CropVideoWidget(CustomCropWidget):
    def __init__(self, crop_worker: Cropper,
                 width_line_edit: NumberLineEdit,
                 height_line_edit: NumberLineEdit,
                 ext_widget: ExtWidget,
                 sensitivity_dial_area: CustomDialWidget,
                 face_dial_area: CustomDialWidget,
                 gamma_dial_area: CustomDialWidget,
                 top_dial_area: CustomDialWidget,
                 bottom_dial_area: CustomDialWidget,
                 left_dial_area: CustomDialWidget,
                 right_dial_area: CustomDialWidget,
                 parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(crop_worker, width_line_edit, height_line_edit, ext_widget, sensitivity_dial_area,
                         face_dial_area, gamma_dial_area, top_dial_area, bottom_dial_area, left_dial_area,
                         right_dial_area, parent)
        self.vol_cache = 70
        self.rewind_timer = QtCore.QTimer()
        self.default_directory = f'{Path.home()}\\Videos'
        self.player = QtMultimedia.QMediaPlayer()
        self.audio = QtMultimedia.QAudioOutput()
        self.start_position, self.stop_position, self.step = 0.0, 0.0, 2
        self.speed = 0
        self.reverse = 0
        self.setObjectName('Form')
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self)
        self.verticalLayout_3.setObjectName('verticalLayout_3')
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName('horizontalLayout_3')
        self.videoLineEdit = PathLineEdit(path_type=PathType.VIDEO, parent=self)
        self.videoLineEdit.setMinimumSize(QtCore.QSize(0, 24))
        self.videoLineEdit.setMaximumSize(QtCore.QSize(16777215, 24))
        self.videoLineEdit.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhUrlCharactersOnly)
        self.videoLineEdit.setObjectName('videoLineEdit')
        self.horizontalLayout_3.addWidget(self.videoLineEdit)
        self.videoButton = QtWidgets.QPushButton(parent=self)
        self.videoButton.setMinimumSize(QtCore.QSize(124, 0))
        self.videoButton.setMaximumSize(QtCore.QSize(16777215, 24))
        icon = QtGui.QIcon()
        icon.addPixmap(
            QtGui.QPixmap('resources\\icons\\clapperboard.svg'), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.videoButton.setIcon(icon)
        self.videoButton.setObjectName('videoButton')
        self.horizontalLayout_3.addWidget(self.videoButton)
        self.verticalLayout_3.addLayout(self.horizontalLayout_3)
        self.frame = QtWidgets.QFrame(parent=self)
        self.frame.setStyleSheet('background: #1f2c33')
        self.frame.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.frame.setObjectName('frame')
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.frame)
        self.verticalLayout_2.setObjectName('verticalLayout_2')
        self.horizontalLayout_1 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_1.setObjectName('horizontalLayout_1')
        self.muteButton = QtWidgets.QPushButton(parent=self.frame)
        self.muteButton.setText('')
        icon1 = QtGui.QIcon()
        icon1.addPixmap(
            QtGui.QPixmap('resources\\icons\\multimedia_mute.svg'), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.muteButton.setIcon(icon1)
        self.muteButton.setObjectName('muteButton')
        self.horizontalLayout_1.addWidget(self.muteButton)
        self.volumeSlider = QtWidgets.QSlider(parent=self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.volumeSlider.sizePolicy().hasHeightForWidth())
        self.volumeSlider.setSizePolicy(sizePolicy)
        self.volumeSlider.setMinimum(-1)
        self.volumeSlider.setMaximum(100)
        self.volumeSlider.setProperty('value', 70)
        self.volumeSlider.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.volumeSlider.setObjectName('volumeSlider')
        self.horizontalLayout_1.addWidget(self.volumeSlider)
        self.positionLabel = QtWidgets.QLabel(parent=self.frame)
        self.positionLabel.setObjectName('positionLabel')
        self.horizontalLayout_1.addWidget(self.positionLabel)
        self.timelineSlider = QtWidgets.QSlider(parent=self.frame)
        self.timelineSlider.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.timelineSlider.setObjectName('timelineSlider')
        self.horizontalLayout_1.addWidget(self.timelineSlider)
        self.durationLabel = QtWidgets.QLabel(parent=self.frame)
        self.durationLabel.setObjectName('durationLabel')
        self.horizontalLayout_1.addWidget(self.durationLabel)
        spacerItem = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout_1.addItem(spacerItem)
        self.mfaceCheckBox = QtWidgets.QCheckBox(parent=self.frame)
        self.mfaceCheckBox.setStyleSheet(self.CHECKBOX_STYLESHEET)
        self.mfaceCheckBox.setObjectName('mfaceCheckBox')
        self.horizontalLayout_1.addWidget(self.mfaceCheckBox)
        self.tiltCheckBox = QtWidgets.QCheckBox(parent=self.frame)
        self.tiltCheckBox.setStyleSheet(self.CHECKBOX_STYLESHEET)
        self.tiltCheckBox.setObjectName('tiltCheckBox')
        self.horizontalLayout_1.addWidget(self.tiltCheckBox)
        self.exposureCheckBox = QtWidgets.QCheckBox(parent=self.frame)
        self.exposureCheckBox.setStyleSheet(self.CHECKBOX_STYLESHEET)
        self.exposureCheckBox.setObjectName('exposureCheckBox')
        self.horizontalLayout_1.addWidget(self.exposureCheckBox)
        self.verticalLayout_2.addLayout(self.horizontalLayout_1)
        self.videoWidget = QVideoWidget(parent=self.frame)
        self.videoWidget.setStyleSheet('background: #1f2c33')
        self.videoWidget.setObjectName('videoWidget')
        self.create_mediaPlayer()
        self.verticalLayout_2.addWidget(self.videoWidget)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName('horizontalLayout_2')
        self.cropButton = QtWidgets.QPushButton(parent=self.frame)
        self.cropButton.setMinimumSize(QtCore.QSize(0, 24))
        self.cropButton.setMaximumSize(QtCore.QSize(16777215, 24))
        self.cropButton.setText('')
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap('resources\\icons\\crop.svg'), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.cropButton.setIcon(icon2)
        self.cropButton.setObjectName('cropButton')
        self.horizontalLayout_2.addWidget(self.cropButton)
        self.videocropButton = QtWidgets.QPushButton(parent=self.frame)
        self.videocropButton.setMinimumSize(QtCore.QSize(0, 24))
        self.videocropButton.setMaximumSize(QtCore.QSize(16777215, 24))
        self.videocropButton.setText('')
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap('resources\\icons\\crop_video.svg'), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.videocropButton.setIcon(icon3)
        self.videocropButton.setObjectName('videocropButton')
        self.horizontalLayout_2.addWidget(self.videocropButton)
        self.cancelButton = QtWidgets.QPushButton(parent=self.frame)
        self.cancelButton.setMinimumSize(QtCore.QSize(0, 24))
        self.cancelButton.setMaximumSize(QtCore.QSize(16777215, 24))
        self.cancelButton.setText('')
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap('resources\\icons\\cancel.svg'), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.cancelButton.setIcon(icon4)
        self.cancelButton.setObjectName('cancelButton')
        self.horizontalLayout_2.addWidget(self.cancelButton)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.progressBar = QtWidgets.QProgressBar(parent=self.frame)
        self.progressBar.setMinimumSize(QtCore.QSize(0, 12))
        self.progressBar.setMaximumSize(QtCore.QSize(16777215, 12))
        self.progressBar.setProperty('value', 0)
        self.progressBar.setTextVisible(False)
        self.progressBar.setObjectName('progressBar')
        self.verticalLayout_2.addWidget(self.progressBar)
        self.verticalLayout_2.setStretch(0, 1)
        self.verticalLayout_2.setStretch(1, 10)
        self.verticalLayout_2.setStretch(2, 1)
        self.verticalLayout_2.setStretch(3, 1)
        self.verticalLayout_3.addWidget(self.frame)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName('horizontalLayout_4')
        self.destinationLineEdit = PathLineEdit(parent=self)
        self.destinationLineEdit.setMinimumSize(QtCore.QSize(0, 24))
        self.destinationLineEdit.setMaximumSize(QtCore.QSize(16777215, 24))
        self.destinationLineEdit.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhUrlCharactersOnly)
        self.destinationLineEdit.setObjectName('destinationLineEdit')
        self.horizontalLayout_4.addWidget(self.destinationLineEdit)
        self.destinationButton = QtWidgets.QPushButton(parent=self)
        self.destinationButton.setMinimumSize(QtCore.QSize(124, 24))
        self.destinationButton.setMaximumSize(QtCore.QSize(16777215, 24))
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap('resources\\icons\\folder.svg'), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.destinationButton.setIcon(icon5)
        self.destinationButton.setObjectName('destinationButton')
        self.horizontalLayout_4.addWidget(self.destinationButton)
        self.verticalLayout_3.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName('horizontalLayout_5')
        self.playButton = create_media_button('playButton', 'play', self.horizontalLayout_5, parent=self)
        self.stopButton = create_media_button('stopButton', 'stop', self.horizontalLayout_5, parent=self)
        self.stepbackButton = create_media_button('stepbackButton', 'left', self.horizontalLayout_5, parent=self)
        self.stepfwdButton = create_media_button('stepfwdButton', 'right', self.horizontalLayout_5, parent=self)
        self.fastfwdButton = create_media_button('fastfwdButton', 'fastfwd', self.horizontalLayout_5, parent=self)
        self.goto_beginingButton = create_media_button('goto_beginingButton', 'begining', self.horizontalLayout_5, parent=self)
        self.goto_endButton = create_media_button("goto_endButton", "end", self.horizontalLayout_5, parent=self)
        self.startmarkerButton = create_media_button('startmarkerButton', 'leftmarker', self.horizontalLayout_5, parent=self)
        self.endmarkerButton = create_media_button('endmarkerButton', 'rightmarker', self.horizontalLayout_5, parent=self)
        spacerItem1 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem1)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName('gridLayout')
        self.label_A = QtWidgets.QLabel(parent=self)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_A.sizePolicy().hasHeightForWidth())
        self.label_A.setSizePolicy(sizePolicy)
        self.label_A.setMaximumSize(QtCore.QSize(14, 14))
        self.label_A.setText('')
        self.label_A.setPixmap(QtGui.QPixmap('resources\\icons\\marker_label_a.svg'))
        self.label_A.setScaledContents(True)
        self.label_A.setObjectName('label_A')
        self.gridLayout.addWidget(self.label_A, 0, 0, 1, 1)
        self.selectStartMarkerButton = QtWidgets.QPushButton(parent=self)
        self.selectStartMarkerButton.setMinimumSize(QtCore.QSize(150, 0))
        self.selectStartMarkerButton.setObjectName('selectStartMarkerButton')
        self.gridLayout.addWidget(self.selectStartMarkerButton, 0, 1, 1, 1)
        self.label_B = QtWidgets.QLabel(parent=self)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_B.sizePolicy().hasHeightForWidth())
        self.label_B.setSizePolicy(sizePolicy)
        self.label_B.setMaximumSize(QtCore.QSize(14, 14))
        self.label_B.setText('')
        self.label_B.setPixmap(QtGui.QPixmap('resources\\icons\\marker_label_b.svg'))
        self.label_B.setScaledContents(True)
        self.label_B.setObjectName('label_B')
        self.gridLayout.addWidget(self.label_B, 1, 0, 1, 1)
        self.selectEndMarkerButton = QtWidgets.QPushButton(parent=self)
        self.selectEndMarkerButton.setMinimumSize(QtCore.QSize(150, 0))
        self.selectEndMarkerButton.setObjectName('selectEndMarkerButton')
        self.gridLayout.addWidget(self.selectEndMarkerButton, 1, 1, 1, 1)
        self.horizontalLayout_5.addLayout(self.gridLayout)
        self.verticalLayout_3.addLayout(self.horizontalLayout_5)
        self.verticalLayout_3.setStretch(0, 1)
        self.verticalLayout_3.setStretch(1, 19)
        self.verticalLayout_3.setStretch(3, 2)

        # Connections
        self.crop_worker.video_progress.connect(self.update_progress)
        self.player.positionChanged.connect(self.position_changed)
        self.player.durationChanged.connect(self.duration_changed)
        self.volumeSlider.sliderMoved.connect(self.volume_slider_changed)
        self.timelineSlider.sliderMoved.connect(self.player_slider_changed)
        self.videoButton.clicked.connect(lambda: self.open_video())
        self.destinationButton.clicked.connect(lambda: self.open_folder(self.destinationLineEdit))
        self.cropButton.clicked.connect(lambda: self.crop_frame())
        self.videocropButton.clicked.connect(lambda: self.video_process())
        self.cancelButton.clicked.connect(lambda: self.crop_worker.terminate(FunctionType.VIDEO))
        self.cancelButton.clicked.connect(lambda: self.cancel_button_operation(self.cancelButton, self.videocropButton, self.cropButton))

        self.connect_input_widgets(self.widthLineEdit, self.heightLineEdit, self.destinationLineEdit,
                                   self.exposureCheckBox, self.mfaceCheckBox, self.tiltCheckBox)
        
        # Media connections
        self.audio.mutedChanged.connect(lambda: self.change_audio_icon())
        self.player.playbackStateChanged.connect(
            lambda: self.change_media_widget_state(self.stopButton, self.stepbackButton, self.stepfwdButton,
                                                   self.fastfwdButton, self.goto_beginingButton, self.goto_endButton, self.startmarkerButton,
                                                   self.endmarkerButton, self.selectStartMarkerButton, self.selectEndMarkerButton,))
        self.player.playbackStateChanged.connect(lambda: self.change_playback_icons())

        # Video start connection
        self.crop_worker.video_started.connect(
            lambda: disable_widget(self.widthLineEdit, self.heightLineEdit, self.sensitivity_dialArea.dial,
                                   self.face_dialArea.dial, self.gamma_dialArea.dial, self.top_dialArea.dial,
                                   self.bottom_dialArea.dial, self.left_dialArea.dial, self.right_dialArea.dial,
                                   self.videoLineEdit, self.destinationLineEdit, self.destinationButton,
                                   self.extWidget.radioButton_1, self.extWidget.radioButton_2,
                                   self.extWidget.radioButton_3, self.extWidget.radioButton_4,
                                   self.extWidget.radioButton_5, self.extWidget.radioButton_6, self.cropButton,
                                   self.videocropButton, self.exposureCheckBox, self.mfaceCheckBox, self.tiltCheckBox,
                                   self.videocropButton, self.playButton, self.stopButton, self.stepbackButton,
                                   self.stepfwdButton, self.fastfwdButton, self.goto_beginingButton,
                                   self.goto_endButton, self.startmarkerButton, self.endmarkerButton,
                                   self.selectStartMarkerButton, self.selectEndMarkerButton))
        self.crop_worker.video_started.connect(lambda: enable_widget(self.cancelButton))

        # Video end connection
        self.crop_worker.video_finished.connect(
            lambda: enable_widget(self.widthLineEdit, self.heightLineEdit, self.sensitivity_dialArea.dial,
                                  self.face_dialArea.dial, self.gamma_dialArea.dial, self.top_dialArea.dial,
                                  self.bottom_dialArea.dial, self.left_dialArea.dial, self.right_dialArea.dial,
                                  self.videoLineEdit, self.destinationLineEdit, self.destinationButton,
                                  self.extWidget.radioButton_1, self.extWidget.radioButton_2,
                                  self.extWidget.radioButton_3, self.extWidget.radioButton_4,
                                  self.extWidget.radioButton_5, self.extWidget.radioButton_6, self.cropButton,
                                  self.videocropButton, self.exposureCheckBox, self.mfaceCheckBox, self.tiltCheckBox,
                                  self.videocropButton, self.playButton, self.stopButton, self.stepbackButton,
                                  self.stepfwdButton, self.fastfwdButton, self.goto_beginingButton,
                                  self.goto_endButton, self.startmarkerButton, self.endmarkerButton,
                                  self.selectStartMarkerButton, self.selectEndMarkerButton))
        self.crop_worker.video_finished.connect(lambda: disable_widget(self.cancelButton))
        self.crop_worker.video_finished.connect(lambda: show_message_box(self.destinationLineEdit))
        self.crop_worker.video_progress.connect(lambda: self.update_progress(self.crop_worker.bar_value_v))

        self.playButton.clicked.connect(lambda: self.change_playback_state())
        self.playButton.clicked.connect(
            lambda: change_widget_state(True, self.stopButton, self.stepbackButton,  self.stepfwdButton,
                                        self.fastfwdButton, self.goto_beginingButton,
                                        self.goto_endButton, self.startmarkerButton, self.endmarkerButton,
                                        self.selectEndMarkerButton, self.selectStartMarkerButton))
        self.stopButton.clicked.connect(lambda: self.stop_playback())
        self.stepbackButton.clicked.connect(lambda: self.step_back())
        self.stepfwdButton.clicked.connect(lambda: self.step_forward())
        self.fastfwdButton.clicked.connect(lambda: self.fast_forward())
        self.goto_beginingButton.clicked.connect(lambda: self.goto_beginning())
        self.goto_endButton.clicked.connect(lambda: self.goto_end())
        self.startmarkerButton.clicked.connect(lambda: self.set_startPosition(self.selectStartMarkerButton))
        self.endmarkerButton.clicked.connect(lambda: self.set_stopPosition(self.selectEndMarkerButton))
        self.selectStartMarkerButton.clicked.connect(lambda: self.goto(self.selectStartMarkerButton))
        self.selectEndMarkerButton.clicked.connect(lambda: self.goto(self.selectEndMarkerButton))
        self.muteButton.clicked.connect(lambda: self.volume_mute())

        self.retranslateUi()
        self.disable_buttons()
        change_widget_state(False, self.cropButton, self.videocropButton, self.cancelButton, self.playButton,
                            self.stopButton, self.stepbackButton, self.stepfwdButton,
                            self.fastfwdButton, self.goto_beginingButton, self.goto_endButton, self.startmarkerButton,
                            self.endmarkerButton, self.selectStartMarkerButton, self.selectEndMarkerButton,
                            self.timelineSlider)
        QtCore.QMetaObject.connectSlotsByName(self)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate('Form', 'Form'))
        self.videoLineEdit.setPlaceholderText(_translate('Form', 'Choose the video you want to crop'))
        self.videoButton.setText(_translate('Form', 'Open Video'))
        self.positionLabel.setText(_translate('Form', '00:00:00'))
        self.durationLabel.setText(_translate('Form', '00:00:00'))
        self.mfaceCheckBox.setText(_translate('Form', 'Multi-Face'))
        self.tiltCheckBox.setText(_translate('Form', 'Autotilt'))
        self.exposureCheckBox.setText(_translate('Form', 'Autocorrect'))
        self.destinationLineEdit.setPlaceholderText(
            _translate('Form', 'Choose where you want to save the cropped images'))
        self.destinationButton.setText(_translate('Form', 'Destination Folder'))
        self.selectStartMarkerButton.setText(_translate('Form', '00:00:00'))
        self.selectEndMarkerButton.setText(_translate('Form', '00:00:00'))
    
    def open_folder(self, line_edit: PathLineEdit) -> None:
        self.check_playback_state()
        f_name = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Directory', Photo().default_directory)
        line_edit.setText(f_name)

    def open_video(self) -> None:
        self.check_playback_state()
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open Video', self.default_directory,
                                                            'Video files (*.mp4 *.avi)')
        if file_name != '':
            self.player.setSource(QtCore.QUrl.fromLocalFile(file_name))
            self.videoLineEdit.setText(file_name)
            self.playButton.setEnabled(True)

    def change_audio_icon(self) -> None:
        if self.audio.isMuted():
            self.muteButton.setIcon(QtGui.QIcon('resources\\icons\\multimedia_unmute.svg'))
        else:
            self.muteButton.setIcon(QtGui.QIcon('resources\\icons\\multimedia_mute.svg'))

    def playback_bool(self,
                      a0: QtMultimedia.QMediaPlayer.PlaybackState = QtMultimedia.QMediaPlayer.PlaybackState.PausedState,
                      a1: QtMultimedia.QMediaPlayer.PlaybackState = QtMultimedia.QMediaPlayer.PlaybackState.StoppedState) -> Tuple[bool, bool]:
        """Returns a tuple of bools comparing the playback state to the Class attributes of PyQt6.QtMultimedia.QMediaPlayer.PlaybackState"""
        return self.player.playbackState() == a0, self.player.playbackState() == a1

    def check_playback_state(self) -> None:
        """Stops playback if in the paused state or playing state"""
        x, y = self.playback_bool(a1=QtMultimedia.QMediaPlayer.PlaybackState.PlayingState)
        if x ^ y:
            self.stop_playback()

    def change_playback_state(self):
        if self.player.playbackState() == QtMultimedia.QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
            self.speed = 0
        else:
            self.player.play()

    def change_playback_icons(self):
        if self.player.playbackState() == QtMultimedia.QMediaPlayer.PlaybackState.PlayingState:
            self.playButton.setIcon(QtGui.QIcon('resources\\icons\\multimedia_pause.svg'))
            self.timelineSlider.setEnabled(True)
            self.player.setPlaybackRate(1)
        else:
            self.playButton.setIcon(QtGui.QIcon('resources\\icons\\multimedia_play.svg'))

    def change_media_widget_state(self, *buttons: QtWidgets.QPushButton):
        x, y = self.playback_bool()
        for button in buttons:
            button.setDisabled(x ^ y)
        
        for button in (self.stopButton, self.fastfwdButton):
            button.setEnabled(x)

    def create_mediaPlayer(self) -> None:
        self.player.setAudioOutput(self.audio)
        self.player.setVideoOutput(self.videoWidget)

    def stop_playback(self) -> None:
        self.timelineSlider.setDisabled(True)
        x, y = self.playback_bool(a1=QtMultimedia.QMediaPlayer.PlaybackState.PlayingState)
        if x ^ y: self.player.stop()

    def fast_forward(self) -> None:
        x, y = self.playback_bool()
        if x ^ y: return None
        VIDEO_SPEEDS = np.array([1.0, 1.25, 1.5, 1.75, 2.0])
        self.reverse = 0
        self.speed += 1
        if self.speed > VIDEO_SPEEDS.size - 1:
            self.player.setPlaybackRate(VIDEO_SPEEDS[-1])
        else:
            self.player.setPlaybackRate(VIDEO_SPEEDS[self.speed])

    def step_forward(self):
        new_position = self.player.position() + 10_000
        if new_position >= self.player.duration():
            self.player.setPosition(self.player.duration())
        else:
            self.player.setPosition(new_position)
    
    def step_back(self):
        new_position = self.player.position() - 10_000
        if new_position <= 0:
            self.player.setPosition(0)
        else:
            self.player.setPosition(new_position)

    def position_changed(self, position: int) -> None:
        def callback():
            if self.timelineSlider.maximum() != self.player.duration():
                self.timelineSlider.setMaximum(self.player.duration())

            self.timelineSlider.blockSignals(True)
            self.timelineSlider.setValue(position)
            minutes, seconds = divmod(round(position * 0.001), 60)
            hours, minutes = divmod(minutes, 60)
            self.timelineSlider.blockSignals(False)
            self.positionLabel.setText(QtCore.QTime(hours, minutes, seconds).toString())
        
        thread = Thread(target=callback)
        thread.start()

    def duration_changed(self, duration: int) -> None:
        self.timelineSlider.setMaximum(duration)
        if duration >= 0:
            minutes, seconds = divmod(round(self.player.duration() * 0.001), 60)
            hours, minutes = divmod(minutes, 60)
            self.durationLabel.setText(QtCore.QTime(hours, minutes, seconds).toString())
            self.selectEndMarkerButton.setText(QtCore.QTime(hours, minutes, seconds).toString())

    def player_slider_changed(self, position: int) -> None:
        self.player.setPosition(position)

    def volume_slider_changed(self, position: int) -> None:
        self.audio.setVolume(position)
        self.vol_cache = position

    def volume_mute(self) -> None:
        if self.audio.isMuted():
            self.audio.setMuted(False)
            self.volumeSlider.setValue(self.vol_cache)
        else:
            self.audio.setMuted(True)
            self.volumeSlider.setValue(0)

    def goto_beginning(self) -> None:
        self.player.setPosition(0)

    def goto_end(self) -> None:
        self.player.setPosition(self.player.duration())

    @staticmethod
    def set_marker_time(button: QtWidgets.QPushButton, position: Union[int, float]) -> None:
        minutes, seconds = divmod(round(position), 60)
        hours, minutes = divmod(minutes, 60)
        button.setText(QtCore.QTime(hours, minutes, seconds).toString())

    def set_startPosition(self, button: QtWidgets.QPushButton) -> None:
        if (time_value := self.timelineSlider.value() * 0.001) < self.stop_position:
            self.start_position = time_value
            self.set_marker_time(button, self.start_position)
        elif self.start_position == 0 and self.stop_position == 0:
            self.start_position = time_value
            self.set_marker_time(button, self.start_position) 

    def set_stopPosition(self, button: QtWidgets.QPushButton) -> None:
        if (time_value := self.timelineSlider.value() * 0.001) > self.start_position:
            self.stop_position = time_value
            self.set_marker_time(button, self.stop_position)
        elif self.start_position == 0 and self.stop_position == 0:
            self.start_position = time_value
            self.set_marker_time(button, self.start_position) 

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
            if isinstance(input_widget, (NumberLineEdit, PathLineEdit)):
                input_widget.textChanged.connect(lambda: self.disable_buttons())
            elif isinstance(input_widget, QtWidgets.QCheckBox):
                if input_widget is self.mfaceCheckBox:
                    input_widget.clicked.connect(lambda: uncheck_boxes(self.exposureCheckBox, self.tiltCheckBox))
                else:
                    input_widget.clicked.connect(lambda: uncheck_boxes(self.mfaceCheckBox))
 
    def disable_buttons(self) -> None:
        def all_filled(*line_edits: Union[PathLineEdit, NumberLineEdit, QtWidgets.QComboBox]) -> bool:
            x = all(edit.state == LineEditState.VALID_INPUT
                    for edit in line_edits if isinstance(edit, (PathLineEdit, NumberLineEdit)))
            y = all(edit.currentText() for edit in line_edits if isinstance(edit, QtWidgets.QComboBox))
            return x and y

        def update_widget_state(condition: bool, *widgets: QtWidgets.QWidget) -> None:
            for widget in widgets:
                change_widget_state(condition, widget)

        # Video logic
        update_widget_state(
            all_filled(self.videoLineEdit, self.destinationLineEdit, self.widthLineEdit, self.heightLineEdit),
            self.cropButton, self.videocropButton)

    def crop_frame(self) -> None:
        job = self.create_job(self.exposureCheckBox, 
                              self.mfaceCheckBox, 
                              self.tiltCheckBox,
                              video_path=self.videoLineEdit, 
                              destination=self.destinationLineEdit)
        self.crop_worker.crop_frame(job, self.positionLabel, self.timelineSlider)

    def video_process(self) -> None:
        job = self.create_job(self.exposureCheckBox, 
                              self.mfaceCheckBox, 
                              self.tiltCheckBox,
                              video_path=self.videoLineEdit, 
                              destination=self.destinationLineEdit,
                              start_position=self.start_position, 
                              stop_position=self.stop_position)
        self.player.pause()
        self.run_batch_process(self.crop_worker.extract_frames, self.crop_worker.reset_v_task, job)
