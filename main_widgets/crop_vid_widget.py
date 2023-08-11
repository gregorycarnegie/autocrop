from threading import Thread
from typing import Optional, Tuple, Union
from pathlib import Path

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets, QtMultimedia
from PyQt6.QtMultimediaWidgets import QVideoWidget

from core import Cropper, CustomDialWidget, ExtWidget, FunctionType, window_functions
from file_types import Photo, Video
from line_edits import PathLineEdit, PathType, NumberLineEdit, LineEditState
from .crop_batch_widget import CropBatchWidget
from .enums import ButtonType


class CropVideoWidget(CropBatchWidget):
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
        self.default_directory = Video().default_directory
        self.player = QtMultimedia.QMediaPlayer()
        self.audio = QtMultimedia.QAudioOutput()
        self.start_position, self.stop_position, self.step = 0.0, 0.0, 2
        self.speed = 0
        self.reverse = 0
        self.setObjectName('Form')
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self)
        self.verticalLayout_2.setObjectName('verticalLayout_2')
        self.videoLineEdit = self.setup_path_line_edit('videoLineEdit', PathType.VIDEO)
        self.horizontalLayout_3.addWidget(self.videoLineEdit)
        self.videoButton = self.setup_process_button('videoButton', 'clapperboard', ButtonType.NAVIGATION_BUTTON)
        self.horizontalLayout_3.addWidget(self.videoButton)
        self.verticalLayout_2.addLayout(self.horizontalLayout_3)
        self.muteButton = QtWidgets.QPushButton(parent=self.frame)
        self.muteButton.setText('')
        icon1 = QtGui.QIcon()
        icon1.addPixmap(
            QtGui.QPixmap('resources\\icons\\multimedia_mute.svg'), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.muteButton.setIcon(icon1)
        self.muteButton.setObjectName('muteButton')
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
        self.positionLabel = QtWidgets.QLabel(parent=self.frame)
        self.positionLabel.setObjectName('positionLabel')
        self.timelineSlider = QtWidgets.QSlider(parent=self.frame)
        self.timelineSlider.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.timelineSlider.setObjectName('timelineSlider')
        self.durationLabel = QtWidgets.QLabel(parent=self.frame)
        self.durationLabel.setObjectName('durationLabel')
        window_functions.add_widgets(self.horizontalLayout_1, self.muteButton, self.volumeSlider,
                                     self.positionLabel, self.timelineSlider, self.durationLabel)
        spacerItem = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout_1.addItem(spacerItem)
        window_functions.add_widgets(self.horizontalLayout_1, self.mfaceCheckBox,
                                     self.tiltCheckBox, self.exposureCheckBox)
        self.verticalLayout_1.addLayout(self.horizontalLayout_1)
        self.videoWidget = QVideoWidget(parent=self.frame)
        self.videoWidget.setStyleSheet('background: #1f2c33')
        self.videoWidget.setObjectName('videoWidget')
        self.create_mediaPlayer()
        self.verticalLayout_1.addWidget(self.videoWidget)
        self.videocropButton = self.setup_process_button('videocropButton', 'crop_video', ButtonType.PROCESS_BUTTON)
        window_functions.add_widgets(self.horizontalLayout_2, self.cropButton,
                                     self.videocropButton, self.cancelButton)
        self.verticalLayout_1.addLayout(self.horizontalLayout_2)
        self.verticalLayout_1.addWidget(self.progressBar)
        self.verticalLayout_1.setStretch(0, 1)
        self.verticalLayout_1.setStretch(1, 10)
        self.verticalLayout_1.setStretch(2, 1)
        self.verticalLayout_1.setStretch(3, 1)
        self.verticalLayout_2.addWidget(self.frame)
        window_functions.add_widgets(self.horizontalLayout_4, self.destinationLineEdit, self.destinationButton)
        self.verticalLayout_2.addLayout(self.horizontalLayout_4)
        self.playButton = window_functions.create_media_button(
            'playButton', 'play', self.horizontalLayout_5, parent=self)
        self.stopButton = window_functions.create_media_button(
            'stopButton', 'stop', self.horizontalLayout_5, parent=self)
        self.stepbackButton = window_functions.create_media_button(
            'stepbackButton', 'left', self.horizontalLayout_5, parent=self)
        self.stepfwdButton = window_functions.create_media_button(
            'stepfwdButton', 'right', self.horizontalLayout_5, parent=self)
        self.fastfwdButton = window_functions.create_media_button(
            'fastfwdButton', 'fastfwd', self.horizontalLayout_5, parent=self)
        self.goto_beginingButton = window_functions.create_media_button(
            'goto_beginingButton', 'begining', self.horizontalLayout_5, parent=self)
        self.goto_endButton = window_functions.create_media_button(
            "goto_endButton", "end", self.horizontalLayout_5, parent=self)
        self.startmarkerButton = window_functions.create_media_button(
            'startmarkerButton', 'leftmarker', self.horizontalLayout_5, parent=self)
        self.endmarkerButton = window_functions.create_media_button(
            'endmarkerButton', 'rightmarker', self.horizontalLayout_5, parent=self)
        spacerItem1 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem1)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName('gridLayout')
        self.label_A = self.setup_label('a')
        self.gridLayout.addWidget(self.label_A, 0, 0, 1, 1)
        self.selectStartMarkerButton = QtWidgets.QPushButton(parent=self)
        self.selectStartMarkerButton.setMinimumSize(QtCore.QSize(150, 0))
        self.selectStartMarkerButton.setObjectName('selectStartMarkerButton')
        self.gridLayout.addWidget(self.selectStartMarkerButton, 0, 1, 1, 1)
        self.label_B = self.setup_label('b')
        self.gridLayout.addWidget(self.label_B, 1, 0, 1, 1)
        self.selectEndMarkerButton = QtWidgets.QPushButton(parent=self)
        self.selectEndMarkerButton.setMinimumSize(QtCore.QSize(150, 0))
        self.selectEndMarkerButton.setObjectName('selectEndMarkerButton')
        self.gridLayout.addWidget(self.selectEndMarkerButton, 1, 1, 1, 1)
        self.horizontalLayout_5.addLayout(self.gridLayout)
        self.verticalLayout_2.addLayout(self.horizontalLayout_5)
        self.verticalLayout_2.setStretch(0, 1)
        self.verticalLayout_2.setStretch(1, 19)
        self.verticalLayout_2.setStretch(3, 2)

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
        self.cancelButton.clicked.connect(
            lambda: self.cancel_button_operation(self.cancelButton, self.videocropButton, self.cropButton))

        self.connect_input_widgets(self.widthLineEdit, self.heightLineEdit, self.destinationLineEdit,
                                   self.exposureCheckBox, self.mfaceCheckBox, self.tiltCheckBox)
        
        # Media connections
        self.audio.mutedChanged.connect(lambda: self.change_audio_icon())
        self.player.playbackStateChanged.connect(
            lambda: self.change_media_widget_state(
                self.stopButton, self.stepbackButton, self.stepfwdButton, self.fastfwdButton,
                self.goto_beginingButton, self.goto_endButton, self.startmarkerButton, self.endmarkerButton,
                self.selectStartMarkerButton, self.selectEndMarkerButton))
        self.player.playbackStateChanged.connect(lambda: self.change_playback_icons())

        # Connect crop worker
        self.connect_crop_worker()

        self.playButton.clicked.connect(lambda: self.change_playback_state())
        self.playButton.clicked.connect(
            lambda: window_functions.change_widget_state(
                True, self.stopButton, self.stepbackButton,  self.stepfwdButton, self.fastfwdButton,
                self.goto_beginingButton, self.goto_endButton, self.startmarkerButton, self.endmarkerButton,
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
        window_functions.change_widget_state(
            False, self.cropButton, self.videocropButton, self.cancelButton, self.playButton,
            self.stopButton, self.stepbackButton, self.stepfwdButton, self.fastfwdButton, self.goto_beginingButton,
            self.goto_endButton, self.startmarkerButton, self.endmarkerButton, self.selectStartMarkerButton,
            self.selectEndMarkerButton, self.timelineSlider)
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
    
    def connect_crop_worker(self) -> None:
        widget_list = (self.widthLineEdit, self.heightLineEdit, self.sensitivity_dialArea.dial, self.face_dialArea.dial,
                       self.gamma_dialArea.dial, self.top_dialArea.dial, self.bottom_dialArea.dial,
                       self.left_dialArea.dial, self.right_dialArea.dial, self.videoLineEdit, self.destinationLineEdit,
                       self.destinationButton, self.extWidget.radioButton_1, self.extWidget.radioButton_2,
                       self.extWidget.radioButton_3, self.extWidget.radioButton_4, self.extWidget.radioButton_5,
                       self.extWidget.radioButton_6, self.cropButton, self.videocropButton, self.exposureCheckBox,
                       self.mfaceCheckBox, self.tiltCheckBox, self.videocropButton, self.playButton, self.stopButton,
                       self.stepbackButton, self.stepfwdButton, self.fastfwdButton, self.goto_beginingButton,
                       self.goto_endButton, self.startmarkerButton, self.endmarkerButton, self.selectStartMarkerButton,
                       self.selectEndMarkerButton)
        # Video start connection
        self.crop_worker.video_started.connect(lambda: window_functions.disable_widget(*widget_list))
        self.crop_worker.video_started.connect(lambda: window_functions.enable_widget(self.cancelButton))
        # Video end connection
        self.crop_worker.video_started.connect(lambda: window_functions.disable_widget(*widget_list))
        self.crop_worker.video_finished.connect(lambda: window_functions.disable_widget(self.cancelButton))
        self.crop_worker.video_finished.connect(lambda: window_functions.show_message_box(self.destination))
        self.crop_worker.video_progress.connect(self.update_progress)
    
    def setup_label(self, name: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(parent=self)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(label.sizePolicy().hasHeightForWidth())
        label.setSizePolicy(sizePolicy)
        label.setMaximumSize(QtCore.QSize(14, 14))
        label.setText('')
        label.setPixmap(QtGui.QPixmap(f'resources\\icons\\marker_label_{name.lower()}.svg'))
        label.setScaledContents(True)
        label.setObjectName(f'label_{name.upper()}')
        return label
    
    def open_folder(self, line_edit: PathLineEdit) -> None:
        self.check_playback_state()
        f_name = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Directory', Photo().default_directory)
        line_edit.setText(f_name)

    def open_video(self) -> None:
        self.check_playback_state()
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open Video', self.default_directory,
                                                             Video().type_string)
        self.videoLineEdit.setText(file_name)
        if self.videoLineEdit.state is LineEditState.INVALID_INPUT:
            return None
        self.player.setSource(QtCore.QUrl.fromLocalFile(file_name))
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
        x: bool = (time_value := self.timelineSlider.value() * 0.001) < self.stop_position
        y: bool = self.start_position == 0.0 and self.stop_position == 0.0
        if x | y:
            self.start_position = time_value
            self.set_marker_time(button, self.start_position)

    def set_stopPosition(self, button: QtWidgets.QPushButton) -> None:
        x: bool = (time_value := self.timelineSlider.value() * 0.001) > self.start_position
        y: bool = self.start_position == 0.0 and self.stop_position == 0.0
        if x | y:
            self.stop_position = time_value
            self.set_marker_time(button, self.stop_position)

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
                    self.connect_checkboxs(input_widget)
                case _: pass
 
    def disable_buttons(self) -> None:
        def all_filled(*line_edits: Union[PathLineEdit, NumberLineEdit, QtWidgets.QComboBox]) -> bool:
            x = all(edit.state == LineEditState.VALID_INPUT
                    for edit in line_edits if isinstance(edit, (PathLineEdit, NumberLineEdit)))
            y = all(edit.currentText() for edit in line_edits if isinstance(edit, QtWidgets.QComboBox))
            return x and y

        def update_widget_state(condition: bool, *widgets: QtWidgets.QWidget) -> None:
            for widget in widgets:
                window_functions.change_widget_state(condition, widget)

        # Video logic
        update_widget_state(
            all_filled(self.videoLineEdit, self.destinationLineEdit, self.widthLineEdit, self.heightLineEdit),
            self.cropButton, self.videocropButton)

    def crop_frame(self) -> None:
        def callback():
            job = self.create_job(self.exposureCheckBox, 
                                self.mfaceCheckBox, 
                                self.tiltCheckBox,
                                video_path=Path(self.videoLineEdit.text()), 
                                destination=Path(self.destinationLineEdit.text()))
            self.player.pause()
            self.crop_worker.crop_frame(job, self.positionLabel, self.timelineSlider)

        if Path(self.videoLineEdit.text()).parent == Path(self.destinationLineEdit.text()):
            match window_functions.show_warning(FunctionType.FRAME):
                case QtWidgets.QMessageBox.StandardButton.Yes:
                    callback()
                case _: return
        callback()

    def video_process(self) -> None:
        def callback():
            job = self.create_job(self.exposureCheckBox, 
                                self.mfaceCheckBox, 
                                self.tiltCheckBox,
                                video_path=Path(self.videoLineEdit.text()), 
                                destination=Path(self.destinationLineEdit.text()),
                                start_position=self.start_position, 
                                stop_position=self.stop_position)
            self.player.pause()
            self.run_batch_process(self.crop_worker.extract_frames, self.crop_worker.reset_v_task, job)

        if Path(self.videoLineEdit.text()).parent == Path(self.destinationLineEdit.text()):
            match window_functions.show_warning(FunctionType.VIDEO):
                case QtWidgets.QMessageBox.StandardButton.Yes:
                    callback()
                case _: return
        callback()
