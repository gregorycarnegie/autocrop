from pathlib import Path
from threading import Thread
from typing import Union

import numpy as np
from PyQt6.QtCore import QTime, QTimer, QUrl
from PyQt6.QtGui import QIcon
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtWidgets import QFileDialog, QLabel, QLineEdit, QMainWindow, QPushButton, QSlider

from .media_state import MediaPlaybackState

VIDEO_TYPES = np.array(['.avi', '.m4v', '.mkv', '.mov', '.mp4', '.wmv'])

class Video:
    def __init__(self, audio: QAudioOutput,
                 video_widget: QVideoWidget,
                 media_player: QMediaPlayer,
                 timeline_slider: QSlider,
                 volume_slider: QSlider,
                 position_label: QLabel,
                 duration_label: QLabel,
                 select_end_marker_button: QPushButton) -> None:
        self.rewind_timer = QTimer()
        self.default_directory = f'{Path.home()}\\Videos'

        self.audio_state = MediaPlaybackState.UNMUTED

        self.player, self.video_widget, self.audio = (media_player, video_widget, audio)
        self.create_mediaPlayer()
        self.timeline_slider = timeline_slider
        self.position_label = position_label
        self.volume_slider = volume_slider
        self.vol_cache = 70
        self.durationLabel = duration_label
        self.selectEndMarkerButton = select_end_marker_button

        self.start_position, self.stop_position, self.step = 0.0, 0.0, 2

        self.speed = 0
        self.reverse = 0

        self.player.positionChanged.connect(self.position_changed)
        self.player.durationChanged.connect(self.duration_changed)
        self.volume_slider.sliderMoved.connect(self.volume_slider_changed)
        self.timeline_slider.sliderMoved.connect(self.player_slider_changed)

    @staticmethod
    def type_string() -> str:
        return 'All Files (*);;' + ';;'.join(f'{_} Files (*{_})' for _ in np.sort(VIDEO_TYPES))

    def open_video(self, main_window: QMainWindow,
                   video_line_edit: QLineEdit,
                   play_button: QPushButton) -> None:
        file_name, _ = QFileDialog.getOpenFileName(main_window, 'Open Video', self.default_directory,
                                                   'Video files (*.mp4 *.avi)')
        if file_name != '':
            self.player.setSource(QUrl.fromLocalFile(file_name))
            video_line_edit.setText(file_name)
            play_button.setEnabled(True)

    def create_mediaPlayer(self) -> None:
        self.player.setAudioOutput(self.audio)
        self.player.setVideoOutput(self.video_widget)

    def play_video(self) -> None:
        self.timeline_slider.setEnabled(True)
        self.player.play()
        self.player.setPlaybackRate(1)

    def pause_video(self):
        self.player.pause()

    def stop_btn(self) -> None:
        self.timeline_slider.setDisabled(True)
        self.player.stop()

    def fastfwd(self) -> None:
        if self.player.playbackState() in [QMediaPlayer.PlaybackState.PausedState, QMediaPlayer.PlaybackState.StoppedState]:
            return None
        VIDEO_SPEEDS = np.array([1.0, 1.25, 1.5, 1.75, 2.0])
        self.reverse = 0
        self.speed += 1
        if self.speed > VIDEO_SPEEDS.size - 1:
            self.player.setPlaybackRate(VIDEO_SPEEDS[-1])
        else:
            self.player.setPlaybackRate(VIDEO_SPEEDS[self.speed])

    def rewind(self) -> None:
        if self.player.playbackState() in [QMediaPlayer.PlaybackState.PausedState, QMediaPlayer.PlaybackState.StoppedState]:
            return None        
        # Create a QTimer if it doesn't exist yet
        if not hasattr(self, 'rewind_timer'):
            self.rewind_timer = QTimer()
            self.rewind_timer.timeout.connect(self.rewind_step)

        # Start the timer to call rewind_step every 100 milliseconds
        self.rewind_timer.start(100)

    def rewind_step(self) -> None:
        # Calculate the new position
        new_position = self.player.position() - 1_000  # Amount to rewind in milliseconds

        # Make sure we don't go past the start of the video
        new_position = max(new_position, 0)

        # Set the new position
        self.player.setPosition(new_position)

        # If we're at the start of the video, stop the timer
        if new_position == 0:
            self.rewind_timer.stop()

    def stepfwd(self):
        new_position = self.player.position() + 10_000
        if new_position >= self.player.duration():
            self.player.setPosition(self.player.duration())
        else:
            self.player.setPosition(new_position)
    
    def stepback(self):
        new_position = self.player.position() - 10_000
        if new_position <= 0:
            self.player.setPosition(0)
        else:
            self.player.setPosition(new_position)

    def position_changed(self, position: int) -> None:
        def callback():
            if self.timeline_slider.maximum() != self.player.duration():
                self.timeline_slider.setMaximum(self.player.duration())

            self.timeline_slider.blockSignals(True)
            self.timeline_slider.setValue(position)
            minutes, seconds = divmod(round(position / 1_000), 60)
            hours, minutes = divmod(minutes, 60)
            self.timeline_slider.blockSignals(False)

            self.position_label.setText(QTime(hours, minutes, seconds).toString())
        
        thread = Thread(target=callback)
        thread.start()

    def duration_changed(self, duration: int) -> None:
        self.timeline_slider.setMaximum(duration)
        if duration >= 0:
            minutes, seconds = divmod(round(self.player.duration() / 1_000), 60)
            hours, minutes = divmod(minutes, 60)
            self.durationLabel.setText(QTime(hours, minutes, seconds).toString())
            self.selectEndMarkerButton.setText(QTime(hours, minutes, seconds).toString())

    def player_slider_changed(self, position: int) -> None:
        self.player.setPosition(position)

    def volume_slider_changed(self, position: int) -> None:
        self.audio.setVolume(position)
        self.vol_cache = position

    def volume_mute(self,
                    volume_slider: QSlider,
                    mute_button: QPushButton) -> None:
        if self.audio_state == MediaPlaybackState.UNMUTED:
            self.audio.setMuted(True)
            self.audio_state = MediaPlaybackState.MUTED
            volume_slider.setValue(0)
            mute_button.setIcon(QIcon('resources\\icons\\multimedia_unmute.svg'))
        elif self.audio_state == MediaPlaybackState.MUTED:
            self.audio.setMuted(False)
            self.audio_state = MediaPlaybackState.UNMUTED
            volume_slider.setValue(self.vol_cache)
            mute_button.setIcon(QIcon('resources\\icons\\multimedia_mute.svg'))

    def goto_begining(self) -> None:
        self.player.setPosition(0)

    def goto_end(self) -> None:
        self.player.setPosition(self.player.duration())

    @staticmethod
    def set_marker_time(button: QPushButton, position: Union[int, float]) -> None:
        minutes, seconds = divmod(round(position), 60)
        hours, minutes = divmod(minutes, 60)
        button.setText(QTime(hours, minutes, seconds).toString())

    def set_startPosition(self, button: QPushButton,
                          timeline_slider: QSlider) -> None:
        if (time_value := timeline_slider.value() / 1_000) < self.stop_position:
            self.start_position = time_value
            self.set_marker_time(button, self.start_position)
        elif self.start_position == 0 and self.stop_position == 0:
            self.start_position = time_value
            self.set_marker_time(button, self.start_position) 

    def set_stopPosition(self, button: QPushButton,
                         timeline_slider: QSlider) -> None:
        if (time_value := timeline_slider.value() / 1_000) > self.start_position:
            self.stop_position = time_value
            self.set_marker_time(button, self.stop_position)
        elif self.start_position == 0 and self.stop_position == 0:
            self.start_position = time_value
            self.set_marker_time(button, self.start_position) 

    def goto(self, marker_button: QPushButton) -> None:
        m = np.array(marker_button.text().split(':')).astype(int)
        if (x := np.sum([60 ** (2 - i) * m[i] for i in np.arange(m.size)]) * 1_000) >= self.player.duration():
            self.player.setPosition(self.player.duration())
        elif x == 0:
            self.player.setPosition(0)
        else:
            self.player.setPosition(int(x))
