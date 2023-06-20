import numpy as np
from threading import Thread
from pathlib import Path
from typing import Union, Optional
from PyQt6 import QtCore, QtGui, QtMultimedia, QtMultimediaWidgets, QtWidgets
from PyQt6 import QtMultimedia, QtMultimediaWidgets

VIDEO_TYPES = np.array([".avi", ".m4v", ".mkv", ".mov", ".mp4", ".wmv"])

class Video:
    def __init__(self,
                 audio: QtMultimedia.QAudioOutput,
                 video_widget: QtMultimediaWidgets.QVideoWidget,
                 media_player: QtMultimedia.QMediaPlayer,
                 timeline_slider: QtWidgets.QSlider,
                 volume_slider: QtWidgets.QSlider,
                 position_label: QtWidgets.QLabel,
                 duration_label: QtWidgets.QLabel,
                 select_end_marker_button: QtWidgets.QPushButton) -> None:
        self.rewind_timer = Optional[QtCore.QTimer()]
        self.default_directory = f"{Path.home()}\\Videos"
        self.muted = False
        self.paused = False

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
        return "All Files (*);;" + ";;".join(f"{_} Files (*{_})" for _ in np.sort(VIDEO_TYPES))

    def open_video(self,
                   main_window: QtWidgets.QMainWindow,
                   video_line_edit: QtWidgets.QLineEdit,
                   play_button: QtWidgets.QPushButton,
                   crop_button: QtWidgets.QPushButton) -> None:
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(main_window, 'Open Video', self.default_directory,
                                                             'Video files (*.mp4 *.avi)')
        if file_name != '':
            self.player.setSource(QtCore.QUrl.fromLocalFile(file_name))
            video_line_edit.setText(file_name)
            play_button.setEnabled(True)
            crop_button.setEnabled(True)

    def create_mediaPlayer(self) -> None:
        self.player.setAudioOutput(self.audio)
        self.player.setVideoOutput(self.video_widget)

    def play_video(self, play_button: QtWidgets.QPushButton) -> None:
        self.timeline_slider.setEnabled(True)
        if self.paused:
            # Stop timer to update video frames
            # self.timer.stop()
            self.player.pause()
            play_button.setIcon(QtGui.QIcon('resources\\icons\\multimedia_play.svg'))
        else:
            # Start timer to update video frames
            # self.timer.start(33)
            self.player.play()
            self.player.setPlaybackRate(1)
            play_button.setIcon(QtGui.QIcon('resources\\icons\\multimedia_pause.svg'))
        self.paused = not self.paused

    def stop_btn(self) -> None:
        self.player.stop()

    def fastfwd(self) -> None:
        VIDEO_SPEEDS = np.array([1.0, 1.25, 1.5, 1.75, 2.0])
        self.reverse = 0
        self.speed += 1
        if self.speed > VIDEO_SPEEDS.size - 1:
            self.player.setPlaybackRate(VIDEO_SPEEDS[-1])
        else:
            self.player.setPlaybackRate(VIDEO_SPEEDS[self.speed])

    def rewind(self) -> None:
        # Create a QTimer if it doesn't exist yet
        if not hasattr(self, 'rewind_timer'):
            self.rewind_timer = QtCore.QTimer()
            self.rewind_timer.timeout.connect(self.rewind_step)

        # Start the timer to call rewind_step every 100 milliseconds
        if isinstance(self.rewind_timer, QtCore.QTimer):
            self.rewind_timer.start(100)

    def rewind_step(self) -> None:
        # Calculate the new position
        new_position = self.player.position() - 1_000  # Amount to rewind in milliseconds

        # Make sure we don't go past the start of the video
        new_position = max(new_position, 0)

        # Set the new position
        self.player.setPosition(new_position)

        # If we're at the start of the video, stop the timer
        if new_position == 0 and isinstance(self.rewind_timer, QtCore.QTimer):
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

    def position_changed(self, position) -> None:
        def callback():
            if self.timeline_slider.maximum() != self.player.duration():
                self.timeline_slider.setMaximum(self.player.duration())

            self.timeline_slider.blockSignals(True)
            self.timeline_slider.setValue(position)
            minutes, seconds = divmod(round(position / 1_000), 60)
            hours, minutes = divmod(minutes, 60)
            self.timeline_slider.blockSignals(False)

            self.position_label.setText(QtCore.QTime(hours, minutes, seconds).toString())
        
        thread = Thread(target=callback)
        thread.start()


    def duration_changed(self, duration: int) -> None:
        self.timeline_slider.setMaximum(duration)
        if duration >= 0:
            minutes, seconds = divmod(round(self.player.duration() / 1_000), 60)
            hours, minutes = divmod(minutes, 60)
            self.durationLabel.setText(QtCore.QTime(hours, minutes, seconds).toString())
            self.selectEndMarkerButton.setText(QtCore.QTime(hours, minutes, seconds).toString())

    def player_slider_changed(self, position: int) -> None:
        self.player.setPosition(position)

    def volume_slider_changed(self, position: int) -> None:
        self.audio.setVolume(position)
        self.vol_cache = position

    def volume_mute(self, volume_slider: QtWidgets.QSlider, mute_button: QtWidgets.QPushButton) -> None:
        if self.muted:
            self.audio.setMuted(True)
            volume_slider.setValue(0)
            mute_button.setIcon(QtGui.QIcon('resources\\icons\\multimedia_unmute.svg'))
        else:
            self.audio.setMuted(False)
            volume_slider.setValue(self.vol_cache)
            mute_button.setIcon(QtGui.QIcon('resources\\icons\\multimedia_mute.svg'))
        self.muted = not self.muted

    def goto_begining(self) -> None:
        self.player.setPosition(0)

    def goto_end(self) -> None:
        self.player.setPosition(self.player.duration())

    @staticmethod
    def set_marker_time(button: QtWidgets.QPushButton, position: Union[int, float]) -> None:
        minutes, seconds = divmod(round(position), 60)
        hours, minutes = divmod(minutes, 60)
        button.setText(QtCore.QTime(hours, minutes, seconds).toString())

    def set_startPosition(self, button: QtWidgets.QPushButton, timeline_slider: QtWidgets.QSlider) -> None:
        self.start_position = timeline_slider.value() / 1_000
        self.set_marker_time(button, self.start_position)

    def set_stopPosition(self, button: QtWidgets.QPushButton, timeline_slider: QtWidgets.QSlider) -> None:
        self.stop_position = timeline_slider.value() / 1_000
        self.set_marker_time(button, self.stop_position)

    def goto(self, marker_button: QtWidgets.QPushButton) -> None:
        m = np.array(marker_button.text().split(':')).astype(int)
        if (x := np.sum([60 ** (2 - i) * m[i] for i in np.arange(m.size)]) * 1_000) >= self.player.duration():
            self.player.setPosition(self.player.duration())
        elif x == 0:
            self.player.setPosition(0)
        else:
            self.player.setPosition(int(x))
