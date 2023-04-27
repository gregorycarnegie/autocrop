import utils
import numpy as np
from pathlib import Path
from typing import Union
from PyQt6 import QtCore, QtGui, QtMultimedia, QtMultimediaWidgets, QtWidgets

class Photo:
    def __init__(self) -> None:
        self.default_directory = f"{Path.home()}\\Pictures"

    def ALL_TYPES(self) -> np.ndarray:
        return np.concatenate([utils.PIL_TYPES, utils.CV2_TYPES, utils.RAW_TYPES])

    def file_filter(self) -> np.ndarray:
        return np.array([f'*{file}' for file in self.ALL_TYPES()])

    def type_string(self) -> str:
        return "All Files (*);;" + ";;".join(f"{_} Files (*{_})" for _ in np.sort(self.ALL_TYPES()))


class Video:
    def __init__(self, audio: QtMultimedia.QAudioOutput, video_widget: QtMultimediaWidgets.QVideoWidget,
                 media_player: QtMultimedia.QMediaPlayer, timeline_slider: QtWidgets.QSlider,
                 volume_slider: QtWidgets.QSlider, position_label: QtWidgets.QLabel, durationLabel: QtWidgets.QLabel,
                 selectEndMarkerButton: QtWidgets.QPushButton) -> None:
        self.default_directory = f"{Path.home()}\\Videos"
        self.muted = False
        self.paused = False

        self.player, self.video_widget, self.audio = (media_player, video_widget, audio)
        self.create_mediaPlayer()

        self.timeline_slider = timeline_slider
        self.position_label = position_label
        self.volume_slider = volume_slider
        self.durationLabel = durationLabel
        self.selectEndMarkerButton = selectEndMarkerButton

        self.start_position, self.stop_position, self.step = 0.0, 0.0, 2

        self.VIDEO_SPEEDS = np.array([1.0, 1.25, 1.5, 1.75, 2.0])
        self.REWIND_SPEEDS = -1 * self.VIDEO_SPEEDS
        self.speed = 0
        self.reverse = 0

        self.player.positionChanged.connect(self.position_changed)
        self.player.durationChanged.connect(self.duration_changed)
        self.volume_slider.sliderMoved.connect(self.volume_slider_changed)
        self.timeline_slider.sliderMoved.connect(self.player_slider_changed)

    def type_string(self) -> str:
        return "All Files (*);;" + ";;".join(f"{_} Files (*{_})" for _ in np.sort(utils.VIDEO_TYPES))

    def open_video(self, main_window: QtWidgets.QMainWindow, video_line_edit: QtWidgets.QLineEdit,
                   play_button: QtWidgets.QPushButton, crop_button: QtWidgets.QPushButton) -> None:
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(main_window, 'Open Video', self.default_directory, 'Video files (*.mp4 *.avi)')
        if file_name != '':
            self.player.setSource(QtCore.QUrl.fromLocalFile(file_name))
            video_line_edit.setText(file_name)
            play_button.setEnabled(True)
            crop_button.setEnabled(True)

    @staticmethod
    def open_video_folder(main_window: QtWidgets.QMainWindow, destination_line_edit: QtWidgets.QLineEdit) -> None:
        folder_name = QtWidgets.QFileDialog.getExistingDirectory(main_window, 'Select Directory', f"{Path.home()}\\Videos")
        
        if folder_name != '':
            destination_line_edit.setText(folder_name)

    def create_mediaPlayer(self) -> None:
        self.player.setAudioOutput(self.audio)
        self.player.setVideoOutput(self.video_widget)

    def play_video(self, play_button: QtWidgets.QPushButton) -> None:
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
        self.reverse = 0
        self.speed += 1
        self.speed = min(self.speed, self.VIDEO_SPEEDS.size - 1)
        self.player.setPlaybackRate(self.VIDEO_SPEEDS[self.speed])

    def rewind(self) -> None:
        self.speed = 0
        self.reverse += 1
        self.speed = min(self.speed, self.REWIND_SPEEDS.size - 1)
        self.player.setPlaybackRate(self.REWIND_SPEEDS[self.reverse])

    def position_changed(self, position) -> None:
        if self.timeline_slider.maximum() != self.player.duration():
            self.timeline_slider.setMaximum(self.player.duration())

        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setValue(position)
        minutes, seconds = divmod(round(position / 1_000), 60)
        hours, minutes = divmod(minutes, 60)
        self.timeline_slider.blockSignals(False)

        self.position_label.setText(QtCore.QTime(hours, minutes, seconds).toString())

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

    def volume_mute(self, volume_slider: QtWidgets.QSlider, mute_button: QtWidgets.QPushButton) -> None:
        if self.muted:
            self.audio.setMuted(True)
            volume_slider.setValue(0)
            mute_button.setIcon(QtGui.QIcon('resources\\icons\\multimedia_unmute.svg'))
        else:
            self.audio.setMuted(False)
            volume_slider.setValue(70)
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
