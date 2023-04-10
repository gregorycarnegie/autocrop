from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import rawpy
from PIL import Image
from PyQt6 import QtCore, QtGui, QtMultimedia, QtMultimediaWidgets, QtWidgets

import utils


class PandasModel(QtCore.QAbstractTableModel):
    def __init__(self, ui_main_window: QtWidgets.QMainWindow, mapping_line_edit: QtWidgets.QLineEdit,
                 table_view: QtWidgets.QTableView, combo_box_1: QtWidgets.QComboBox, combo_box_2: QtWidgets.QComboBox):
        QtCore.QAbstractTableModel.__init__(self)
        self.mapping_line_edit = mapping_line_edit
        self.table_view = table_view
        self.combo_box_1 = combo_box_1
        self.combo_box_2 = combo_box_2
        self.PANDAS_TYPES = [".xlsx", ".xlsm", ".xltx", ".xltm"]
        self._data = self.open_table(ui_main_window)
        self.default_directory = f"{Path.home()}\\Pictures"

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if index.isValid() and role == QtCore.Qt.ItemDataRole.DisplayRole:
            return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if orientation == QtCore.Qt.Orientation.Horizontal and role == QtCore.Qt.ItemDataRole.DisplayRole:
            return self._data.columns[col]
        return None

    def type_string(self):
        return f"All Files (*){''.join(f';;{_} Files (*{_})' for _ in np.sort(self.PANDAS_TYPES))}"

    def open_table(self, ui_main_window: QtWidgets.QMainWindow) -> (None | pd.DataFrame):
        # f_name = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File', self.default_directory, self.PANDAS_TYPES)
        f_name = QtWidgets.QFileDialog.getOpenFileName(ui_main_window, 'Open File', f'{Path.home()}\\Pictures',
                                                       self.type_string())
        self.mapping_line_edit.setText(f_name[0])
        # extension = os.path.splitext(self.mapping_line_edit.text())[1].lower()
        extension = Path(self.mapping_line_edit.text()).suffix.lower()
        if extension == ".csv":
            try:
                data = pd.read_csv(self.mapping_line_edit.text())
                return self.set_model(data)
            except FileNotFoundError:
                return None
        if extension in self.PANDAS_TYPES:
            try:
                data = pd.read_excel(self.mapping_line_edit.text())
                return self.set_model(data)
            except FileNotFoundError:
                return None
        return None

    def set_model(self, data):
        self.table_view.setModel(data)
        for _ in self._data.columns:
            self.combo_box_1.addItem(_)
            self.combo_box_2.addItem(_)
        return data


class Photo:
    def __init__(self, photo_line_edit: Optional[QtWidgets.QLineEdit] = None, main_window: Optional[QtWidgets.QMainWindow] = None) -> None:
        self.main_window = main_window
        self.photo_line_text = None if photo_line_edit is None else photo_line_edit.text()
        self.photo_line_edit = photo_line_edit
        self.PIL_TYPES, self.CV2_TYPES, self.RAW_TYPES = utils.PIL_TYPES, utils.CV2_TYPES, utils.RAW_TYPES
        self.default_directory = f"{Path.home()}\\Pictures"

    def setfileModel(self, ui_main_window: QtWidgets.QMainWindow, tree_view: QtWidgets.QTreeView) -> None:
        fileModel = QtGui.QFileSystemModel(ui_main_window)
        fileModel.setRootPath(self.default_directory)
        fileModel.setFilter(QtCore.QDir.Filter.NoDotAndDotDot | QtCore.QDir.Filter.Files)
        fileModel.setNameFilters(self.file_filter())

        tree_view.setModel(fileModel)
        tree_view.setRootIndex(fileModel.index(self.default_directory))

    def open(self, exposure: bool = True) -> (cv2.Mat | np.ndarray | None):
        if self.photo_line_text is None:
            return None
        # extension = os.path.splitext(self.photo_line_text)[1].lower()
        extension = Path(self.photo_line_text).suffix.lower()
        if extension in self.CV2_TYPES:
            # Try with cv2
            x = cv2.imread(self.photo_line_text)
            assert not isinstance(x, type(None)), 'image not found'
            if exposure:
                x = self.correct_exposure(x)
            return x

        if extension in self.PIL_TYPES:
            # Try with PIL
            with Image.open(self.photo_line_text) as img_orig:
                x = np.fromfile(img_orig)
                if exposure:
                    x = self.correct_exposure(x)
                return x

        if extension in self.RAW_TYPES:
            with rawpy.imread(self.photo_line_text) as raw:
                # Demosaic the RAW data using the bilinear method
                rgb = raw.postprocess(gamma=(1, 1), no_auto_bright=True, output_bps=8)
                # Convert the RGB data to BGR format for OpenCV
                x = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                if exposure:
                    x = self.correct_exposure(x)
                return x
        return None

    def display_folder(self, file_model: QtGui.QFileSystemModel, tree_view: QtWidgets.QTreeView,
                       line_edit: QtWidgets.QLineEdit,
                       line_edit_3: QtWidgets.QLineEdit, line_edit_4: QtWidgets.QLineEdit, h_slider_4: QtWidgets.QDial,
                       h_slider_3: QtWidgets.QDial, h_slider_2: QtWidgets.QDial, crop_label_2: utils.ImageWidget):
        f_name = QtWidgets.QFileDialog.getExistingDirectory(self.main_window, 'Select Directory',
                                                            self.default_directory)
        line_edit.setText(f_name)
        file_model.setRootPath(f_name)
        tree_view.setModel(file_model)
        tree_view.setRootIndex(file_model.index(f_name))
        try:
            # file = np.array([pic for pic in os.listdir(f_name) if os.path.splitext(pic)[1] in self.ALL_TYPES()])[0]
            files = np.fromiter(Path(f_name).glob('*.*'), Path)
            file = np.array([pic for pic in files if pic.suffix in self.ALL_TYPES()])
            file = dict(enumerate(file))[0]
            # file = np.array([pic for pic in files if Path(pic).suffix in self.ALL_TYPES()])[0]
            utils.display_crop(file, int(line_edit_3.text()), int(line_edit_4.text()),
                               h_slider_4.value(), h_slider_3.value(), h_slider_2.value(),
                               crop_label_2, self.ALL_TYPES())
        except (IndexError, FileNotFoundError):
            return

    def display(self, line_edit_3: QtWidgets.QLineEdit, line_edit_4: QtWidgets.QLineEdit,
                h_slider_4: QtWidgets.QDial, h_slider_3: QtWidgets.QDial, h_slider_2: QtWidgets.QDial,
                crop_label_1: utils.ImageWidget):
        f_name, _ = QtWidgets.QFileDialog.getOpenFileName(self.main_window, 'Open File', self.default_directory,
                                                          self.type_string())
        if isinstance(self.photo_line_edit, QtWidgets.QLineEdit):
            self.photo_line_edit.setText(f_name)
        else:
            return None
        try:
            utils.display_crop(Path(f_name), int(line_edit_3.text()), int(line_edit_4.text()),
                               h_slider_4.value(), h_slider_3.value(), h_slider_2.value(), crop_label_1)
        except ValueError:
            return None

    def ALL_TYPES(self):
        return np.concatenate([self.PIL_TYPES, self.CV2_TYPES, self.RAW_TYPES])
        # np.append(self.PIL_TYPES, self.CV2_TYPES)

    def file_filter(self):
        return np.array([f'*{file}' for file in self.ALL_TYPES()])

    def type_string(self):
        return f"All Files (*){''.join(f';;{_} Files (*{_})' for _ in np.sort(self.ALL_TYPES()))}"

    @staticmethod
    def correct_exposure(image_array: cv2.Mat | np.ndarray):
        # Convert to grayscale
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

        # Calculate histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

        # Get the average pixel value
        average_pixel_value = np.sum(hist * range(256)) / np.sum(hist)

        # Determine whether the image is overexposed or underexposed
        if average_pixel_value < 127:
            # Image is underexposed, increase brightness and contrast
            alpha = 1.5
            beta = 50
        elif average_pixel_value > 128:
            # Image is overexposed, decrease brightness and contrast
            alpha = 0.5
            beta = -50
        else:
            # Image is properly exposed
            return image_array

        # Apply brightness and contrast adjustment
        return cv2.convertScaleAbs(image_array, alpha=alpha, beta=beta)


"""
# class Table():
#     def __init__(self, mappingLineEdit: QtWidgets.QLineEdit) -> None:
#         self.mappingLineEdit = mappingLineEdit.text()
#         self.PANDAS = [".csv", ".xlsx", ".xlsm", ".xltx", ".xltm"]

#     def open_table(self) -> (None | pd.DataFrame):
#         extension = os.path.splitext(self.mappingLineEdit)[1].lower()
#         if extension.lower() == ".csv":
#             try:
#                 return pd.read_csv(self.mappingLineEdit)
#             except FileNotFoundError:
#                 return None
#         if extension.lower() in self.PANDAS[1:]:
#             try:
#                 return pd.read_excel(self.mappingLineEdit)
#             except FileNotFoundError:
#                 return None
#         return None
"""


class Video:
    def __init__(self, audio: QtMultimedia.QAudioOutput, video_widget: QtMultimediaWidgets.QVideoWidget,
                 media_player: QtMultimedia.QMediaPlayer, timeline_slider: QtWidgets.QSlider,
                 volume_slider: QtWidgets.QSlider, position_label: QtWidgets.QLabel) -> None:
        self.muted = False
        self.paused = False
        self.GAMMA_THRESHOLD = 0.001

        self.player, self.video_widget, self.audio = (media_player, video_widget, audio)
        self.create_mediaPlayer()

        self.timeline_slider = timeline_slider
        self.position_label = position_label
        self.volume_slider = volume_slider

        self.start_position, self.stop_position, self.step = 0.0, 0.0, 2

        self.VIDEO_TYPES = [".avi", ".m4v", ".mkv", ".mov", ".mp4", ".wmv"]
        self.VIDEO_SPEEDS = [1.0, 1.25, 1.5, 1.75, 2.0]
        self.speed = 0

        self.player.positionChanged.connect(self.position_changed)
        self.player.durationChanged.connect(self.duration_changed)
        self.volume_slider.sliderMoved.connect(self.volume_slider_changed)
        self.timeline_slider.sliderMoved.connect(self.player_slider_changed)

    def type_string(self):
        return f"All Files (*){''.join(f';;{_} Files (*{_})' for _ in np.sort(self.VIDEO_TYPES))}"

    def open_video(self, main_window: QtWidgets.QMainWindow, video_line_edit: QtWidgets.QLineEdit,
                   play_button: QtWidgets.QPushButton, crop_button: QtWidgets.QPushButton):
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(main_window, 'Open Video', '', 'Video files (*.mp4 *.avi)')
        if file_name != '':
            self.player.setSource(QtCore.QUrl.fromLocalFile(file_name))
            video_line_edit.setText(file_name)
            play_button.setEnabled(True)
            crop_button.setEnabled(True)

    @staticmethod
    def open_video_folder(main_window: QtWidgets.QMainWindow, destination_line_edit: QtWidgets.QLineEdit):
        folder_name = QtWidgets.QFileDialog.getExistingDirectory(main_window, 'Select Directory', f"{Path.home()}\\Videos")
        
        if folder_name != '':
            destination_line_edit.setText(folder_name)

    def create_mediaPlayer(self):
        self.player.setAudioOutput(self.audio)
        self.player.setVideoOutput(self.video_widget)

    def play_video(self, play_button: QtWidgets.QPushButton):

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

    def stop_btn(self):
        self.player.stop()

    def fastfwd(self):
        self.speed = max(1, self.speed)
        self.speed += 1
        self.speed = min(self.speed, len(self.VIDEO_SPEEDS) - 1)
        self.player.setPlaybackRate(self.VIDEO_SPEEDS[self.speed])

    def rewind(self):
        self.VIDEO_SPEEDS.reverse()
        x = [i - 1 for i in self.VIDEO_SPEEDS if i > 1]
        self.speed = min(1, self.speed)
        self.speed += 1
        self.speed = min(self.speed, len(x) - 1)
        self.player.setPlaybackRate(self.VIDEO_SPEEDS[self.speed])
        self.VIDEO_SPEEDS.reverse()

    def position_changed(self, position):
        if self.timeline_slider.maximum() != self.player.duration():
            self.timeline_slider.setMaximum(self.player.duration())

        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setValue(position)
        minutes, seconds = divmod(round(position / 1_000), 60)
        hours, minutes = divmod(minutes, 60)
        self.timeline_slider.blockSignals(False)

        self.position_label.setText(QtCore.QTime(hours, minutes, seconds).toString())

    def duration_changed(self, duration, timeline_slider: QtWidgets.QSlider, duration_label: QtWidgets.QLabel,
                         select_end_marker_button: QtWidgets.QPushButton):
        timeline_slider.setMaximum(duration)
        if duration >= 0:
            minutes, seconds = divmod(round(self.player.duration() / 1_000), 60)
            hours, minutes = divmod(minutes, 60)
            duration_label.setText(QtCore.QTime(hours, minutes, seconds).toString())
            select_end_marker_button.setText(QtCore.QTime(hours, minutes, seconds).toString())

    def player_slider_changed(self, position):
        self.player.setPosition(position)

    def volume_slider_changed(self, position):
        self.audio.setVolume(position)

    def volume_mute(self, volume_slider: QtWidgets.QSlider, mute_button: QtWidgets.QPushButton):
        if self.muted:
            self.audio.setMuted(True)
            volume_slider.setValue(0)
            mute_button.setIcon(QtGui.QIcon('resources\\icons\\multimedia_unmute.svg'))
        else:
            self.audio.setMuted(False)
            volume_slider.setValue(70)
            mute_button.setIcon(QtGui.QIcon('resources\\icons\\multimedia_mute.svg'))
        self.muted = not self.muted

    def goto_begining(self):
        self.player.setPosition(0)

    def goto_end(self):
        self.player.setPosition(self.player.duration())

    @staticmethod
    def set_marker_time(button: QtWidgets.QPushButton, position: int | float):
        minutes, seconds = divmod(round(position), 60)
        hours, minutes = divmod(minutes, 60)
        button.setText(QtCore.QTime(hours, minutes, seconds).toString())

    def set_startPosition(self, button: QtWidgets.QPushButton, timeline_slider: QtWidgets.QSlider):
        self.start_position = timeline_slider.value() / 1_000
        self.set_marker_time(button, self.start_position)

    def set_stopPosition(self, button: QtWidgets.QPushButton, timeline_slider: QtWidgets.QSlider):
        self.stop_position = timeline_slider.value() / 1_000
        self.set_marker_time(button, self.stop_position)

    def goto(self, marker_button: QtWidgets.QPushButton):
        x = np.array(marker_button.text().split(':')).astype(int)
        x = sum(60 ** (2 - i) * x[i] for i in range(len(x))) * 1_000
        if x >= self.player.duration():
            self.player.setPosition(self.player.duration())
        elif x == 0:
            self.player.setPosition(0)
        else:
            self.player.setPosition(int(x))
