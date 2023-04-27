import cv2
import utils
import numpy as np
import pandas as pd
from multiprocessing import cpu_count
from pathlib import Path
from PyQt6 import QtCore, QtWidgets
from threading import Thread
from typing import Union

class Cropper(QtCore.QObject):
    started, finished = QtCore.pyqtSignal(), QtCore.pyqtSignal()
    folder_progress = QtCore.pyqtSignal(int)
    mapping_progress = QtCore.pyqtSignal(int)
    video_progress = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super(Cropper, self).__init__(parent)
        self.bar_value = 0
        self.end_task = False
        self.message_box = True

        # Create video capture object
        self.cap = None
        self.start, self.stop, self.step = 0.0, 0.0, 2

    def cropdir(self, file_amount: int, file_list: np.ndarray, destination: Path, width: int, height: int,
                confidence: int, face: int, gamma_dial: int, top: int, bottom: int, left: int, right: int,
                radio_choice: str, line_edit: str, radio_choices: np.ndarray) -> None:
        for image in file_list:
            if self.end_task:
                break
            utils.crop(image, destination, width, height, confidence, face, gamma_dial, top, bottom, left, right,
                       radio_choice, radio_choices, line_edit=line_edit)
            self.bar_value += 1
            x = int(100 * self.bar_value / file_amount)
            self.folder_progress.emit(x)

        if (self.bar_value == file_amount or self.end_task) and self.message_box:
            self.finished.emit()
            self.message_box = False

    def crop_dir(self, file_list: np.ndarray, destination: str, width: QtWidgets.QLineEdit, height: QtWidgets.QLineEdit,
                 confidence: QtWidgets.QDial, face: QtWidgets.QDial, gamma_dial: QtWidgets.QDial, top: QtWidgets.QDial,
                 bottom: QtWidgets.QDial, left: QtWidgets.QDial, right: QtWidgets.QDial, radio_choice: str,
                 line_edit: QtWidgets.QLineEdit, radio_choices: np.ndarray) -> None:
        self.started.emit()
        split_array = np.array_split(file_list, cpu_count())
        threads = []
        file_amount = len(file_list)
        self.bar_value = 0
        self.folder_progress.emit(self.bar_value)
        for array in split_array:
            t = Thread(target=self.cropdir, args=(file_amount, array, Path(destination), int(width.text()),
                                                  int(height.text()), confidence.value(), face.value(),
                                                  gamma_dial.value(), top.value(), bottom.value(), left.value(),
                                                  right.value(), radio_choice, line_edit.text(), radio_choices))
            threads.append(t)
            t.start()

    def map_crop(self, files: int, source_folder: Path, old: np.ndarray, new: np.ndarray, destination: Path, width: int,
                 height: int, confidence: int, face: int, user_gam: int, top: int, bottom: int, left: int, right: int,
                 radio: str, radio_choices: np.ndarray) -> None:
        for i, image in enumerate(old):
            if self.end_task:
                break
            utils.crop(Path(image), destination, width, height, confidence, face, user_gam, top, bottom, left, right,
                       radio, radio_choices, None, source_folder, new=new[i])
            self.bar_value += 1
            x = int(100 * self.bar_value / files)
            self.mapping_progress.emit(x)

        if (self.bar_value == files or self.end_task) and self.message_box:
            self.finished.emit()
            self.message_box = False

    def mapping_crop(self, source_folder: Path, data: pd.DataFrame, name_column: QtWidgets.QComboBox,
                     mapping: QtWidgets.QComboBox, destination: Path, width: QtWidgets.QLineEdit,
                     height: QtWidgets.QLineEdit, confidence: QtWidgets.QDial, face: QtWidgets.QDial,
                     gamma_dial: QtWidgets.QDial, top: QtWidgets.QDial, bottom: QtWidgets.QDial,
                     left: QtWidgets.QDial, right: QtWidgets.QDial, radio: str, radio_choices: np.ndarray) -> None:
        self.started.emit()
        # Get the list of file names.
        file_list = np.array(data[name_column.currentText()]).astype(str)
        # Get the extensions of the file names.
        extensions = np.char.lower([Path(file).suffix for file in file_list])
        # Get the list of supported extensions.
        types = np.concatenate((utils.PIL_TYPES, utils.CV2_TYPES, utils.RAW_TYPES))
        # Create a mask that indicates which files have supported extensions.
        mask = np.in1d(extensions, types)
        # Split the file list and the mapping data into chunks.
        old_file_list = np.array_split(file_list[mask], cpu_count())
        new_file_list = np.array_split(np.array(data[mapping.currentText()])[mask], cpu_count())
        threads = []
        file_amount = file_list[mask].size
        self.bar_value = 0
        self.folder_progress.emit(self.bar_value)
        for i, _ in enumerate(old_file_list):
            t = Thread(target=self.map_crop, args=(file_amount, source_folder, _, new_file_list[i], destination,
                                                   int(width.text()), int(height.text()), confidence.value(),
                                                   face.value(), gamma_dial.value(), top.value(), bottom.value(),
                                                   left.value(), right.value(), radio, radio_choices))
            threads.append(t)
            t.start()

    def grab_frame(self, position_slider: QtWidgets.QSlider, video_line_edit: QtWidgets.QLineEdit) -> cv2.Mat:
        # Set video frame position to timelineSlider value
        self.cap = cv2.VideoCapture(video_line_edit.text())
        self.cap.set(cv2.CAP_PROP_POS_MSEC, position_slider.value())
        # Read frame from video capture object
        ret, frame = self.cap.read()
        self.cap.release()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def crop_frame(self, video_line_edit: QtWidgets.QLineEdit, destination_line_edit_4: QtWidgets.QLineEdit,
                   width: QtWidgets.QLineEdit, height: QtWidgets.QLineEdit, conf: QtWidgets.QDial,
                   face_perc: QtWidgets.QDial, gamma_dial: QtWidgets.QDial,  top: QtWidgets.QDial,
                   bottom: QtWidgets.QDial, left: QtWidgets.QDial, right: QtWidgets.QDial,
                   position_label: QtWidgets.QLabel, timeline_slider: QtWidgets.QSlider, radio: str,
                   radio_options: np.ndarray) -> None:
        frame = self.grab_frame(timeline_slider, video_line_edit)
        utils.crop_frame(frame, video_line_edit, destination_line_edit_4, int(width.text()), int(height.text()), conf,
                         face_perc, gamma_dial, top, bottom, left, right, position_label, radio, radio_options)
    
    def extract_frames(self, video_path: QtWidgets.QLineEdit, start_time: Union[int, float], end_time: Union[int, float],
                       frame_step: int, output_dir: QtWidgets.QLineEdit, width: QtWidgets.QLineEdit,
                       height: QtWidgets.QLineEdit, confidence: QtWidgets.QDial, face: QtWidgets.QDial,
                       user_gam: QtWidgets.QDial,  top: QtWidgets.QDial, bottom: QtWidgets.QDial,
                       left: QtWidgets.QDial, right: QtWidgets.QDial, radio: str, radio_options: np.ndarray) -> None:
        """
        Extracts frames from an MP4 video file between two specific times with a specified time step, and saves each frame as a JPG file.

        Args:
        - video_path (str): The path to the MP4 video file to extract frames from.
        - start_time (float): The start time of the frame extraction in seconds.
        - end_time (float): The end time of the frame extraction in seconds.
        - time_step (float): The time step between extracted frames in seconds.
        - output_dir (str): The directory to save the extracted frames in.
        """
        # Open the video file and get its properties.
        video = cv2.VideoCapture(video_path.text())
        fps = int(video.get(cv2.CAP_PROP_FPS))
        # Calculate the start and end frame numbers.
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        self.video_progress.emit(0)
        frame_number = start_frame
        dx = (end_frame - start_frame) / frame_step
        for i, frame_number in enumerate(range(frame_number, end_frame + 1, frame_step)):
            utils.frame_extraction(video, output_dir, frame_number, frame_step, int(width.text()), int(height.text()),
                                   confidence, face, user_gam,  top, bottom, left, right, radio, radio_options)
            self.video_progress.emit(int(i * dx))
        # Release the video file.
        video.release()
