from multiprocessing import cpu_count
from threading import Thread
import cv2
import re
import os
from pathlib import Path
from PyQt6 import QtCore, QtWidgets
import numpy as np
import pandas as pd
# from utils import crop, m_crop, gamma, box_detect, crop_image, crap, cropframe
import utils


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
                confidence: int, face: int, gamma_dial: int, radio_choice: str, line_edit: str,
                radio_choices: np.ndarray):
        for image in file_list:
            if self.end_task:
                break
            # crop(image, destination, width, height, confidence, face, gamma_dial, radio_choice,
            #      radio_choices, line_edit)
            utils.crap(image, destination, width, height, confidence, face, gamma_dial, radio_choice, radio_choices,
                       line_edit=line_edit)
            self.bar_value += 1
            x = int(100 * self.bar_value / file_amount)
            self.folder_progress.emit(x)

        if (self.bar_value == file_amount or self.end_task) and self.message_box:
            self.finished.emit()
            self.message_box = False

    def crop_dir(self, file_list: np.ndarray, destination: str, width: int, height: int, confidence: int, face: int,
                 gamma_dial: int, radio_choice: str, line_edit: str, radio_choices: np.ndarray):
        self.started.emit()
        split_array = np.array_split(file_list, cpu_count())
        threads = []
        file_amount = len(file_list)
        self.bar_value = 0
        self.folder_progress.emit(self.bar_value)
        for array in split_array:
            t = Thread(target=self.cropdir, args=(file_amount, array, Path(destination), width, height, confidence,
                                                  face, gamma_dial, radio_choice, line_edit, radio_choices))
            threads.append(t)
            t.start()

    def map_crop(self, files: int, source_folder: Path, old: np.ndarray, new: np.ndarray, destination: Path, width: int,
                 height: int, confidence: int, face: int, user_gam: int, radio: str, radio_choices: np.ndarray):
        for i, image in enumerate(old):
            if self.end_task:
                break
            # m_crop(source_folder, image, destination, width, height, confidence,
            #        face, user_gam, radio, radio_choices, new[i])
            utils.crap(image, destination, width, height, confidence, face, user_gam, radio,
                       radio_choices, None, source_folder, new=new[i])
            self.bar_value += 1
            x = int(100 * self.bar_value / files)
            self.mapping_progress.emit(x)

        if (self.bar_value == files or self.end_task) and self.message_box:
            self.finished.emit()
            self.message_box = False

    def mapping_crop(self, source_folder: str, data: pd.DataFrame, name_column: str, mapping: str,
                     destination: str, width: int, height: int, confidence: int, face: int, gamma_dial: int, radio: str,
                     radio_choices: np.ndarray):
        self.started.emit()
        file_list = np.array(data[name_column]).astype(str)
        extensions = np.char.lower([os.path.splitext(i)[1] for i in file_list])
        types = np.append(utils.PIL_TYPES, [utils.CV2_TYPES, utils.RAW_TYPES])

        r, s = np.meshgrid(extensions, types)
        g = r == s
        h = [g[:, i].any() for i in range(len(file_list))]

        old, new = np.array_split(file_list[h], cpu_count()), np.array_split(np.array(data[mapping])[h], cpu_count())
        threads = []
        file_amount = len(file_list[h])
        self.bar_value = 0
        self.folder_progress.emit(self.bar_value)
        for i, _ in enumerate(old):
            t = Thread(target=self.map_crop, args=(file_amount, source_folder, _, new[i], destination, width, height,
                                                   confidence, face, gamma_dial, radio, radio_choices))
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

    # def crop_frame(self, video_line_edit: QtWidgets.QLineEdit, destination_line_edit_4: QtWidgets.QLineEdit, wide, high,
    #                conf, face_perc, gamma_dial: QtWidgets.QDial, position_label: QtWidgets.QLabel,
    #                timeline_slider: QtWidgets.QSlider, radio: str, radio_options):
    #     # frame, frame_width, frame_height = self.grab_frame()
    #     frame = self.grab_frame(timeline_slider.value(), video_line_edit)

    #     if (bounding_box := box_detect(frame, wide, high, conf, face_perc)) is not None:
    #         # destination = destination_line_edit_4.text()
    #         destination = Path(destination_line_edit_4.text())
    #         # base_name = os.path.splitext(os.path.basename(video_line_edit.text()))[0]
    #         base_name = Path(video_line_edit.text()).stem

    #         cropped_image = crop_image(frame, bounding_box, wide, high)
    #         # os.makedirs(destination, exist_ok=True)
    #         destination.mkdir(exist_ok=True)
    #         position = re.sub(':', '_', position_label.text())
    #         # file = os.path.join(destination,
    #         #                     f'{base_name} - ({position}){radio_options[2] if radio == radio_options[0] else radio}')
    #         # file_path = os.path.join(destination, file)
    #         file_path = destination.joinpath(f'{base_name} - ({position}){radio_options[2] if radio == radio_options[0] else radio}')
    #         cv2.imwrite(file_path.as_posix(), cv2.LUT(cropped_image, gamma(gamma_dial.value() * GAMMA_THRESHOLD)))
    #     else:
    #         return None

    def crop_frame(self, video_line_edit: QtWidgets.QLineEdit, destination_line_edit_4: QtWidgets.QLineEdit, width: QtWidgets.QLineEdit,
                   height: QtWidgets.QLineEdit, conf: QtWidgets.QDial, face_perc: QtWidgets.QDial, gamma_dial: QtWidgets.QDial,
                   position_label: QtWidgets.QLabel, timeline_slider: QtWidgets.QSlider, radio: str,
                   radio_options: np.ndarray):
        frame = self.grab_frame(timeline_slider, video_line_edit)

        utils.cropframe(frame, video_line_edit, destination_line_edit_4, int(width.text()), int(height.text()), conf, face_perc, gamma_dial,
                        position_label, radio, radio_options)
    
    def extract_frames(self, video_path: QtWidgets.QLineEdit, start_time: int | float, end_time: int | float,
                       frame_step: int, output_dir: QtWidgets.QLineEdit, width: QtWidgets.QLineEdit,
                       height: QtWidgets.QLineEdit, confidence: QtWidgets.QDial, face: QtWidgets.QDial,
                       user_gam: QtWidgets.QDial, radio: str, radio_options: np.ndarray):
        """
        Extracts frames from an MP4 video file between two specific times with a specified time step, and saves each frame as a JPG file.

        Args:
        - video_path (str): The path to the MP4 video file to extract frames from.
        - start_time (float): The start time of the frame extraction in seconds.
        - end_time (float): The end time of the frame extraction in seconds.
        - time_step (float): The time step between extracted frames in seconds.
        - output_dir (str): The directory to save the extracted frames in.

        Returns:
        - None
        """
        # Open the video file and get its properties.
        video = cv2.VideoCapture(video_path.text())
        fps = int(video.get(cv2.CAP_PROP_FPS))

        # Calculate the start and end frame numbers.
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        # Set the initial frame number and the list of extracted frames.
        frame_number = start_frame

        self.bar_value = 0
        self.video_progress.emit(self.bar_value)

        x = np.linspace(0, 100, int((end_frame - start_frame)/frame_step)).astype(int)

        # Iterate over the frames in the video and extract the desired frames.
        while frame_number <= end_frame and not self.end_task:
            # # Set the current time in the video.
            # video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

            # # Read the current frame from the video.
            # ret, frame = video.read()

            # # Check if the frame was successfully read.
            # if not ret:
            #     break

            # # Save the current frame as a JPG file.
            # os.makedirs(output_dir.text(), exist_ok=True)
            # if (bounding_box := box_detect(frame, width, height, confidence.value(), face.value())) is not None:
            #     cropped_image = crop_image(frame, bounding_box, width, height)
            #     cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
            #     file = f"frame_{frame_number:06d}{radio_options[2] if radio == radio_options[0] else radio}"
            #     file_path = os.path.join(output_dir.text(), file)
            #     cv2.imwrite(file_path, cv2.LUT(cropped_image, gamma(user_gam.value() * GAMMA_THRESHOLD)))
            # else:
            #     output_path = os.path.join(output_dir.text(), f"failed_frame_{frame_number:06d}{radio_options[2] if radio == radio_options[0] else radio}")
            #     cv2.imwrite(output_path, frame)

            # # Move to the next frame.
            # frame_number += frame_step

            utils.frame_extraction(video, output_dir, frame_number, frame_step, int(width.text()), int(height.text()), confidence, face,
                                   user_gam, radio, radio_options)

            self.bar_value += 1
            self.video_progress.emit(x[self.bar_value])

        # Release the video file.
        video.release()
