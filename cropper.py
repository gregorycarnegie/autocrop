import cv2
import utils
import rawpy
import re
import numpy as np
import pandas as pd
from multiprocessing import cpu_count
from pathlib import Path
from PIL import Image
from PyQt6 import QtCore, QtGui, QtWidgets
from threading import Thread
from typing import Union, Optional

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

    def save_detection(self, path: Path, destination: Path, image: Path, width: int, height: int, checkbox: bool, confidence: int,
                       face: int, user_gam: int, top: int, bottom: int, left: int, right: int, radio: str,
                       r_choices: np.ndarray, new: Optional[str] = None) -> None:
        """
        This code first checks if bounding_box is not None, and if so, it proceeds to crop the image and create the
        destination directory if it doesn't already exist. It then constructs the file name using a ternary expression
        that appends the file extension to the file name if radio is equal to the first element in radio_choices,
        and appends radio itself if radio is not equal to the first element in radio_choices. The code then constructs
        the file path by joining the destination directory and the file name and saves the cropped image to the file
        using the imwrite() function. If bounding_box is None, the code calls the reject() function to reject the image.
        """
        # Save the cropped image if a face was detected
        if (bounding_box := utils.box_detect(path, width, height, confidence, face,
                                    top, bottom, left, right)) is not None:
            cropped_image = self.crop_image(path, bounding_box, width, height, checkbox)
            destination.mkdir(exist_ok=True)
            if image.suffix.lower() in utils.RAW_TYPES:
                file = f'{new or image.stem}{r_choices[2] if radio == r_choices[0] else radio}'
            else:
                file = f'{new or image.stem}{image.suffix if radio == r_choices[0] else radio}'
            
            file_path = destination.joinpath(file)
            is_tiff = file_path.suffix in {'.tif', '.tiff'}
            utils.save_image(cropped_image, file_path.as_posix(), user_gam, utils.GAMMA_THRESHOLD, is_tiff=is_tiff)
        else:
            utils.reject(path, destination, image)

    def crop(self, image: Path, destination: Path, width: QtWidgets.QLineEdit, height: QtWidgets.QLineEdit, checkbox: QtWidgets.QCheckBox,
             confidence: QtWidgets.QDial, face: QtWidgets.QDial, user_gam: QtWidgets.QDial, top: QtWidgets.QDial,
             bottom: QtWidgets.QDial, left: QtWidgets.QDial, right: QtWidgets.QDial, radio: str,
             radio_choices: np.ndarray, line_edit: Optional[str] = None, source_folder: Optional[Path] = None,
             new: Optional[str] = None) -> None:
        if line_edit is None and isinstance(new, str) and isinstance(source_folder, Path):
            # Data cropping
            path = source_folder.joinpath(image)
            self.save_detection(path, destination, image, int(width.text()), int(height.text()), checkbox.isChecked(), confidence.value(),
                                face.value(), user_gam.value(),  top.value(), bottom.value(), left.value(),
                                right.value(), radio, radio_choices, new)
        elif isinstance(line_edit, str):
            # Folder cropping
            source, image_name = Path(line_edit), image.name
            path = source.joinpath(image_name)
            self.save_detection(path, destination, Path(image_name), int(width.text()), int(height.text()), checkbox.isChecked(),
                                confidence.value(), face.value(), user_gam.value(), top.value(), bottom.value(),
                                left.value(), right.value(), radio, radio_choices)
        elif image.is_file():
            # File cropping
            self.save_detection(image, destination, image, int(width.text()), int(height.text()), checkbox.isChecked(), confidence.value(),
                                face.value(), user_gam.value(), top.value(), bottom.value(), left.value(),
                                right.value(), radio, radio_choices, new)

    def display_crop(self, img_path: Path, checkbox: QtWidgets.QCheckBox, wide: QtWidgets.QLineEdit, high: QtWidgets.QLineEdit,
                     conf: QtWidgets.QDial,
                     face_perc: QtWidgets.QDial, gam: QtWidgets.QDial, top: QtWidgets.QDial, bottom: QtWidgets.QDial,
                     left: QtWidgets.QDial, right: QtWidgets.QDial, image_widget: utils.ImageWidget,
                     file_types: Optional[np.ndarray] = None) -> None:
        if not img_path or img_path.as_posix() in {'', '.', None}:
            return None

        if img_path.is_dir():
            if not isinstance(file_types, np.ndarray):
                return None

            img_path = utils.get_first_file(img_path, file_types)
            if img_path is None:
                return None

        bounding_box = utils.box_detect(img_path, int(wide.text()), int(high.text()), conf.value(), face_perc.value(),
                                        top.value(), bottom.value(), left.value(), right.value())
        if bounding_box is None:
            return None

        photo_path = img_path.as_posix()
        
        if (extention := img_path.suffix.lower()) in utils.CV2_TYPES or extention in utils.PIL_TYPES:
            with Image.open(photo_path) as img:
                pic = utils.reorient_image_from_object(img)
                self.crop_and_set(pic, bounding_box, gam.value(), image_widget, checkbox.isChecked())
        elif extention in utils.RAW_TYPES:
            with rawpy.imread(photo_path) as raw:
                pic = utils.reorient_image_from_object(raw)
                self.crop_and_set(pic, bounding_box, gam.value(), image_widget, checkbox.isChecked())
        else:
            return None

    def crop_and_set(self, pic: Image.Image, bounding_box: np.ndarray, gam: int,
                     image_widget: utils.ImageWidget, checkbox: bool) -> None:
        try:
            pic_array = utils.correct_exposure(np.array(pic), checkbox)
            picture = Image.fromarray(pic_array)
            cropped_pic = np.array(picture.crop(bounding_box))
            cropped_pic = cv2.LUT(cropped_pic, utils.gamma(gam * utils.GAMMA_THRESHOLD).astype('uint8'))
            pic_array = cv2.cvtColor(cropped_pic, cv2.COLOR_BGR2RGB)
        except (cv2.error, Image.DecompressionBombError):
            return None
        # Convert numpy array to QImage
        height, width, channel = pic_array.shape
        bytes_per_line = channel * width
        qImg = QtGui.QImage(pic_array.data, width, height, bytes_per_line, QtGui.QImage.Format.Format_BGR888)
        # Set image to the image widget
        image_widget.setImage(QtGui.QPixmap.fromImage(qImg))

    def crop_image(self, image: Union[Path, np.ndarray], bounding_box: np.ndarray, width: int, height: int, checkbox: bool) -> cv2.Mat:
        if isinstance(image, Path):
            if image.suffix.lower() in utils.RAW_TYPES:
                raw = rawpy.imread(image.as_posix())
                pic = utils.reorient_image_from_object(raw)
                pic_array = utils.correct_exposure(pic, checkbox)
                cropped_pic = Image.fromarray(pic_array).crop(bounding_box)
            else:
                photo = Image.open(image.as_posix())
                pic = utils.reorient_image_from_object(photo)
                pic_array = utils.correct_exposure(pic, checkbox)
                cropped_pic = Image.fromarray(pic_array).crop(bounding_box)
                photo.close()
        else:
            pic = Image.fromarray(utils.correct_exposure(image))
            cropped_pic = pic.crop(bounding_box)

        pic_array = utils.correct_exposure(np.array(cropped_pic), checkbox)
        if isinstance(image, Path) and image.suffix.lower() in utils.RAW_TYPES:
            pic_array = pic_array[:, :, ::-1]
        else:
            pic_array = cv2.cvtColor(pic_array, cv2.COLOR_BGR2RGB)
        
        return cv2.resize(pic_array, (width, height), interpolation=cv2.INTER_AREA)

    def cropdir(self, file_amount: int, file_list: np.ndarray, destination: Path, width: QtWidgets.QLineEdit,
                height: QtWidgets.QLineEdit,  checkbox: QtWidgets.QCheckBox, confidence: QtWidgets.QDial, face: QtWidgets.QDial,
                gamma_dial: QtWidgets.QDial, top: QtWidgets.QDial, bottom: QtWidgets.QDial, left: QtWidgets.QDial,
                right: QtWidgets.QDial, radio_choice: str, line_edit: QtWidgets.QLineEdit,
                radio_choices: np.ndarray) -> None:
        for image in file_list:
            if self.end_task:
                break
            self.crop(image, destination, width, height, checkbox, confidence, face, gamma_dial, top, bottom, left, right,
                       radio_choice, radio_choices, line_edit=line_edit.text())
            self.bar_value += 1
            x = int(100 * self.bar_value / file_amount)
            self.folder_progress.emit(x)

        if (self.bar_value == file_amount or self.end_task) and self.message_box:
            self.finished.emit()
            self.message_box = False

    def crop_dir(self, file_list: np.ndarray, destination: str, width: QtWidgets.QLineEdit, height: QtWidgets.QLineEdit, checkbox: QtWidgets.QCheckBox,
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
            t = Thread(target=self.cropdir, args=(file_amount, array, Path(destination), width, height, checkbox, confidence,
                                                  face, gamma_dial, top, bottom, left, right, radio_choice, line_edit,
                                                  radio_choices))
            threads.append(t)
            t.start()

    def map_crop(self, files: int, source_folder: Path, old: np.ndarray, new: np.ndarray, destination: Path,
                 width: QtWidgets.QLineEdit, height: QtWidgets.QLineEdit,  checkbox: QtWidgets.QCheckBox, confidence: QtWidgets.QDial,
                 face: QtWidgets.QDial, user_gam: QtWidgets.QDial, top: QtWidgets.QDial, bottom: QtWidgets.QDial,
                 left: QtWidgets.QDial, right: QtWidgets.QDial,
                 radio: str, radio_choices: np.ndarray) -> None:
        for i, image in enumerate(old):
            if self.end_task:
                break
            self.crop(Path(image), destination, width, height, checkbox, confidence, face, user_gam, top, bottom, left, right,
                       radio, radio_choices, None, source_folder, new=new[i])
            self.bar_value += 1
            x = int(100 * self.bar_value / files)
            self.mapping_progress.emit(x)

        if (self.bar_value == files or self.end_task) and self.message_box:
            self.finished.emit()
            self.message_box = False

    def mapping_crop(self, source_folder: Path, data: pd.DataFrame, name_column: QtWidgets.QComboBox,
                     mapping: QtWidgets.QComboBox, destination: Path, width: QtWidgets.QLineEdit,
                     height: QtWidgets.QLineEdit,  checkbox: QtWidgets.QCheckBox, confidence: QtWidgets.QDial, face: QtWidgets.QDial,
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
                                                   width, height, checkbox, confidence,
                                                   face, gamma_dial, top, bottom,
                                                   left, right, radio, radio_choices))
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
                   width: QtWidgets.QLineEdit, height: QtWidgets.QLineEdit, checkbox: QtWidgets.QCheckBox, conf: QtWidgets.QDial,
                   face_perc: QtWidgets.QDial, gamma_dial: QtWidgets.QDial, top: QtWidgets.QDial,
                   bottom: QtWidgets.QDial, left: QtWidgets.QDial, right: QtWidgets.QDial,
                   position_label: QtWidgets.QLabel, timeline_slider: QtWidgets.QSlider, radio: str,
                   radio_options: np.ndarray) -> None:
        frame = self.grab_frame(timeline_slider, video_line_edit)
        if (bounding_box := utils.box_detect(frame, int(width.text()), int(height.text()), conf.value(),
                                             face_perc.value(), top.value(), bottom.value(), left.value(),
                                             right.value())) is not None:
            destination = Path(destination_line_edit_4.text())
            base_name = Path(video_line_edit.text()).stem

            cropped_image = self.crop_image(frame, bounding_box, int(width.text()), int(height.text()), checkbox.isChecked())
            destination.mkdir(exist_ok=True)
            position = re.sub(':', '_', position_label.text())
            file_path = destination.joinpath(
                f'{base_name} - ({position}){radio_options[2] if radio == radio_options[0] else radio}')
            is_tiff = file_path.suffix in {'.tif', '.tiff'}
            utils.save_image(cropped_image, file_path.as_posix(), gamma_dial.value(), utils.GAMMA_THRESHOLD,
                             is_tiff=is_tiff)
        else:
            return None

    def frame_extraction(self, video, output_dir: QtWidgets.QLineEdit, frame_number: int, frame_step: int,
                         width: QtWidgets.QLineEdit, height: QtWidgets.QLineEdit,  checkbox: QtWidgets.QCheckBox, confidence: QtWidgets.QDial,
                         face: QtWidgets.QDial, user_gam: QtWidgets.QDial, top: QtWidgets.QDial,
                         bottom: QtWidgets.QDial, left: QtWidgets.QDial, right: QtWidgets.QDial, radio: str,
                         radio_options: np.ndarray) -> None:
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = video.read()

        if not ret:
            return None

        destination = Path(output_dir.text())
        destination.mkdir(exist_ok=True)

        if (bounding_box := utils.box_detect(frame, int(width.text()), int(height.text()), confidence.value(),
                                             face.value(), top.value(), bottom.value(), left.value(),
                                             right.value())) is not None:
            cropped_image = self.crop_image(frame, bounding_box, int(width.text()), int(height.text()), checkbox.isChecked())
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
            file_enum = f'frame_{frame_number:06d}'
        else:
            cropped_image = frame
            file_enum = f'failed_frame_{frame_number:06d}'

        file_string = f'{file_enum}.jpg' if radio == radio_options[0] else file_enum + radio
        file_path = destination.joinpath(file_string)

        is_tiff = file_path.suffix in {'.tif', '.tiff'}
        utils.save_image(cropped_image, file_path.as_posix(), user_gam.value(), utils.GAMMA_THRESHOLD, is_tiff=is_tiff)
        frame_number += frame_step

    def extract_frames(self, video_path: QtWidgets.QLineEdit, start_time: Union[int, float],
                       end_time: Union[int, float], frame_step: int, output_dir: QtWidgets.QLineEdit,
                       width: QtWidgets.QLineEdit, height: QtWidgets.QLineEdit, checkbox: QtWidgets.QCheckBox, confidence: QtWidgets.QDial,
                       face: QtWidgets.QDial, user_gam: QtWidgets.QDial,  top: QtWidgets.QDial, bottom: QtWidgets.QDial,
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
            self.frame_extraction(video, output_dir, frame_number, frame_step, width, height, checkbox, confidence, face,
                                  user_gam,  top, bottom, left, right, radio, radio_options)
            self.video_progress.emit(int(i * dx))
        # Release the video file.
        video.release()
