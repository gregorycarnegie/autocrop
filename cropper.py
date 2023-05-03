import cv2
import utils
import rawpy
import re
import numpy as np
import pandas as pd
from functools import cache
from multiprocessing import cpu_count
from pathlib import Path
from PIL import Image
from PyQt6 import QtCore, QtWidgets
from threading import Thread, Lock
from typing import Union, Optional


class Cropper(QtCore.QObject):
    folder_started = QtCore.pyqtSignal()
    folder_finished = QtCore.pyqtSignal()
    mapping_started = QtCore.pyqtSignal()
    mapping_finished = QtCore.pyqtSignal()
    video_started = QtCore.pyqtSignal()
    video_finished = QtCore.pyqtSignal()
    folder_progress = QtCore.pyqtSignal(int)
    mapping_progress = QtCore.pyqtSignal(int)
    video_progress = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super(Cropper, self).__init__(parent)
        self.bar_value = 0
        self.end_task = False
        self.message_box = True
        self.video_lock = Lock()

        # Create video capture object
        self.cap = None
        self.start, self.stop, self.step = 0.0, 0.0, 2

    def reset(self):
        self.end_task = False
        self.message_box = True
    
    def save_detection(self, path: Path, destination: Path, image: Path, width: int, height: int, checkbox: bool,
                       confidence: int, face: int, user_gam: int, top: int, bottom: int, left: int, right: int,
                       radio: str, r_choices: np.ndarray, new: Optional[str] = None) -> None:
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
            file_path, is_tiff = utils.set_filename(image, destination, radio, tuple(r_choices), new)
            utils.save_image(cropped_image, file_path.as_posix(), user_gam, utils.GAMMA_THRESHOLD, is_tiff=is_tiff)
        else:
            utils.reject(path, destination, image)

    def crop(self, image: Path, destination: Path, width: QtWidgets.QLineEdit, height: QtWidgets.QLineEdit,
             checkbox: QtWidgets.QCheckBox, confidence: QtWidgets.QDial, face: QtWidgets.QDial,
             user_gam: QtWidgets.QDial, top: QtWidgets.QDial, bottom: QtWidgets.QDial, left: QtWidgets.QDial,
             right: QtWidgets.QDial, radio: str, radio_choices: np.ndarray, line_edit: Optional[str] = None,
             source_folder: Optional[Path] = None,
             new: Optional[str] = None) -> None:
        common_widget_values = (int(width.text()), int(height.text()), checkbox.isChecked(), confidence.value(),
                                face.value(), user_gam.value(),  top.value(), bottom.value(), left.value(),
                                right.value(), radio, radio_choices)
        if line_edit is None and isinstance(new, str) and isinstance(source_folder, Path):
            # Data cropping
            path = source_folder.joinpath(image)
            self.save_detection(path, destination, image, *common_widget_values, new)
        elif isinstance(line_edit, str):
            # Folder cropping
            source, image_name = Path(line_edit), image.name
            path = source.joinpath(image_name)
            self.save_detection(path, destination, Path(image_name), *common_widget_values)
        elif image.is_file():
            # File cropping
            self.save_detection(image, destination, image, *common_widget_values, new)

    def display_crop(self, img_path: Path, checkbox: QtWidgets.QCheckBox, wide: QtWidgets.QLineEdit,
                     high: QtWidgets.QLineEdit, conf: QtWidgets.QDial, face_perc: QtWidgets.QDial, gam: QtWidgets.QDial,
                     top: QtWidgets.QDial, bottom: QtWidgets.QDial, left: QtWidgets.QDial, right: QtWidgets.QDial,
                     image_widget: utils.ImageWidget, file_types: Optional[np.ndarray] = None) -> None:
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

        if isinstance(img_path, Path):
            photo_path = img_path.as_posix()
        else:
            return None
        
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

    @staticmethod
    def _numpy_array_crop(image: np.ndarray, bounding_box: np.ndarray) -> np.ndarray:
        picture = Image.fromarray(image)
        return np.array(picture.crop(bounding_box))

    def crop_and_set(self, image: np.ndarray, bounding_box: np.ndarray, gamma: int, image_widget: utils.ImageWidget,
                     exposure_correction: bool) -> None:
        """
        Crop the given image using the bounding box, adjust its exposure and gamma, and set it to an image widget.

        :param image: The input image as a numpy array.
        :param bounding_box: The bounding box coordinates to crop the image.
        :param gamma: The gamma value for gamma correction.
        :param image_widget: The image widget to display the processed image.
        :param exposure_correction: A boolean flag to enable or disable exposure correction.
        :return: None
        """
        try:
            processed_image = utils.correct_exposure(image, exposure_correction)
            cropped_image = self._numpy_array_crop(processed_image, bounding_box)
            adjusted_image = utils.adjust_gamma(cropped_image, gamma)
            final_image = utils.convert_color_space(adjusted_image)
        except (cv2.error, Image.DecompressionBombError):
            return None

        utils.display_image_on_widget(final_image, image_widget)

    @staticmethod
    def crop_image(image: Union[Path, np.ndarray], bounding_box: np.ndarray, width: int, height: int,
                   checkbox: bool) -> cv2.Mat:
        if isinstance(image, Path):
            if image.suffix.lower() in utils.RAW_TYPES:
                raw = rawpy.imread(image.as_posix())
                cropped_pic = utils.preprocess_image(raw, bounding_box, checkbox)
            else:
                photo = Image.open(image.as_posix())
                cropped_pic = utils.preprocess_image(photo, bounding_box, checkbox)
                photo.close()
        else:
            pic_array = utils.correct_exposure(image, checkbox)
            cropped_pic = Image.fromarray(pic_array).crop(bounding_box)

        if len(cropped_pic.getbands()) >= 3:
            result = utils.convert_color_space(np.array(cropped_pic))
        else:
            result = np.array(cropped_pic)
        
        return cv2.resize(result, (width, height), interpolation=cv2.INTER_AREA)

    def folder_worker(self, file_amount: int, file_list: np.ndarray, destination: Path, width: QtWidgets.QLineEdit,
                height: QtWidgets.QLineEdit,  checkbox: QtWidgets.QCheckBox, confidence: QtWidgets.QDial,
                face: QtWidgets.QDial, gamma_dial: QtWidgets.QDial, top: QtWidgets.QDial, bottom: QtWidgets.QDial,
                left: QtWidgets.QDial, right: QtWidgets.QDial, radio_choice: str, line_edit: QtWidgets.QLineEdit,
                radio_choices: np.ndarray) -> None:
        for image in file_list:
            if self.end_task:
                break
            self.crop(image, destination, width, height, checkbox, confidence, face, gamma_dial, top, bottom, left,
                      right, radio_choice, radio_choices, line_edit=line_edit.text())
            self.bar_value += 1
            x = int(100 * self.bar_value / file_amount)
            self.folder_progress.emit(x)

        if (self.bar_value == file_amount or self.end_task) and self.message_box:
            self.folder_finished.emit()
            self.message_box = False

    def crop_dir(self, file_list: np.ndarray, destination: str, width: QtWidgets.QLineEdit, height: QtWidgets.QLineEdit,
                 checkbox: QtWidgets.QCheckBox, confidence: QtWidgets.QDial, face: QtWidgets.QDial,
                 gamma_dial: QtWidgets.QDial, top: QtWidgets.QDial, bottom: QtWidgets.QDial, left: QtWidgets.QDial,
                 right: QtWidgets.QDial, radio_choice: str, line_edit: QtWidgets.QLineEdit,
                 radio_choices: np.ndarray) -> None:
        self.folder_started.emit()
        split_array = np.array_split(file_list, cpu_count())
        threads = []
        file_amount = len(file_list)
        self.bar_value = 0
        self.folder_progress.emit(self.bar_value)
        for array in split_array:
            t = Thread(target=self.folder_worker,
                       args=(file_amount, array, Path(destination), width, height, checkbox, confidence, face,
                             gamma_dial, top, bottom, left, right, radio_choice, line_edit, radio_choices))
            threads.append(t)
            t.start()

    def mapping_worker(self, files: int, source_folder: Path, old: np.ndarray, new: np.ndarray, destination: Path,
                       width: QtWidgets.QLineEdit, height: QtWidgets.QLineEdit,  checkbox: QtWidgets.QCheckBox,
                       confidence: QtWidgets.QDial, face: QtWidgets.QDial, user_gam: QtWidgets.QDial, top: QtWidgets.QDial,
                       bottom: QtWidgets.QDial, left: QtWidgets.QDial, right: QtWidgets.QDial, radio: str,
                       radio_choices: np.ndarray) -> None:
        for i, image in enumerate(old):
            if self.end_task:
                break
            self.crop(Path(image), destination, width, height, checkbox, confidence, face, user_gam, top, bottom, left,
                      right, radio, radio_choices, None, source_folder, new=new[i])
            self.bar_value += 1
            x = int(100 * self.bar_value / files)
            self.mapping_progress.emit(x)

        if (self.bar_value == files or self.end_task) and self.message_box:
            self.mapping_finished.emit()
            self.message_box = False

    def mapping_crop(self, source_folder: Path, data: pd.DataFrame, name_column: QtWidgets.QComboBox,
                     mapping: QtWidgets.QComboBox, destination: Path, width: QtWidgets.QLineEdit,
                     height: QtWidgets.QLineEdit,  checkbox: QtWidgets.QCheckBox, confidence: QtWidgets.QDial,
                     face: QtWidgets.QDial, gamma_dial: QtWidgets.QDial, top: QtWidgets.QDial, bottom: QtWidgets.QDial,
                     left: QtWidgets.QDial, right: QtWidgets.QDial, radio: str, radio_choices: np.ndarray) -> None:
        self.mapping_started.emit()
        # Get the list of file names.
        file_list = np.array(data[name_column.currentText()]).astype(str)
        # Get the extensions of the file names.
        extensions = np.char.lower([Path(file).suffix for file in file_list])
        # Create a mask that indicates which files have supported extensions.
        mask = np.in1d(extensions, utils.IMAGE_TYPES)
        # Split the file list and the mapping data into chunks.
        old_file_list = np.array_split(file_list[mask], cpu_count())
        new_file_list = np.array_split(np.array(data[mapping.currentText()])[mask], cpu_count())
        threads = []
        file_amount = file_list[mask].size
        self.bar_value = 0
        self.folder_progress.emit(self.bar_value)
        for i, _ in enumerate(old_file_list):
            t = Thread(target=self.mapping_worker,
                       args=(file_amount, source_folder, _, new_file_list[i], destination, width, height, checkbox,
                             confidence, face, gamma_dial, top, bottom, left, right, radio, radio_choices))
            threads.append(t)
            t.start()

    @cache
    def grab_frame(self, position_slider: int, video_line_edit: str) -> cv2.Mat:
        # Set video frame position to timelineSlider value
        self.cap = cv2.VideoCapture(video_line_edit)
        self.cap.set(cv2.CAP_PROP_POS_MSEC, position_slider)
        # Read frame from video capture object
        ret, frame = self.cap.read()
        self.cap.release()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def crop_frame(self, video_line_edit: QtWidgets.QLineEdit, destination_line_edit_4: QtWidgets.QLineEdit,
                   width: QtWidgets.QLineEdit, height: QtWidgets.QLineEdit, checkbox: QtWidgets.QCheckBox,
                   conf: QtWidgets.QDial, face_perc: QtWidgets.QDial, gamma_dial: QtWidgets.QDial, top: QtWidgets.QDial,
                   bottom: QtWidgets.QDial, left: QtWidgets.QDial, right: QtWidgets.QDial,
                   position_label: QtWidgets.QLabel, timeline_slider: QtWidgets.QSlider, radio: str,
                   radio_options: np.ndarray) -> None:
        frame = self.grab_frame(timeline_slider.value(), video_line_edit.text())
        if (bounding_box := utils.box_detect_frame(frame, int(width.text()), int(height.text()), conf.value(),
                                                   face_perc.value(), top.value(), bottom.value(), left.value(),
                                                   right.value())) is not None:
            destination, base_name = Path(destination_line_edit_4.text()), Path(video_line_edit.text()).stem

            cropped_image = self.crop_image(frame, bounding_box, int(width.text()), int(height.text()),
                                            checkbox.isChecked())
            destination.mkdir(exist_ok=True)
            position = re.sub(':', '_', position_label.text())
            file_path = destination.joinpath(
                f'{base_name} - ({position}){radio_options[2] if radio == radio_options[0] else radio}')
            is_tiff = file_path.suffix in {'.tif', '.tiff'}
            utils.save_image(cropped_image, file_path.as_posix(), gamma_dial.value(), utils.GAMMA_THRESHOLD,
                             is_tiff=is_tiff)
        else:
            return None

    def frame_extraction(self, video, output_dir: QtWidgets.QLineEdit, frame_number: int, width: QtWidgets.QLineEdit,
                         height: QtWidgets.QLineEdit,  checkbox: QtWidgets.QCheckBox, confidence: QtWidgets.QDial,
                         face: QtWidgets.QDial, user_gam: QtWidgets.QDial, top: QtWidgets.QDial,
                         bottom: QtWidgets.QDial, left: QtWidgets.QDial, right: QtWidgets.QDial, radio: str,
                         radio_options: np.ndarray, progress_callback) -> None:
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = video.read()

        if not ret:
            return None

        destination = Path(output_dir.text())
        destination.mkdir(exist_ok=True)

        if (bounding_box := utils.box_detect_frame(frame, int(width.text()), int(height.text()), confidence.value(),
                                                   face.value(), top.value(), bottom.value(), left.value(),
                                                   right.value())) is not None:
            cropped_image = self.crop_image(frame, bounding_box, int(width.text()), int(height.text()),
                                            checkbox.isChecked())
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
            file_enum = f'frame_{frame_number:06d}'
        else:
            cropped_image = frame
            file_enum = f'failed_frame_{frame_number:06d}'

        file_string = f'{file_enum}.jpg' if radio == radio_options[0] else file_enum + radio
        file_path = destination.joinpath(file_string)

        is_tiff = file_path.suffix in {'.tif', '.tiff'}
        utils.save_image(cropped_image, file_path.as_posix(), user_gam.value(), utils.GAMMA_THRESHOLD, is_tiff=is_tiff)
        
        progress_callback()

    def extract_frames(self, video_path: QtWidgets.QLineEdit, start_time: Union[int, float], end_time: Union[int, float],
                       output_dir: QtWidgets.QLineEdit, width: QtWidgets.QLineEdit, height: QtWidgets.QLineEdit,
                       checkbox: QtWidgets.QCheckBox, confidence: QtWidgets.QDial, face: QtWidgets.QDial,
                       user_gam: QtWidgets.QDial, top: QtWidgets.QDial, bottom: QtWidgets.QDial, left: QtWidgets.QDial,
                       right: QtWidgets.QDial, radio: str, radio_options: np.ndarray) -> None:
        self.video_started.emit()
        video = cv2.VideoCapture(video_path.text())
        fps = int(video.get(cv2.CAP_PROP_FPS))
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        self.video_progress.emit(0)

        frame_numbers = np.arange(start_frame, end_frame + 1)

        self.bar_value = 0
        def progress_callback():
            self.bar_value += 1
            x = int(100 * self.bar_value / frame_numbers.size)
            self.video_progress.emit(x)
        
        for frame_number in frame_numbers:
            self.frame_extraction(video, output_dir, frame_number, width, height, checkbox, confidence, face,
                                  user_gam, top, bottom, left, right, radio, radio_options, progress_callback)
            
            if (self.bar_value == frame_numbers.size or self.end_task) and self.message_box:
                self.video_finished.emit()
                self.message_box = False

        video.release()
