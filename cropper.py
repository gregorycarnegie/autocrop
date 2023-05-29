import cv2
import dlib
import utils
import re
import numpy as np
from custom_widgets import ImageWidget
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

        self.face_workers = self.start_face_workers()

        # Create video capture object
        self.cap = None
        self.start, self.stop, self.step = 0.0, 0.0, 2

    def start_face_workers(self) -> list[tuple]:
        """Load the face detectors and shape predictors"""
        x = min(cpu_count(), 8)
        return [(dlib.get_frontal_face_detector(),
                 dlib.shape_predictor('resources\\models\\shape_predictor_68_face_landmarks.dat')) for _ in range(x)]

    def reset(self):
        self.end_task = False
        self.message_box = True
    
    def save_detection1(self, source_image: Path, image: Path, job: utils.Job, face_detector, predictor,
                        new: str) -> None:
        if (destination_path := job.destination_path()) is None:
            return None
        if (cropped_image := self.crop_image(source_image, job, face_detector, predictor)) is not None:
            file_path, is_tiff = utils.set_filename(image, destination_path, job.radio_choice(),
                                                    tuple(job.radio_options), new)
            utils.save_image(cropped_image, file_path.as_posix(), job.gamma.value(), utils.GAMMA_THRESHOLD,
                             is_tiff=is_tiff)
        else:
            utils.reject(source_image, destination_path, image)

    def save_detection2(self, source_image: Path, image_name: Path, job: utils.Job, face_detector, predictor) -> None:
        if (destination_path := job.destination_path()) is None:
            return None
        if (cropped_image := self.crop_image(source_image, job, face_detector, predictor)) is not None:
            file_path, is_tiff = utils.set_filename(image_name, destination_path, job.radio_choice(),
                                                    tuple(job.radio_options))
            utils.save_image(cropped_image, file_path.as_posix(), job.gamma.value(), utils.GAMMA_THRESHOLD,
                             is_tiff=is_tiff)
        else:
            utils.reject(source_image, destination_path, image_name)

    def save_detection3(self, source_image: Path, job: utils.Job, face_detector, predictor) -> None:
        if (destination_path := job.destination_path()) is None:
            return None
        if (cropped_image := self.crop_image(source_image, job, face_detector, predictor)) is not None:
            file_path, is_tiff = utils.set_filename(source_image, destination_path, job.radio_choice(),
                                                    tuple(job.radio_options))
            utils.save_image(cropped_image, file_path.as_posix(), job.gamma.value(), utils.GAMMA_THRESHOLD,
                             is_tiff=is_tiff)
        else:
            utils.reject(source_image, destination_path, source_image)

    def multi_save_detection1(self, source_image: Path, image: Path, job: utils.Job, new: str) -> None:
        if (destination_path := job.destination_path()) is None:
            return None
        if (cropped_images := self.multi_crop(source_image, job)) is None:
            utils.reject(source_image, destination_path, image)
        else:
            file_path, is_tiff = utils.set_filename(image, destination_path, job.radio_choice(), 
                                                    tuple(job.radio_options), new)
            for i, image in enumerate(cropped_images):
                new_file_path = file_path.with_stem(f'{file_path.stem}_{i}')
                utils.save_image(image, new_file_path.as_posix(), job.gamma.value(), utils.GAMMA_THRESHOLD,
                                 is_tiff=is_tiff)

    def multi_save_detection2(self, source_image: Path, image_name: Path, job: utils.Job) -> None:
        if (destination_path := job.destination_path()) is None:
            return None
        if (cropped_images := self.multi_crop(source_image, job)) is None:
            utils.reject(source_image, destination_path, image_name)
        else:
            file_path, is_tiff = utils.set_filename(image_name, destination_path, job.radio_choice(),
                                                    tuple(job.radio_options))
            for i, image in enumerate(cropped_images):
                new_file_path = file_path.with_stem(f'{file_path.stem}_{i}')
                utils.save_image(image, new_file_path.as_posix(), job.gamma.value(), utils.GAMMA_THRESHOLD,
                                 is_tiff=is_tiff)

    def multi_save_detection3(self, source_image: Path, job: utils.Job) -> None:
        if (destination_path := job.destination_path()) is None:
            return None

        if (cropped_images := self.multi_crop(source_image, job)) is None:
            utils.reject(source_image, destination_path, source_image)
        else:
            file_path, is_tiff = utils.set_filename(source_image, destination_path, job.radio_choice(),
                                                    tuple(job.radio_options))
            for i, image in enumerate(cropped_images):
                new_file_path = file_path.with_stem(f'{file_path.stem}_{i}')
                utils.save_image(image, new_file_path.as_posix(), job.gamma.value(), utils.GAMMA_THRESHOLD,
                                 is_tiff=is_tiff)

    def multi_crop(self, source_image: Path, job: utils.Job) -> Optional[list[cv2.Mat]]:
        img = utils.open_file(source_image, job.fix_exposure_job.isChecked(), job.autotilt_job.isChecked())

        detections, crop_positions = utils.multi_box_positions(img, job.sensitivity.value(), job.facepercent.value(),
                                                               job.width_value(), job.height_value(), job.top.value(),
                                                               job.bottom.value(), job.left.value(), job.right.value())
        # Check if any faces were detected
        if not np.any(detections[0, 0, :, 2] > job.sensitivity.value() * 0.01):
            return None

        images = [Image.fromarray(img).crop(crop_position) for crop_position in crop_positions]

        image_array = [np.array(image) for image in images]

        results = [utils.convert_color_space(image) for image in image_array]

        return [cv2.resize(result, (job.width_value(), job.height_value()), interpolation=cv2.INTER_AREA) for result in results]

    def crop(self, image: Path, job: utils.Job, face_detector, predictor, new: Optional[str] = None) -> None:
        common_widget_values = (job, face_detector, predictor)
        if job.table is not None and job.folder_path is not None and new is not None:
            # Data cropping
            path = Path(job.folder_path.text()).joinpath(image)
            if job.multiface_job.isChecked():
                self.multi_save_detection1(path, image, job, new)
            else:
                self.save_detection1(path, image, *common_widget_values, new)
        elif job.folder_path is not None:
            # Folder cropping
            source, image_name = Path(job.folder_path.text()), image.name
            path = source.joinpath(image_name)
            if job.multiface_job.isChecked():
                self.multi_save_detection2(path, Path(image_name), job)
            else:
                self.save_detection2(path, Path(image_name), *common_widget_values)
        elif job.multiface_job.isChecked():
            self.multi_save_detection3(image, job)
        else:
            self.save_detection3(image, *common_widget_values)

    def display_crop(self, job: utils.Job, line_edit: Union[Path, QtWidgets.QLineEdit], image_widget: ImageWidget) -> None:
        img_path = line_edit if isinstance(line_edit, Path) else Path(line_edit.text())
        # if input field is empty, then do nothing
        if not img_path or img_path.as_posix() in {'', '.', None}:
            return None
        
        # if width or height fields are empty, then do nothing
        if not job.width.text() or not job.height.text():
            return None

        if img_path.is_dir():
            first_file = utils.get_first_file(img_path)
            if first_file is None:
                return None
            img_path = first_file

        pic_array = utils.open_file(img_path, job.fix_exposure_job.isChecked(), job.autotilt_job.isChecked(), *self.face_workers[0])
        if pic_array is None:
            return None
        
        if job.multiface_job.isChecked():
            # TODO: add gamm adjustment for multiface
            adjusted = utils.convert_color_space(pic_array)
            pic = utils.multi_box(adjusted, job.sensitivity.value(), job.facepercent.value(), job.width_value(), job.height_value(),
                                  job.top.value(), job.bottom.value(), job.left.value(), job.right.value())
            
            utils.display_image_on_widget(pic, image_widget)
        else:
            bounding_box = utils.box_detect_numpy(pic_array, job.width_value(), job.height_value(), job.sensitivity.value(),
                                                  job.facepercent.value(), job.top.value(), job.bottom.value(), job.left.value(),
                                                  job.right.value())
            if bounding_box is None:
                return None

            self.crop_and_set(pic_array, bounding_box, job.gamma.value(), image_widget)

    @staticmethod
    def _numpy_array_crop(image: np.ndarray, bounding_box: tuple[int, int, int, int]) -> np.ndarray:
        picture = Image.fromarray(image)
        return np.array(picture.crop(bounding_box))

    def crop_and_set(self, image: np.ndarray, bounding_box: tuple[int, int, int, int], gamma: int,
                     image_widget: ImageWidget) -> None:
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
            cropped_image = self._numpy_array_crop(image, bounding_box)
            adjusted_image = utils.adjust_gamma(cropped_image, gamma)
            final_image = utils.convert_color_space(adjusted_image)
        except (cv2.error, Image.DecompressionBombError):
            return None

        utils.display_image_on_widget(final_image, image_widget)

    @staticmethod
    def crop_image(image: Union[Path, cv2.Mat, np.ndarray], job: utils.Job, face_detector, predictor) -> cv2.Mat:
        if isinstance(image, Path):
            pic_array = utils.open_file(image, job.fix_exposure_job.isChecked(), job.autotilt_job.isChecked(),
                                        face_detector, predictor)
            bounding_box = utils.box_detect_numpy(
                pic_array, job.width_value(), job.height_value(), job.sensitivity.value(), job.facepercent.value(),
                job.top.value(), job.bottom.value(), job.left.value(), job.right.value())
            if bounding_box is None:
                return None

        else:
            pic_array = image
            if job.fix_exposure_job.isChecked():
                pic_array = utils.correct_exposure(pic_array, job.fix_exposure_job.isChecked())

            if job.autotilt_job.isChecked():
                pic_array = utils.align_head(pic_array, face_detector, predictor)
            
            bounding_box = utils.box_detect_numpy(
                pic_array, job.width_value(), job.height_value(), job.sensitivity.value(), job.facepercent.value(),
                job.top.value(), job.bottom.value(), job.left.value(), job.right.value())

        cropped_pic = Image.fromarray(pic_array).crop(bounding_box)
        if len(cropped_pic.getbands()) >= 3:
            result = utils.convert_color_space(np.array(cropped_pic))
        else:
            result = np.array(cropped_pic)

        return cv2.resize(result, (job.width_value(), job.height_value()), interpolation=cv2.INTER_AREA)

    def folder_worker(self, file_amount: int, file_list: np.ndarray, job: utils.Job, face_detector, predictor) -> None:
        for image in file_list:
            if self.end_task:
                break
            self.crop(image, job, face_detector, predictor)
            self.bar_value += 1
            x = int(100 * self.bar_value / file_amount)
            self.folder_progress.emit(x)

        if (self.bar_value == file_amount or self.end_task) and self.message_box:
            self.folder_finished.emit()
            self.message_box = False

    def crop_dir(self, job: utils.Job) -> None:
        self.folder_started.emit()
        split_array = np.array_split(job.file_list(), min(cpu_count(), 8))
        threads = []
        file_amount = len(job.file_list())
        self.bar_value = 0
        self.folder_progress.emit(self.bar_value)
        for i, array in enumerate(split_array):
            detector, predictor = self.face_workers[i]
            t = Thread(target=self.folder_worker, args=(file_amount, array, job, detector, predictor))
            threads.append(t)
            t.start()

    def mapping_worker(self, files: int, old: np.ndarray, new: np.ndarray, job: utils.Job,
                       face_detector, predictor) -> None:
        for i, image in enumerate(old):
            if self.end_task:
                break
            self.crop(Path(image), job, face_detector, predictor, new=new[i])
            self.bar_value += 1
            x = int(100 * self.bar_value / files)
            self.mapping_progress.emit(x)

        if (self.bar_value == files or self.end_task) and self.message_box:
            self.mapping_finished.emit()
            self.message_box = False

    def mapping_crop(self, job: utils.Job) -> None:
        if (g := job.file_list_to_numpy()) is None:
            return None
        file_list1, file_list2 = g
        # Get the extensions of the file names and 
        # Create a mask that indicates which files have supported extensions.
        mask, file_amount = utils.mask_extensions(file_list1)
        # Split the file lists and the mapping data into chunks.
        old_file_list, new_file_list = utils.split_by_cpus(mask, min(cpu_count(), 8), file_list1, file_list2)
        self.mapping_started.emit()
        self.bar_value = 0
        self.folder_progress.emit(self.bar_value)
        threads = []
        for i, _ in enumerate(old_file_list):
            detector, predictor = self.face_workers[i]
            t = Thread(target=self.mapping_worker, args=(file_amount, _, new_file_list[i], job, detector, predictor))
            threads.append(t)
            t.start()

    @cache
    def grab_frame(self, position_slider: int, video_line_edit: str) -> Optional[cv2.Mat]:
        # Set video frame position to timelineSlider value
        self.cap = cv2.VideoCapture(video_line_edit)
        self.cap.set(cv2.CAP_PROP_POS_MSEC, position_slider)
        # Read frame from video capture object
        ret, frame = self.cap.read()
        if not ret:
            return None
        self.cap.release()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def crop_frame(self, job: utils.Job, position_label: QtWidgets.QLabel, timeline_slider: QtWidgets.QSlider) -> None:
        if job.video_path is None or job.destination is None:
            return None
        if (frame := self.grab_frame(timeline_slider.value(), job.video_path.text())) is not None:
            detector, predictor = self.face_workers[0]
            if (cropped_image := self.crop_image(frame, job, detector, predictor)) is not None:        
                destination, base_name = Path(job.destination.text()), Path(job.video_path.text()).stem

                destination.mkdir(exist_ok=True)
                position = re.sub(':', '_', position_label.text())
                file_path = destination.joinpath(
                    f'{base_name} - ({position}){job.radio_options[2] if job.radio_choice() == job.radio_options[0] else job.radio_choice()}')
                is_tiff = file_path.suffix in {'.tif', '.tiff'}
                utils.save_image(cropped_image, file_path.as_posix(), job.gamma.value(), utils.GAMMA_THRESHOLD,
                                 is_tiff=is_tiff)
            else:
                return None
        else:
            return None

    def frame_extraction(self, video, frame_number: int, job: utils.Job, progress_callback) -> None:
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = video.read()

        if not ret:
            return None

        if job.destination is None:
            return None
        destination = Path(job.destination.text())
        destination.mkdir(exist_ok=True)

        detector, predictor = self.face_workers[0]
        if (cropped_image := self.crop_image(frame, job, detector, predictor)) is not None:        
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
            file_enum = f'frame_{frame_number:06d}'
        else:
            cropped_image = frame
            file_enum = f'failed_frame_{frame_number:06d}'

        file_string = f'{file_enum}.jpg' if job.radio_choice() == job.radio_options[0] else file_enum + job.radio_choice()
        file_path = destination.joinpath(file_string)

        is_tiff = file_path.suffix in {'.tif', '.tiff'}
        utils.save_image(cropped_image, file_path.as_posix(), job.gamma.value(), utils.GAMMA_THRESHOLD, is_tiff=is_tiff)
        
        progress_callback()

    def extract_frames(self, job: utils.Job) -> None:
        if job.video_path is None or job.start_position is None or job.stop_position is None:
            return None
        self.video_started.emit()
        video = cv2.VideoCapture(job.video_path.text())
        fps = int(video.get(cv2.CAP_PROP_FPS))
        start_frame = int(job.start_position * fps)
        end_frame = int(job.stop_position * fps)
        self.video_progress.emit(0)

        frame_numbers = np.arange(start_frame, end_frame + 1)

        self.bar_value = 0
        def progress_callback():
            self.bar_value += 1
            x = 100 * self.bar_value // frame_numbers.size
            self.video_progress.emit(x)
        
        for frame_number in frame_numbers:
            self.frame_extraction(video, frame_number, job, progress_callback)
            if (self.bar_value == frame_numbers.size or self.end_task) and self.message_box:
                self.video_finished.emit()
                self.message_box = False

        video.release()
