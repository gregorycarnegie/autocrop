import re
from functools import cache
from multiprocessing import cpu_count
from pathlib import Path
from threading import Thread, Lock
from typing import Any, Callable, List, Union, Optional, Tuple

import cv2
import cv2.typing as cvt
import numpy as np
import numpy.typing as npt
from PIL import Image
from PyQt6.QtCore import pyqtSignal, QObject
from PyQt6.QtWidgets import QLabel, QLineEdit, QSlider

from . import utils
from .enums import FunctionType
from .face_worker import FaceWorker
from .image_widget import ImageWidget
from .job import Job


class Cropper(QObject):
    folder_started = pyqtSignal()
    folder_finished = pyqtSignal()
    mapping_started = pyqtSignal()
    mapping_finished = pyqtSignal()
    video_started = pyqtSignal()
    video_finished = pyqtSignal()
    folder_progress = pyqtSignal(object)
    mapping_progress = pyqtSignal(object)
    video_progress = pyqtSignal(object)

    def __init__(self, parent: Optional[QObject]=None):
        super(Cropper, self).__init__(parent)
        self.bar_value_f = 0
        self.bar_value_m = 0
        self.bar_value_v = 0

        self.end_f_task = False
        self.end_m_task = False
        self.end_v_task = False

        self.message_box_f = True
        self.message_box_m = True
        self.message_box_v = True

        self.video_lock = Lock()
        self.face_workers = self.start_face_workers()

        # Create video capture object
        self.cap = None
        self.start, self.stop, self.step = 0.0, 0.0, 2

    @staticmethod
    def start_face_workers() -> List[FaceWorker]:
        """Load the face detectors and shape predictors"""
        return [FaceWorker() for _ in range(min(cpu_count(), 8))]

    def reset_f_task(self):
        self.bar_value_f = 0
        self.end_f_task = False
        self.message_box_f = True

    def reset_m_task(self):
        self.bar_value_m = 0
        self.end_m_task = False
        self.message_box_m = True

    def reset_v_task(self):
        self.bar_value_v = 0
        self.end_v_task = False
        self.message_box_v = True
    
    def save_detection(self, source_image: Path,
                       job: Job,
                       face_worker: FaceWorker,
                       image_name: Optional[Path] = None,
                       new: Optional[str] = None) -> None:
        if (destination_path := job.get_destination()) is None: return None
        if image_name is None and new is None:
            if (cropped_image := self.crop_image(source_image, job, face_worker)) is not None:
                file_path, is_tiff = utils.set_filename(source_image, destination_path, job.radio_choice(), job.radio_tuple())
                utils.save_image(cropped_image, file_path.as_posix(), job.gamma, is_tiff=is_tiff)
            else:
                utils.reject(source_image, destination_path, source_image)
        elif image_name is not None and new is None:
            if (cropped_image := self.crop_image(source_image, job, face_worker)) is not None:
                file_path, is_tiff = utils.set_filename(image_name, destination_path, job.radio_choice(), job.radio_tuple())
                utils.save_image(cropped_image, file_path.as_posix(), job.gamma, is_tiff=is_tiff)
            else:
                utils.reject(source_image, destination_path, image_name)
        elif image_name is not None:
            if (cropped_image := self.crop_image(source_image, job, face_worker)) is not None:
                file_path, is_tiff = utils.set_filename( image_name, destination_path, job.radio_choice(), job.radio_tuple(), new)
                utils.save_image(cropped_image, file_path.as_posix(), job.gamma, is_tiff=is_tiff)
            else:
                utils.reject(source_image, destination_path, image_name)

    @staticmethod
    def multi_save_loop(cropped_images: List[cvt.MatLike],
                        file_path: Path,
                        gamma: int,
                        is_tiff: bool) -> None:
        for i, image in enumerate(cropped_images):
            new_file_path = file_path.with_stem(f'{file_path.stem}_{i}')
            utils.save_image(image, new_file_path.as_posix(), gamma, is_tiff=is_tiff)

    def multi_save_detection(self, source_image: Path,
                             job: Job,
                             face_worker: FaceWorker,
                             image_name: Optional[Path] = None,
                             new: Optional[str] = None) -> None:
        if (destination_path := job.get_destination()) is None: return None
        if image_name is None and new is None:
            if (cropped_images := self.multi_crop(source_image, job, face_worker)) is None:
                utils.reject(source_image, destination_path, source_image)
            else:
                file_path, is_tiff = utils.set_filename(source_image, destination_path, job.radio_choice(), job.radio_tuple())
                self.multi_save_loop(cropped_images, file_path, job.gamma, is_tiff)
        elif image_name is not None and new is None:
            if (cropped_images := self.multi_crop(source_image, job, face_worker)) is None:
                utils.reject(source_image, destination_path, image_name)
            else:
                file_path, is_tiff = utils.set_filename(image_name, destination_path, job.radio_choice(), job.radio_tuple())
                self.multi_save_loop(cropped_images, file_path, job.gamma, is_tiff)
        elif image_name is not None:
            if (cropped_images := self.multi_crop(source_image, job, face_worker)) is None:
                utils.reject(source_image, destination_path, image_name)
            else:
                file_path, is_tiff = utils.set_filename(
                    image_name, destination_path, job.radio_choice(), job.radio_tuple(), new)
                self.multi_save_loop(cropped_images, file_path, job.gamma, is_tiff)

    @staticmethod
    def multi_crop(source_image: Union[cvt.MatLike, Path],
                   job: Job,
                   face_worker: FaceWorker) -> Optional[List[cvt.MatLike]]:
        img = utils.open_pic(source_image, job.fix_exposure_job.isChecked(), job.auto_tilt_job.isChecked(), face_worker) \
                        if isinstance(source_image, Path) else source_image
        if img is None:
            return None
        detections, crop_positions = utils.multi_box_positions(img, job, face_worker)
        # Check if any faces were detected
        if not np.any(100 * detections[0, 0, :, 2] > (100 - job.sensitivity)): return None
        images = [Image.fromarray(img).crop(crop_position) for crop_position in crop_positions]

        image_array = [utils.pillow_to_numpy(image) for image in images]

        results = [utils.convert_color_space(array) for array in image_array]
        output: List[cvt.MatLike] = [cv2.resize(src=result, dsize=job.size, interpolation=cv2.INTER_AREA)
                                     for result in results]
        return output

    def crop(self, image: Path,
             job: Job,
             face_worker: FaceWorker,
             new: Optional[str] = None) -> None:
        if job.table is not None and job.folder_path is not None and new is not None:
            # Data cropping
            path = job.folder_path.joinpath(image)
            if job.multi_face_job.isChecked():
                self.multi_save_detection(path, job, face_worker, image, new)
            else:
                self.save_detection(path, job, face_worker, image, new)
        elif job.folder_path is not None:
            # Folder cropping
            source, image_name = job.folder_path, image.name
            path = source.joinpath(image_name)
            if job.multi_face_job.isChecked():
                self.multi_save_detection(path, job, face_worker, Path(image_name))
            else:
                self.save_detection(path, job, face_worker, Path(image_name))
        elif job.multi_face_job.isChecked():
            self.multi_save_detection(image, job, face_worker)
        else:
            self.save_detection(image, job, face_worker)

    def display_crop(self, job: Job,
                     line_edit: Union[Path, QLineEdit],
                     image_widget: ImageWidget) -> None:
        img_path = line_edit if isinstance(line_edit, Path) else Path(line_edit.text())
        # if input field is empty, then do nothing
        if not img_path or img_path.as_posix() in {'', '.', None}: return None

        if img_path.is_dir():
            first_file = utils.get_first_file(img_path)
            if first_file is None: return None
            img_path = first_file

        pic_array = utils.open_pic(
            img_path, job.fix_exposure_job.isChecked(), job.auto_tilt_job.isChecked(), self.face_workers[0]
        )
        if pic_array is None: return None

        pic_array = utils.adjust_gamma(pic_array, job.gamma)
        if job.multi_face_job.isChecked():
            adjusted = utils.convert_color_space(pic_array)
            pic = utils.multi_box(adjusted, job, self.face_workers[0])
            utils.display_image_on_widget(pic, image_widget)
        else:
            bounding_box = utils.box_detect(pic_array, job, self.face_workers[0])
            if bounding_box is None: return None
            self.crop_and_set(pic_array, bounding_box, job.gamma, image_widget)

    @staticmethod
    def _numpy_array_crop(image: cvt.MatLike,
                          bounding_box: Tuple[int, int, int, int]) -> npt.NDArray[np.uint8]:
        # Load and crop image using PIL
        picture = Image.fromarray(image).crop(bounding_box)
        return utils.pillow_to_numpy(picture)

    def crop_and_set(self, image: cvt.MatLike,
                     bounding_box: Tuple[int, int, int, int],
                     gamma: int,
                     image_widget: ImageWidget) -> None:
        """
        Crop the given image using the bounding box, adjust its exposure and gamma, and set it to an image widget.

        Parameters:
            image: The input image as a numpy array.
            bounding_box: The bounding box coordinates to crop the image.
            gamma: The gamma value for gamma correction.
            image_widget: The image widget to display the processed image.
        
        Returns: None
        """
        try:
            cropped_image = self._numpy_array_crop(image, bounding_box)
            adjusted_image = utils.adjust_gamma(cropped_image, gamma)
            final_image = utils.convert_color_space(adjusted_image)
        except (cv2.error, Image.DecompressionBombError):
            return None
        utils.display_image_on_widget(final_image, image_widget)

    def crop_image(self, image: Union[Path, cvt.MatLike],
                   job: Job,
                   face_worker: FaceWorker) -> Optional[cvt.MatLike]:
        pic_array = utils.open_pic(image, job.fix_exposure_job.isChecked(), job.auto_tilt_job.isChecked(), face_worker) \
            if isinstance(image, Path) else image
        if pic_array is None: return None
        if (bounding_box := utils.box_detect(pic_array, job, face_worker)) is None: return None
        cropped_pic = self._numpy_array_crop(pic_array, bounding_box)
        result = utils.convert_color_space(cropped_pic) if len(cropped_pic.shape) >= 3 else cropped_pic
        return cv2.resize(result, job.size, interpolation=cv2.INTER_AREA)

    def _update_progress(self, file_amount: int,
                         process_type: FunctionType) -> None:
        match process_type:
            case FunctionType.FOLDER:
                self.bar_value_f += 1
                if self.bar_value_f == file_amount:
                    self.folder_progress.emit((file_amount, file_amount))
                    self.folder_finished.emit()
                elif self.bar_value_f < file_amount:
                    self.folder_progress.emit((self.bar_value_f, file_amount))
            case FunctionType.MAPPING:
                self.bar_value_m += 1
                if self.bar_value_m == file_amount:
                    self.mapping_progress.emit((file_amount, file_amount))
                    self.mapping_finished.emit()
                elif self.bar_value_m < file_amount:
                    self.mapping_progress.emit((self.bar_value_m, file_amount))
            case FunctionType.VIDEO:
                self.bar_value_v += 1
                if self.bar_value_v == file_amount:
                    self.video_progress.emit((file_amount, file_amount))
                    self.video_finished.emit()
                elif self.bar_value_v < file_amount:
                    self.video_progress.emit((self.bar_value_v, file_amount))
            case _: pass
    
    def folder_worker(self, file_amount: int,
                      file_list: npt.NDArray[Any],
                      job: Job,
                      face_worker: FaceWorker) -> None:
        for image in file_list:
            if self.end_f_task:
                break
            self.crop(image, job, face_worker)
            self._update_progress(file_amount, FunctionType.FOLDER)

        if self.bar_value_f == file_amount or self.end_f_task:
            self.message_box_f = False

    def crop_dir(self, job: Job) -> None:
        if (file_tuple := job.file_list()) is None: return None
        file_list, amount = file_tuple
        split_array = np.array_split(file_list, min(cpu_count(), 8))
        self.bar_value_f = 0
        self.folder_progress.emit((self.bar_value_f, amount))
        self.folder_started.emit()
        
        threads: List[Thread] = []
        for i, array in enumerate(split_array):
            t = Thread(target=self.folder_worker, args=(amount, array, job, self.face_workers[i]))
            threads.append(t)
            t.start()

    def mapping_worker(self, file_amount: int,
                       old: npt.NDArray[np.str_],
                       new: npt.NDArray[np.str_],
                       job: Job,
                       face_worker: FaceWorker) -> None:
        for i, image in enumerate(old):
            if self.end_m_task:
                break
            self.crop(Path(image), job, face_worker, new=new[i])
            self._update_progress(file_amount, FunctionType.MAPPING)

        if self.bar_value_m == file_amount or self.end_m_task:
            self.message_box_m = False

    def mapping_crop(self, job: Job) -> None:
        if (g := job.file_list_to_numpy()) is None: return None
        file_list1, file_list2 = g
        # Get the extensions of the file names and 
        # Create a mask that indicates which files have supported extensions.
        mask, amount = utils.mask_extensions(file_list1)
        # Split the file lists and the mapping data into chunks.
        old_file_list, new_file_list = utils.split_by_cpus(mask, min(cpu_count(), 8), file_list1, file_list2)
        
        self.bar_value_m = 0
        self.mapping_progress.emit((self.bar_value_m, amount))
        self.mapping_started.emit()
        
        threads: List[Thread] = []
        for i, _ in enumerate(old_file_list):
            t = Thread(target=self.mapping_worker, args=(amount, _, new_file_list[i], job, self.face_workers[i]))
            threads.append(t)
            t.start()

    @cache
    def grab_frame(self, position_slider: int,
                   video_line_edit: str) -> Optional[cvt.MatLike]:
        # Set video frame position to timelineSlider value
        self.cap = cv2.VideoCapture(video_line_edit)
        self.cap.set(cv2.CAP_PROP_POS_MSEC, position_slider)
        # Read frame from video capture object
        ret, frame = self.cap.read()
        if not ret: return None
        self.cap.release()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def crop_frame(self, job: Job,
                   position_label: QLabel,
                   timeline_slider: QSlider) -> None:
        if job.video_path is None or job.destination is None:
            return None
        if (frame := self.grab_frame(timeline_slider.value(), job.video_path.as_posix())) is None:
            return None

        destination, base_name = job.destination, job.video_path.stem
        destination.mkdir(exist_ok=True)
        position = re.sub(':', '_', position_label.text())
        file_path = destination.joinpath(
            f'{base_name} - ({position}){job.radio_options[2] if job.radio_choice() == job.radio_options[0] else job.radio_choice()}')
        is_tiff = file_path.suffix in {'.tif', '.tiff'}

        if job.multi_face_job.isChecked():
            if (images := self.multi_crop(frame, job, self.face_workers[0])) is None:
                return None

            for i, image in enumerate(images):
                new_file_path = file_path.with_stem(f'{file_path.stem}_{i}')
                self.save_frame(image, new_file_path, job, is_tiff)
        elif (cropped_image := self.crop_image(frame, job, self.face_workers[0])) is None:
            return None

        else:
            self.save_frame(cropped_image, file_path, job, is_tiff)

    @staticmethod
    def get_frame_path(destination: Path,
                       file_enum: str,
                       job: Job) -> Tuple[Path, bool]:
        file_str = f'{file_enum}.jpg' if job.radio_choice() == job.radio_options[0] else file_enum + job.radio_choice()
        file_path = destination.joinpath(file_str)
        return file_path, file_path.suffix in {'.tif', '.tiff'}

    @staticmethod
    def save_frame(image: cvt.MatLike,
                   file_path: Path,
                   job: Job,
                   is_tiff: bool) -> None:
        utils.save_image(image, file_path.as_posix(), job.gamma, is_tiff=is_tiff)

    def process_multiface_frame_job(self, frame: cvt.MatLike,
                                    job: Job,
                                    file_enum: str,
                                    destination: Path) -> None:
        if (images := self.multi_crop(frame, job, self.face_workers[0])) is None:
            file_enum = f'failed_{file_enum}'
            file_path, is_tiff = self.get_frame_path(destination, file_enum, job)
            self.save_frame(utils.convert_color_space(frame), file_path, job, is_tiff)
        else:
            for i, image in enumerate(images):
                file_path, is_tiff = self.get_frame_path(destination, f'{file_enum}_{i}', job)
                self.save_frame(utils.convert_color_space(image), file_path, job, is_tiff)

    def process_singleface_frame_job(self, frame: cvt.MatLike,
                                     job: Job,
                                     file_enum: str,
                                     destination: Path) -> None:
        if (cropped_image := self.crop_image(frame, job, self.face_workers[0])) is None:
            cropped_image = frame
            file_enum = f'failed_{file_enum}'           
        else:
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

        file_path, is_tiff = self.get_frame_path(destination, file_enum, job)
        self.save_frame(cropped_image, file_path, job, is_tiff)

    def frame_extraction(self, video: cv2.VideoCapture,
                         frame_number: int,
                         job: Job,
                         progress_callback: Callable[..., Any]) -> None:
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = video.read()
        if not ret: return None
        
        try:
            assert job.video_path is not None
        except AssertionError:
            return None
        
        if (destination := job.get_destination()) is None: return None
        file_enum = f'{job.video_path.stem}_frame_{frame_number:06d}'
        if job.multi_face_job.isChecked():
            self.process_multiface_frame_job(frame, job, file_enum, destination)
        else:
            self.process_singleface_frame_job(frame, job, file_enum, destination)
        progress_callback()
    
    def extract_frames(self, job: Job) -> None:
        if job.video_path is None or job.start_position is None or job.stop_position is None:
            return None
        
        video = cv2.VideoCapture(job.video_path.as_posix())
        fps = int(video.get(cv2.CAP_PROP_FPS))
        start_frame = int(job.start_position * fps)
        end_frame = int(job.stop_position * fps)
        frame_numbers = np.arange(start_frame, end_frame + 1)

        self.video_progress.emit((0, frame_numbers.size))
        self.video_started.emit()

        def progress_callback() -> None:
            self._update_progress(frame_numbers.size, FunctionType.VIDEO)
        
        for frame_number in frame_numbers:
            if self.end_v_task:
                break
            self.frame_extraction(video, frame_number, job, progress_callback)
            if self.bar_value_v == frame_numbers.size or self.end_v_task:
                self.message_box_v = False
        video.release()

    def terminate(self, series: FunctionType) -> None:
        match series:
            case FunctionType.FOLDER:
                self.end_f_task = True
                self.folder_finished.emit()
            case FunctionType.MAPPING:
                self.end_m_task = True
                self.mapping_finished.emit()
            case FunctionType.VIDEO:
                self.end_v_task = True
                self.video_finished.emit()
            case _: return None
