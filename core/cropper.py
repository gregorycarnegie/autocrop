import re
from functools import cache
from multiprocessing import cpu_count
from pathlib import Path
from threading import Thread
from typing import Any, Callable, ClassVar, Generator, List, Optional, Tuple, Union

import cv2
import cv2.typing as cvt
import numpy as np
import numpy.typing as npt
from PyQt6.QtCore import pyqtSignal, QObject
from PyQt6.QtWidgets import QLabel, QLineEdit, QSlider

from . import utils as ut
from . import window_functions as wf
from .enums import FunctionType
from .face_worker import FaceWorker
from .image_widget import ImageWidget
from .job import Job


class Cropper(QObject):
    THREAD_NUMBER: ClassVar[int] = min(cpu_count(), 8)
    TASK_VALUES: ClassVar[Tuple[int, bool, bool]] = 0, False, True

    f_started, f_finished, f_progress = pyqtSignal(), pyqtSignal(), pyqtSignal(object)
    m_started, m_finished, m_progress = pyqtSignal(), pyqtSignal(), pyqtSignal(object)
    v_started, v_finished, v_progress = pyqtSignal(), pyqtSignal(), pyqtSignal(object)

    def __init__(self, parent: Optional[QObject]=None):
        super(Cropper, self).__init__(parent)
        self.bar_value_f, self.end_f_task, self.message_box_f = self.TASK_VALUES
        self.bar_value_m, self.end_m_task, self.message_box_m = self.TASK_VALUES
        self.bar_value_v, self.end_v_task, self.message_box_v = self.TASK_VALUES
        self.face_workers = self.start_face_workers()

    @classmethod
    def start_face_workers(cls) -> List[FaceWorker]:
        """Load the face detectors and shape predictors"""
        return [FaceWorker() for _ in range(cls.THREAD_NUMBER)]

    def reset_task(self, function_type: FunctionType):
        match function_type:
            case FunctionType.FOLDER:
                self.bar_value_f, self.end_f_task, self.message_box_f = self.TASK_VALUES
            case FunctionType.MAPPING:
                self.bar_value_m, self.end_m_task, self.message_box_m = self.TASK_VALUES
            case FunctionType.VIDEO:
                self.bar_value_v, self.end_v_task, self.message_box_v = self.TASK_VALUES
            case _: pass
    
    @staticmethod
    def photo_crop(image: Path,
                   job: Job,
                   face_worker: FaceWorker,
                   new: Optional[str] = None) -> None:
        ut.crop(image, job, face_worker, new)

    def display_crop(self, job: Job,
                     line_edit: Union[Path, QLineEdit],
                     image_widget: ImageWidget) -> None:
        img_path = line_edit if isinstance(line_edit, Path) else Path(line_edit.text())
        # if input field is empty, then do nothing
        if not img_path or img_path.as_posix() in {'', '.', None}: return None

        if img_path.is_dir():
            first_file = ut.get_first_file(img_path)
            if first_file is None: return None
            img_path = first_file

        pic_array = ut.open_pic(
            img_path, job.fix_exposure_job.isChecked(), job.auto_tilt_job.isChecked(), self.face_workers[0]
        )
        if pic_array is None: return None

        pic_array = ut.adjust_gamma(pic_array, job.gamma)
        if job.multi_face_job.isChecked():
            adjusted = ut.convert_color_space(pic_array)
            pic = ut.multi_box(adjusted, job, self.face_workers[0])
            wf.display_image_on_widget(pic, image_widget)
        else:
            bounding_box = ut.box_detect(pic_array, job, self.face_workers[0])
            if bounding_box is None: return None
            ut.crop_and_set(pic_array, bounding_box, job.gamma, image_widget)

    def _update_progress(self, file_amount: int,
                         process_type: FunctionType) -> None:
        match process_type:
            case FunctionType.FOLDER:
                self.bar_value_f += 1
                if self.bar_value_f == file_amount:
                    self.f_progress.emit((file_amount, file_amount))
                    self.f_finished.emit()
                elif self.bar_value_f < file_amount:
                    self.f_progress.emit((self.bar_value_f, file_amount))
            case FunctionType.MAPPING:
                self.bar_value_m += 1
                if self.bar_value_m == file_amount:
                    self.m_progress.emit((file_amount, file_amount))
                    self.m_finished.emit()
                elif self.bar_value_m < file_amount:
                    self.m_progress.emit((self.bar_value_m, file_amount))
            case FunctionType.VIDEO:
                self.bar_value_v += 1
                if self.bar_value_v == file_amount:
                    self.v_progress.emit((file_amount, file_amount))
                    self.v_finished.emit()
                elif self.bar_value_v < file_amount:
                    self.v_progress.emit((self.bar_value_v, file_amount))
            case _: pass
    
    def folder_worker(self, file_amount: int,
                      file_list: npt.NDArray[Any],
                      job: Job,
                      face_worker: FaceWorker) -> None:
        for image in file_list:
            if self.end_f_task:
                break
            ut.crop(image, job, face_worker)
            self._update_progress(file_amount, FunctionType.FOLDER)

        if self.bar_value_f == file_amount or self.end_f_task:
            self.message_box_f = False

    def crop_dir(self, job: Job) -> None:
        if (file_tuple := job.file_list()) is None: return None
        file_list, amount = file_tuple
        # Split the file list into chunks.
        split_array = np.array_split(file_list, self.THREAD_NUMBER)

        self.bar_value_f = 0
        self.f_progress.emit((self.bar_value_f, amount))
        self.f_started.emit()
        
        threads: Generator[Thread, None, None] = (
            Thread(target=self.folder_worker, args=(amount, split_array[i], job, self.face_workers[i]))
            for i in range(len(split_array)))
        for t in threads:
            t.start() 

    def mapping_worker(self, file_amount: int,
                       old: npt.NDArray[np.str_],
                       new: npt.NDArray[np.str_],
                       job: Job,
                       face_worker: FaceWorker) -> None:
        for i, image in enumerate(old):
            if self.end_m_task:
                break
            ut.crop(Path(image), job, face_worker, new=new[i])
            self._update_progress(file_amount, FunctionType.MAPPING)

        if self.bar_value_m == file_amount or self.end_m_task:
            self.message_box_m = False

    def mapping_crop(self, job: Job) -> None:
        if (file_tuple := job.file_list_to_numpy()) is None: return None
        file_list1, file_list2 = file_tuple
        # Get the extensions of the file names and 
        # Create a mask that indicates which files have supported extensions.
        mask, amount = ut.mask_extensions(file_list1)
        # Split the file lists and the mapping data into chunks.
        old_file_list, new_file_list = ut.split_by_cpus(mask, self.THREAD_NUMBER, file_list1, file_list2)
        
        self.bar_value_m = 0
        self.m_progress.emit((self.bar_value_m, amount))
        self.m_started.emit()
        
        threads: Generator[Thread, None, None] = (
            Thread(target=self.mapping_worker, args=(amount, old_file_list[i], new_file_list[i], job, self.face_workers[i]))
            for i in range(len(new_file_list)))
        for t in threads:
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
            if (images := ut.multi_crop(frame, job, self.face_workers[0])) is None:
                return None

            for i, image in enumerate(images):
                new_file_path = file_path.with_stem(f'{file_path.stem}_{i}')
                ut.save_image(image, new_file_path, job.gamma, is_tiff)
        elif (cropped_image := ut.crop_image(frame, job, self.face_workers[0])) is None:
            return None
        else:
            ut.save_image(cropped_image, file_path, job.gamma, is_tiff)

    def process_multiface_frame_job(self, frame: cvt.MatLike,
                                    job: Job,
                                    file_enum: str,
                                    destination: Path) -> None:
        if (images := ut.multi_crop(frame, job, self.face_workers[0])) is None:
            file_enum = f'failed_{file_enum}'
            file_path, is_tiff = ut.get_frame_path(destination, file_enum, job)
            ut.save_image(ut.convert_color_space(frame), file_path, job.gamma, is_tiff)
        else:
            for i, image in enumerate(images):
                file_path, is_tiff = ut.get_frame_path(destination, f'{file_enum}_{i}', job)
                ut.save_image(ut.convert_color_space(image), file_path, job.gamma, is_tiff)

    def process_singleface_frame_job(self, frame: cvt.MatLike,
                                     job: Job,
                                     file_enum: str,
                                     destination: Path) -> None:
        def callback(file_enum_str: str, cropped_image_data: cvt.MatLike) -> None:
            file_path, is_tiff = ut.get_frame_path(destination, file_enum_str, job)
            ut.save_image(cropped_image_data, file_path, job.gamma, is_tiff)

        if (cropped_image := ut.crop_image(frame, job, self.face_workers[0])) is None:
            callback(f'failed_{file_enum}', frame)
        else:
            callback(file_enum, cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

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
        fps = video.get(cv2.CAP_PROP_FPS)
        start_frame, end_frame = int(job.start_position * fps), int(job.stop_position * fps)
        frame_numbers = np.arange(start_frame, end_frame + 1)
        x = frame_numbers.size

        self.v_progress.emit((0, x))
        self.v_started.emit()

        for frame_number in frame_numbers:
            if self.end_v_task:
                break
            self.frame_extraction(video, frame_number, job,
                                  lambda: self._update_progress(x, FunctionType.VIDEO))
            if self.bar_value_v == x or self.end_v_task:
                self.message_box_v = False
        video.release()

    def terminate(self, series: FunctionType) -> None:
        match series:
            case FunctionType.FOLDER:
                self.end_f_task = True
                self.f_finished.emit()
            case FunctionType.MAPPING:
                self.end_m_task = True
                self.m_finished.emit()
            case FunctionType.VIDEO:
                self.end_v_task = True
                self.v_finished.emit()
            case _: return None
