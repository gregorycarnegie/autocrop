import re
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Callable, ClassVar, List, Optional, Tuple, Union

import cv2
import cv2.typing as cvt
import dlib
import numpy as np
import numpy.typing as npt
from PyQt6.QtCore import pyqtSignal, QObject
from PyQt6.QtWidgets import QLabel, QLineEdit, QSlider

from . import utils as ut
from . import window_functions as wf
from .enums import FunctionType, ResourcePath
from .image_widget import ImageWidget
from .job import Job


class Cropper(QObject):
    """
    A class that represents a Cropper that inherits from the QObject class.

    Attributes:
        THREAD_NUMBER: ClassVar[int] = min(cpu_count(), 8)
        TASK_VALUES: ClassVar[Tuple[int, bool, bool]] = 0, False, True

    Methods:
        generate_face_detection_tools() -> List[Tuple[dlib.fhog_object_detector, dlib.shape_predictor]]:
            Starts the face detection tools by creating a list.

        reset_task(function_type: FunctionType):
            Resets the task values based on the provided function type.

        photo_crop(image: Path, job: Job, face_detection_tools: Tuple[Any, Any], new: Optional[str] = None) -> None:
            Crops the photo image based on the provided job parameters.

        display_crop(self, job: Job, line_edit: Union[Path, QLineEdit], image_widget: ImageWidget) -> None:
            Displays the cropped image on the image widget based on the provided job parameters.

        _update_progress(self, file_amount: int, process_type: FunctionType) -> None:
            Updates the progress bar value based on the process type.

        folder_worker(self, file_amount: int, file_list: npt.NDArray[Any], job: Job, face_detection_tools: Tuple[Any, Any]) -> None:
            Performs cropping for a folder job by iterating over the file list, cropping each image, and updating the progress.

        crop_dir(self, job: Job) -> None:
            Crops all files in a directory by splitting the file list into chunks and running folder workers in separate threads.

        mapping_worker(self, file_amount: int, job: Job, face_detection_tools: Tuple[Any, Any], old: npt.NDArray[np.str_], new: npt.NDArray[np.str_]) -> None:
            Performs cropping for a mapping job by iterating over the old file list, cropping each image, and updating the progress.

        mapping_crop(self, job: Job) -> None:
            Performs cropping for a mapping job by splitting the file lists and mapping data into chunks and running mapping workers in separate threads.

        crop_frame(self, job: Job, position_label: QLabel, timeline_slider: QSlider) -> None:
            Crops and saves a frame based on the specified job parameters.

        _process_multiface_frame_job(self, frame: cvt.MatLike, job: Job, file_enum: str, destination: Path) -> None:
            Processes a frame for a multi-face job by cropping and saving the individual faces.

        _process_singleface_frame_job(self, frame: cvt.MatLike, job: Job, file_enum: str, destination: Path) -> None:
            Processes a single-face frame job by cropping the frame and saving the cropped image.

        frame_extraction(self, video: cv2.VideoCapture, frame_number: int, job: Job, progress_callback: Callable[..., Any]) -> None:
            Performs frame extraction from a video based on the specified frame number and job parameters.

        extract_frames(self, job: Job) -> None:
            Extracts frames from a video based on the specified job parameters.

        terminate(self, series: FunctionType) -> None:
            Terminates the specified series of tasks.
    """

    THREAD_NUMBER: ClassVar[int] = min(cpu_count(), 8)
    TASK_VALUES: ClassVar[Tuple[int, bool, bool]] = 0, False, True
    LANDMARKS: ClassVar[str] = ResourcePath('resources\\models\\shape_predictor_68_face_landmarks.dat').meipass_path

    f_started, f_finished, f_progress = pyqtSignal(), pyqtSignal(), pyqtSignal(object)
    m_started, m_finished, m_progress = pyqtSignal(), pyqtSignal(), pyqtSignal(object)
    v_started, v_finished, v_progress = pyqtSignal(), pyqtSignal(), pyqtSignal(object)

    def __init__(self, parent: Optional[QObject] = None):
        super(Cropper, self).__init__(parent)
        self.bar_value_f, self.end_f_task, self.message_box_f = self.TASK_VALUES
        self.bar_value_m, self.end_m_task, self.message_box_m = self.TASK_VALUES
        self.bar_value_v, self.end_v_task, self.message_box_v = self.TASK_VALUES
        self.face_detection_tools: List[Tuple[Any, Any]] = []
        self.generate_face_detection_tools()

    def reset_task(self, function_type: FunctionType):
        """
        Resets the task values based on the provided function type.

        Args:
            self: The Cropper instance.
            function_type (FunctionType): The type of function to reset.

        Returns:
            None
        """

        match function_type:
            case FunctionType.FOLDER:
                self.bar_value_f, self.end_f_task, self.message_box_f = self.TASK_VALUES
            case FunctionType.MAPPING:
                self.bar_value_m, self.end_m_task, self.message_box_m = self.TASK_VALUES
            case FunctionType.VIDEO:
                self.bar_value_v, self.end_v_task, self.message_box_v = self.TASK_VALUES
            case _:
                pass

    @classmethod
    def _create_tool_pair(cls) -> Tuple[Any, Any]:
        """
        Create a pair of face detection and shape prediction tools.

        Returns:
            Tuple[dlib.fhog_object_detector, dlib.shape_predictor]:
                A pair of face detection and shape prediction tools.
        """

        # Functions to create each tool
        def create_detector() -> Any:
            return dlib.get_frontal_face_detector()

        def create_predictor() -> Any:
            return dlib.shape_predictor(cls.LANDMARKS)

        # Execute each function in a separate thread
        with ThreadPoolExecutor(max_workers=2) as executor:
            d_future, p_future = executor.submit(create_detector), executor.submit(create_predictor)
            # Now, wait for both tasks to complete and get the results
            detector, predictor = d_future.result(), p_future.result()

        return detector, predictor

    def generate_face_detection_tools(self) -> None:
        """
        Generate a list of face detection and shape prediction tools.
    
        This method creates a list of tuples, where each tuple contains an instance of
        `dlib.fhog_object_detector` for frontal face detection and `dlib.shape_predictor`
        for facial landmark prediction. The number of tuples in the list is determined by
        the class attribute `THREAD_NUMBER`.
    
        Returns:
            List[Tuple[dlib.fhog_object_detector, dlib.shape_predictor]]:
                A list of face detection and shape prediction tools.
        """

        for _ in range(self.THREAD_NUMBER):
            self.face_detection_tools.append(self._create_tool_pair())

    def photo_crop(self, image: Path,
                   job: Job,
                   new: Optional[str] = None) -> None:
        """
        Crops the photo image based on the provided job parameters.

        Args:
            image (Path): The path to the image file.
            job (Job): The job containing the parameters for cropping.
            new (Optional[str]): The optional new file name.

        Returns:
            None
        """

        if image.is_file():
            ut.crop(image, job, self.face_detection_tools[0], new)

    def display_crop(self, job: Job,
                     line_edit: Union[Path, QLineEdit],
                     image_widget: ImageWidget) -> None:
        """
        Displays the cropped image on the image widget based on the provided job parameters.

        Args:
            self: The Cropper instance.
            job (Job): The job containing the parameters for cropping.
            line_edit (Union[Path, QLineEdit]): The input field for the image path.
            image_widget (ImageWidget): The widget to display the cropped image.

        Returns:
            None
        """

        img_path = line_edit if isinstance(line_edit, Path) else Path(line_edit.text())
        # if input field is empty, then do nothing
        if not img_path or img_path.as_posix() in {'', '.', None}:
            return

        if img_path.is_dir():
            first_file = ut.get_first_file(img_path)
            if first_file is None:
                return
            img_path = first_file

        if not img_path.is_file():
            return

        pic_array = ut.open_pic(img_path, self.face_detection_tools[0],
                                exposure=job.fix_exposure_job,
                                tilt=job.auto_tilt_job)
        if pic_array is None:
            return

        if job.multi_face_job:
            pic = ut.multi_box(pic_array, job)
            wf.display_image_on_widget(pic, image_widget)
        else:
            bounding_box = ut.box_detect(pic_array, job)
            if bounding_box is None:
                return
            ut.crop_and_set(pic_array, bounding_box, job.gamma, image_widget)

    def _update_progress(self, file_amount: int,
                         process_type: FunctionType) -> None:
        """
        Updates the progress bar value based on the process type.

        Args:
            self: The Cropper instance.
            file_amount (int): The total number of files to process.
            process_type (FunctionType): The type of process being updated.

        Returns:
            None
        """

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
            case _:
                pass

    def folder_worker(self, file_amount: int,
                      file_list: npt.NDArray[Any],
                      job: Job,
                      face_detection_tools: Tuple[Any, Any]) -> None:
        """
        Performs cropping for a folder job by iterating over the file list, cropping each image, and updating the progress.

        Args:
            self: The Cropper instance.
            file_amount (int): The total number of files to process.
            file_list (npt.NDArray[Any]): The array of file paths.
            job (Job): The job containing the parameters for cropping.
            face_detection_tools(Tuple[Any, Any]): The worker for face-related tasks.

        Returns:
            None
        """

        for image in file_list:
            if self.end_f_task:
                break
            ut.crop(image, job, face_detection_tools)
            self._update_progress(file_amount, FunctionType.FOLDER)

        if self.bar_value_f == file_amount or self.end_f_task:
            self.message_box_f = False

    def crop_dir(self, job: Job) -> None:
        """
        Crops all files in a directory by splitting the file list into chunks and running folder workers in separate threads.

        Args:
            self: The Cropper instance.
            job (Job): The job containing the file list.

        Returns:
            None
        """

        if (file_tuple := job.file_list()) is None:
            return
        file_list, amount = file_tuple
        # Split the file list into chunks.
        split_array = np.array_split(file_list, self.THREAD_NUMBER)

        self.bar_value_f = 0
        self.f_progress.emit((self.bar_value_f, amount))
        self.f_started.emit()

        executor = ThreadPoolExecutor(max_workers=self.THREAD_NUMBER)
        _ = [executor.submit(self.folder_worker, amount, split_array[i], job, self.face_detection_tools[i])
             for i in range(len(split_array))]

    def mapping_worker(self, file_amount: int,
                       job: Job,
                       face_detection_tools: Tuple[Any, Any], *,
                       old: npt.NDArray[np.str_],
                       new: npt.NDArray[np.str_]) -> None:
        """
        Performs cropping for a mapping job by iterating over the old file list, cropping each image, and updating the progress.

        Args:
            self: The Cropper instance.
            file_amount (int): The total number of files to process.
            job (Job): The job containing the parameters for cropping.
            face_detection_tools(Tuple[Any, Any]): The worker for face-related tasks.
            old (npt.NDArray[np.str_]): The array of old file paths.
            new (npt.NDArray[np.str_]): The array of new file paths.

        Returns:
            None
        """

        for i, image in enumerate(old):
            if self.end_m_task:
                break
            x = Path(image)
            if x.is_file():
                ut.crop(Path(image), job, face_detection_tools, new=new[i])
            self._update_progress(file_amount, FunctionType.MAPPING)

        if self.bar_value_m == file_amount or self.end_m_task:
            self.message_box_m = False

    def mapping_crop(self, job: Job) -> None:
        """
        Performs cropping for a mapping job by splitting the file lists and mapping data into chunks and running mapping workers in separate threads.

        Args:
            self: The Cropper instance.
            job (Job): The job containing the file lists and mapping data.

        Returns:
            None
        """

        if (file_tuple := job.file_list_to_numpy()) is None:
            return
        file_list1, file_list2 = file_tuple
        # Get the extensions of the file names and
        # Create a mask that indicates which files have supported extensions.
        mask, amount = ut.mask_extensions(file_list1)
        # Split the file lists and the mapping data into chunks.
        old_file_list, new_file_list = ut.split_by_cpus(mask, self.THREAD_NUMBER, file_list1, file_list2)

        self.bar_value_m = 0
        self.m_progress.emit((self.bar_value_m, amount))
        self.m_started.emit()

        executor = ThreadPoolExecutor(max_workers=self.THREAD_NUMBER)
        _ = [executor.submit(self.mapping_worker, amount, job, self.face_detection_tools[i],
                             old=old_file_list[i], new=new_file_list[i])
             for i in range(len(new_file_list))]

    def crop_frame(self, job: Job, position_label: QLabel, timeline_slider: QSlider) -> None:
        """
        Crops and saves a frame based on the specified job parameters.

        Args:
            self: The Cropper instance.
            job (Job): The job containing the parameters for cropping.
            position_label (QLabel): The label displaying the current position.
            timeline_slider (QSlider): The slider representing the timeline position.

        Returns:
            None
        """

        # Early exits for conditions where no processing is required
        if not job.video_path or not job.destination:
            return

        if (frame := ut.grab_frame(timeline_slider.value(), job.video_path.as_posix())) is None:
            return

        destination = job.destination
        # base_name = job.video_path.stem
        destination.mkdir(exist_ok=True)

        # Swap ':' to '_' in position text
        position = re.sub(':', '_', position_label.text())

        # Determine file suffix based on radio choice
        file_suffix = job.radio_options[2] if job.radio_choice() == job.radio_options[0] else job.radio_choice()

        file_path = destination.joinpath(f'{job.video_path.stem} - ({position}){file_suffix}')
        is_tiff = file_path.suffix in {'.tif', '.tiff'}

        # Handle multi-face job
        if job.multi_face_job:
            if (images := ut.multi_crop(frame, job, self.face_detection_tools[0])) is None:
                return

            for i, image in enumerate(images):
                new_file_path = file_path.with_stem(f'{file_path.stem}_{i}')
                ut.save_image(image, new_file_path, job.gamma, is_tiff)
            return

        if cropped_image := ut.crop_image(frame, job, self.face_detection_tools[0]):
            ut.save_image(cropped_image, file_path, job.gamma, is_tiff)

    def process_multiface_frame_job(self, frame: cvt.MatLike,
                                    job: Job,
                                    file_enum: str,
                                    destination: Path) -> None:
        """
        Processes a frame for a multi-face job by cropping and saving the individual faces.

        Args:
            self: The Cropper instance.
            frame (cvt.MatLike): The frame to process.
            job (Job): The job containing the parameters for cropping.
            file_enum (str): The file enumeration for the frame.
            destination (Path): The destination path to save the cropped faces.

        Returns:
            None
        """

        if (images := ut.multi_crop(frame, job, self.face_detection_tools[0])) is None:
            file_path, is_tiff = ut.get_frame_path(destination, f'failed_{file_enum}', job)
            ut.save_image(ut.convert_color_space(frame), file_path, job.gamma, is_tiff)
        else:
            for i, image in enumerate(images):
                file_path, is_tiff = ut.get_frame_path(destination, f'{file_enum}_{i}', job)
                ut.save_image(ut.convert_color_space(image), file_path, job.gamma, is_tiff)

    def process_singleface_frame_job(self, frame: cvt.MatLike,
                                     job: Job,
                                     file_enum: str,
                                     destination: Path) -> None:
        """
        Processes a single-face frame job by cropping the frame and saving the cropped image.

        Args:
            self: The Cropper instance.
            frame (cvt.MatLike): The frame to process.
            job (Job): The job containing the parameters for cropping.
            file_enum (str): The file enumeration string.
            destination (Path): The destination path to save the file.

        Returns:
            None
        """

        if (cropped_image := ut.crop_image(frame, job, self.face_detection_tools[0])) is None:
            ut.frame_save(frame, file_enum, destination, job)
        else:
            ut.frame_save(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB), file_enum, destination, job)

    def frame_extraction(self, video: cv2.VideoCapture,
                         frame_number: int,
                         job: Job,
                         progress_callback: Callable[..., Any]) -> None:
        """
        Performs frame extraction from a video based on the specified frame number and job parameters.

        Args:
            self: The Cropper instance.
            video (cv2.VideoCapture): The video capture object.
            frame_number (int): The frame number to extract.
            job (Job): The job containing the video path and other parameters.
            progress_callback (Callable[..., Any]): A callback function to update the progress.

        Returns:
            None
        """

        # Check if video_path is set in the job
        if job.video_path is None:
            return

        # Set the video capture object to the specified frame number and read the frame
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = video.read()

        # Return early if frame read is unsuccessful
        if not ret:
            return

        # Get the destination, return early if not set
        if (destination := job.get_destination()) is None:
            return

        # Determine the file name enumeration based on frame number
        file_enum = f'{job.video_path.stem}_frame_{frame_number:06d}'

        # Process the frame based on whether it's a multi-face job or single-face job
        if job.multi_face_job:
            self.process_multiface_frame_job(frame, job, file_enum, destination)
        else:
            self.process_singleface_frame_job(frame, job, file_enum, destination)

        # Update progress
        progress_callback()

    def extract_frames(self, job: Job) -> None:
        """
        Extracts frames from a video based on the specified job parameters.

        Args:
            self: The Cropper instance.
            job (Job): The job containing the video path, start position, and stop position.

        Returns:
            None
        """

        if job.video_path is None or job.start_position is None or job.stop_position is None:
            return

        video = cv2.VideoCapture(job.video_path.as_posix())
        fps = video.get(cv2.CAP_PROP_FPS)
        start_frame, end_frame = int(job.start_position * fps), int(job.stop_position * fps)
        # frame_numbers = np.arange(start_frame, end_frame + 1)
        # x = frame_numbers.size
        # frame_numbers = range(start_frame, end_frame + 1)
        # x = len(frame_numbers)
        x = 1 + end_frame - start_frame

        self.v_progress.emit((0, x))
        self.v_started.emit()

        for frame_number in range(start_frame, end_frame + 1):
            if self.end_v_task:
                break
            self.frame_extraction(video, frame_number, job,
                                  lambda: self._update_progress(x, FunctionType.VIDEO))
            if self.bar_value_v == x or self.end_v_task:
                self.message_box_v = False
        video.release()

    def terminate(self, series: FunctionType) -> None:
        """
        Terminates the specified series of tasks.

        Args:
            self: The Cropper instance.
            series (FunctionType): The type of series to terminate.

        Returns:
            None
        """

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
            case _:
                return
