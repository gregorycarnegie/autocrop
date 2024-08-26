from multiprocessing import cpu_count
from typing import ClassVar, Optional

from PyQt6.QtCore import pyqtSignal, QObject, pyqtBoundSignal

from .job import Job
from .resource_path import ResourcePath


class Cropper(QObject):
    """
    A class that represents a Cropper that inherits from the QObject class.

    Attributes:
        THREAD_NUMBER: ClassVar[int] = min(cpu_count(), 8)
        TASK_VALUES: ClassVar[Tuple[int, bool, bool]] = 0, False, True

    Methods:
        reset_task():
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
    TASK_VALUES: ClassVar[tuple[int, bool, bool]] = 0, False, True
    LANDMARKS: ClassVar[str] = ResourcePath('resources\\models\\shape_predictor_68_face_landmarks.dat').meipass_path

    started, finished, progress = pyqtSignal(), pyqtSignal(), pyqtSignal(object)

    def __init__(self, parent: Optional[QObject] = None):
        super(Cropper, self).__init__(parent)
        self.bar_value, self.end_task, self.message_box = self.TASK_VALUES

    def worker(self):
        pass
    
    def crop(self, job: Job):
        pass

    def reset_task(self):
        """
        Resets the task values based on the provided function type.

        Returns:
            None
        """

        self.bar_value, self.end_task, self.message_box = self.TASK_VALUES

    def _update_progress(self, file_amount: int) -> None:
        """
        Updates the progress bar value based on the process type.

        Args:
            self: The Cropper instance.
            file_amount (int): The total number of files to process.

        Returns:
            None
        """

        def _update_bar(attr_name: str, *,
                        progress_signal: pyqtBoundSignal,
                        finished_signal: pyqtBoundSignal) -> None:
            """Updates the progress bar value."""
            current_value = getattr(self, attr_name)
            current_value += 1
            setattr(self, attr_name, current_value)
            if current_value == file_amount:
                progress_signal.emit((file_amount, file_amount))
                finished_signal.emit()
            elif current_value < file_amount:
                progress_signal.emit((current_value, file_amount))

        _update_bar('bar_value', progress_signal=self.progress, finished_signal=self.finished)

    def terminate(self) -> None:
        """
        Terminates the specified series of tasks.

        Returns:
            None
        """

        self.end_task = True
        self.finished.emit()
