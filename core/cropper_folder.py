from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import numpy.typing as npt

from . import utils as ut
from .cropper import Cropper
from .job import Job
from .operation_types import FaceToolPair


class FolderCropper(Cropper):
    """
    A class that represents a Cropper that inherits from the QObject class.

    Attributes:
        THREAD_NUMBER: ClassVar[int] = min(cpu_count(), 8)
        TASK_VALUES: ClassVar[Tuple[int, bool, bool]] = 0, False, True

    Methods:
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
    def __init__(self, face_detection_tools: list[FaceToolPair]):
        super().__init__()
        self.face_detection_tools = face_detection_tools

    def worker(self, file_amount: int,
               file_list: npt.NDArray[Any],
               job: Job,
               face_detection_tools: FaceToolPair) -> None:
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
            if self.end_task:
                break
            ut.crop(image, job, face_detection_tools)
            self._update_progress(file_amount)

        if self.bar_value == file_amount or self.end_task:
            self.message_box = False

    def crop(self, job: Job) -> None:
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

        self.bar_value = 0
        self.progress.emit((self.bar_value, amount))
        self.started.emit()

        executor = ThreadPoolExecutor(max_workers=self.THREAD_NUMBER)
        _ = [executor.submit(self.worker, amount, split_array[i], job, self.face_detection_tools[i])
             for i in range(len(split_array))]
