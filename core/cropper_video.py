import collections.abc as c
import re
from pathlib import Path
from typing import Any

import cv2
import cv2.typing as cvt
from PyQt6.QtWidgets import QLabel, QSlider

from . import utils as ut
from .cropper import Cropper
from .job import Job
from .operation_types import FaceToolPair


class VideoCropper(Cropper):
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
                         progress_callback: c.Callable[..., Any]) -> None:
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

        x = 1 + end_frame - start_frame

        self.progress.emit((0, x))
        self.started.emit()

        for frame_number in range(start_frame, end_frame + 1):
            if self.end_task:
                break
            self.frame_extraction(video, frame_number, job,
                                  lambda: self._update_progress(x))
            if self.bar_value == x or self.end_task:
                self.message_box = False
        video.release()
