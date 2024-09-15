import collections.abc as c
import re
from pathlib import Path
from typing import Any

import cv2
import cv2.typing as cvt
from PyQt6.QtWidgets import QLabel, QSlider

from core import utils as ut
from core.job import Job
from core.operation_types import FaceToolPair
from .cropper import Cropper


class VideoCropper(Cropper):
    def __init__(self, face_detection_tools: list[FaceToolPair]):
        super().__init__()
        self.face_detection_tools = face_detection_tools[1]

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
            if (images := ut.multi_crop(frame, job, self.face_detection_tools)) is None:
                return

            for i, image in enumerate(images):
                new_file_path = file_path.with_stem(f'{file_path.stem}_{i}')
                ut.save_image(image, new_file_path, job.gamma, is_tiff)
            return

        cropped_image = ut.crop_image(frame, job, self.face_detection_tools)
        if cropped_image is not None:
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

        if (images := ut.multi_crop(frame, job, self.face_detection_tools)) is None:
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

        if (cropped_image := ut.crop_image(frame, job, self.face_detection_tools)) is None:
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
        
        if job.destination:
            # Check if the destination directory is writable.
            if not job.destination_accessible:
                return self.access_error()

            video = cv2.VideoCapture(job.video_path.as_posix())
            fps = video.get(cv2.CAP_PROP_FPS)
            start_frame, end_frame = int(job.start_position * fps), int(job.stop_position * fps)

            size = job.byte_size * (end_frame - start_frame)

            # Check if there is enough space on disk to process the files.
            if job.available_space == 0 or job.available_space < size:
                return self.capacity_error()

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
