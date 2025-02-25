import collections.abc as c
import re
from pathlib import Path
from typing import Optional

import cv2
import cv2.typing as cvt
import ffmpeg
import numpy as np
from PyQt6.QtWidgets import QLabel, QSlider

from core import utils as ut
from core.job import Job
from core.operation_types import FaceToolPair
from .cropper import Cropper


class VideoCropper(Cropper):
    def __init__(self, face_detection_tools: c.Iterator[FaceToolPair]):
        super().__init__()
        self.face_detection_tools = next(face_detection_tools)

    def ffmpeg_error(self, error: BaseException) -> None:
        """
        Raises a permission error if the destination directory is not writable.
        """
        return self._display_error(
            error,
            "Please check the video file and try again."
        )

    def grab_frame(self, position_slider: int, video_line_edit: str) -> Optional[cvt.MatLike]:
        """
        Grabs and returns a frame at the given millisecond position from a video file using FFmpeg.

        Args:
            position_slider (int): The time position (in milliseconds) to extract the frame.
            video_line_edit (str): The path to the video file.

        Returns:
            Optional[np.ndarray]: Extracted frame as a NumPy array in RGB format, or None if extraction fails.
        """

        try:
            # Convert milliseconds to seconds
            timestamp_seconds = position_slider / 1000.0

            # Probe video metadata to get width and height
            probe = ffmpeg.probe(video_line_edit)
            video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)

            if not video_stream:
                self.file_error()
                return None

            width, height = int(video_stream['width']), int(video_stream['height'])

            # Extract frame at the given timestamp
            out, _ = (
                ffmpeg.input(video_line_edit, ss=timestamp_seconds)
                .output('pipe:', format='rawvideo', pix_fmt='rgb24', vframes=1)
                .run(capture_stdout=True, quiet=True)
            )

            # Convert raw byte output to a NumPy array and reshape to image dimensions
            return np.frombuffer(out, np.uint8).reshape((height, width, 3))

        except ffmpeg.Error as e:
            self.ffmpeg_error(e)
            return None

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

        if (frame := self.grab_frame(timeline_slider.value(), job.video_path.as_posix())) is None:
            return None

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
                return None

            for i, image in enumerate(images):
                new_file_path = file_path.with_stem(f'{file_path.stem}_{i}')
                ut.save_image(image, new_file_path, job.gamma, is_tiff)
            return None

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

    def extract_frames(self, job: Job) -> None:
        """
        Extracts frames from a video using ffmpeg and saves them based on the job parameters.

        Args:
            self: The Cropper instance.
            job (Job): The job containing video path, start position, and stop position.

        Returns:
            None
        """

        if job.video_path is None or job.start_position is None or job.stop_position is None:
            return None

        if not job.destination:
            return None

        if not job.destination_accessible:
            return self.access_error()

        # Get video metadata
        try:
            probe = ffmpeg.probe(job.video_path.as_posix())
            video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)

            if not video_stream:
                self.file_error()
                return None
            
            fps = eval(video_stream['r_frame_rate'])  # Convert "30/1" to int
            width, height = int(video_stream['width']), int(video_stream['height'])

        except ffmpeg.Error as e:
            self.ffmpeg_error(e)
            return None

        start_frame = int(job.start_position * fps)
        end_frame = int(job.stop_position * fps)

        size = job.byte_size * (end_frame - start_frame)

        if job.available_space == 0 or job.available_space < size:
            return self.capacity_error()

        if self.MEM_FACTOR < 1:
            return self.memory_error()

        total_frames = 1 + end_frame - start_frame
        self.progress.emit(0, total_frames)
        self.started.emit()

        for frame_number in range(start_frame, end_frame + 1):
            if self.end_task:
                break

            frame = self.extract_frame_ffmpeg(job.video_path.as_posix(), frame_number, width, height)

            if frame is not None:
                file_enum = f"{job.video_path.stem}_frame_{frame_number:06d}"

                if job.multi_face_job:
                    self.process_multiface_frame_job(frame, job, file_enum, job.destination)
                else:
                    self.process_singleface_frame_job(frame, job, file_enum, job.destination)

            self._update_progress(total_frames)

            if self.progress_count == total_frames or self.end_task:
                self.show_message_box = False

    def extract_frame_ffmpeg(self, video_path: str, frame_number: int, width: int, height: int) -> Optional[np.ndarray]:
        """
        Extracts a single frame from a video using ffmpeg.

        Args:
            video_path (str): Path to the video file.
            frame_number (int): Frame number to extract.
            width (int): Width of the video.
            height (int): Height of the video.

        Returns:
            np.ndarray: Extracted frame as a NumPy array, or None if extraction fails.
        """

        try:
            out, _ = (
                ffmpeg.input(video_path, ss=frame_number / 30)  # Adjust if FPS is variable
                .output('pipe:', format='rawvideo', pix_fmt='rgb24', vframes=1)
                .run(capture_stdout=True, quiet=True)
            )

            result = np.frombuffer(out, np.uint8).reshape((height, width, 3))

            return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        except ffmpeg.Error as e:
            self.ffmpeg_error(e)
            return None
