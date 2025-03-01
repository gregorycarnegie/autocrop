import collections.abc as c
from fractions import Fraction
from functools import cache
from pathlib import Path
from typing import Optional

import ffmpeg
import numpy as np
from PyQt6.QtWidgets import QLabel, QSlider

from core import utils as ut
from core.job import Job
from core.operation_types import FaceToolPair
from .cropper import Cropper

@cache
def get_video_stream(video_line_edit: str) -> Optional[dict]:
    probe = ffmpeg.probe(video_line_edit)
    return next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)

@cache
def frame_to_timestamp(frame_number: int, fps: float) -> float:
    return frame_number / fps

@cache
def ffmpeg_input(video_line_edit: str, timestamp_seconds: float, width: int, height: int) -> np.ndarray:
    out, _ = (
        ffmpeg.input(video_line_edit, ss=timestamp_seconds)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24', vframes=1)
        .run(capture_stdout=True, quiet=True)
    )
    return np.frombuffer(out, np.uint8).reshape((height, width, 3))

class VideoCropper(Cropper):
    def __init__(self, face_detection_tools: c.Iterator[FaceToolPair]):
        super().__init__()
        self.face_detection_tools = next(face_detection_tools)

    def ffmpeg_error(self, error: BaseException, message: str = "Please check the video file and try again.") -> None:
        """
        Raises a permission error if the destination directory is not writable.
        """
        return self._display_error(
            error,
            message
        )

    def grab_frame(self, position_slider: int, video_line_edit: str) -> Optional[np.ndarray]:
        try:
            timestamp_seconds = frame_to_timestamp(position_slider, 1000.0)
            video_stream = get_video_stream(video_line_edit)
            if not video_stream:
                self.file_error("Video Stream not found")
                return None
            width, height = int(video_stream['width']), int(video_stream['height'])
            return ffmpeg_input(video_line_edit, timestamp_seconds, width, height)
        except ffmpeg.Error as e:
            self.ffmpeg_error(e, "Unable to grab frame")
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
            return None

        if (frame := self.grab_frame(timeline_slider.value(), job.video_path.as_posix())) is None:
            return None

        destination = job.destination
        # base_name = job.video_path.stem
        destination.mkdir(exist_ok=True)

        # Swap ':' to '_' in position text
        position = position_label.text().replace(':', '_')

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

    def process_multiface_frame_job(self, frame: np.ndarray,
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
            ut.save_image(frame, file_path, job.gamma, is_tiff)
        else:
            for i, image in enumerate(images):
                file_path, is_tiff = ut.get_frame_path(destination, f'{file_enum}_{i}', job)
                ut.save_image(image, file_path, job.gamma, is_tiff)

    def process_singleface_frame_job(self, frame: np.ndarray,
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
            ut.frame_save(cropped_image, file_enum, destination, job)

    def extract_frame_ffmpeg(self, video_path: str, frame_number: int, width: int, height: int, fps: float) -> Optional[np.ndarray]:
        try:
            timestamp = frame_to_timestamp(frame_number, fps)
            return ffmpeg_input(video_path, timestamp, width, height)
        except ffmpeg.Error as e:
            self.ffmpeg_error(e, "Error extracting frame")
            return None

    def extract_frames(self, job: Job) -> None:
        if not job.video_path or not job.start_position or not job.stop_position or not job.destination:
            return None
        if not job.destination_accessible:
            return self.access_error()

        try:
            video_stream = get_video_stream(job.video_path.as_posix())
            if not video_stream:
                self.file_error("Video Stream not found")
                return None
            fps = float(Fraction(video_stream['r_frame_rate']))
            width, height = int(video_stream['width']), int(video_stream['height'])
        except ffmpeg.Error as e:
            self.ffmpeg_error(e, "Error extracting frames")
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

            frame = self.extract_frame_ffmpeg(job.video_path.as_posix(), frame_number, width, height, fps)

            if frame is not None:
                file_enum = f"{job.video_path.stem}_frame_{frame_number:06d}"

                if job.multi_face_job:
                    self.process_multiface_frame_job(frame, job, file_enum, job.destination)
                else:
                    self.process_singleface_frame_job(frame, job, file_enum, job.destination)

            self._update_progress(total_frames)

            if self.progress_count == total_frames or self.end_task:
                self.show_message_box = False
