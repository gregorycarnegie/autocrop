from pathlib import Path
from typing import Optional

import cv2
import cv2.typing as cvt
from PIL import Image
from PyQt6.QtGui import QImage
from cachetools import cached, TTLCache

from core import utils as ut
from core.enums import FunctionType
from core.job import Job
from core.operation_types import Box, FaceToolPair
from file_types import Photo

RadioButtonTuple = tuple[bool, bool, bool, bool, bool, bool]
WidgetState = tuple[str, str, str, bool, bool, bool, int, int, int, int, int, int, int, RadioButtonTuple]

cache = TTLCache(maxsize=128, ttl=60)  # Entries expire after 60 seconds

@cached(cache)
def path_iterator(path: Path) -> Optional[Path]:
    if not path or not path.is_dir():
        return None
    return next(
        (
            file
            for file in path.iterdir()
            if file.is_file() and file.suffix.lower() in Photo.file_types
        ),
        None,
    )

# Define helper functions within the scope
def matlike_to_qimage(image: cvt.MatLike) -> QImage:
    height, width, channel = image.shape
    bytes_per_line = channel * width
    return QImage(
        image.data.tobytes(),
        width,
        height,
        bytes_per_line,
        QImage.Format.Format_BGR888
    )

def crop_to_qimage(input_image: cvt.MatLike,
                   bounding_box: Box,
                   gamma_value: int) -> QImage:
    try:
        cropped_image = ut.numpy_array_crop(input_image, bounding_box)
        adjusted_image = ut.adjust_gamma(cropped_image, gamma_value)
        final_image = ut.convert_color_space(adjusted_image)
        return matlike_to_qimage(final_image)
    except (cv2.error, Image.DecompressionBombError):
        return QImage()


def perform_crop_helper(function_type: FunctionType,
                        widget_state: WidgetState,
                        img_path_str: str,
                        face_detection_tools: FaceToolPair) -> Optional[QImage]:
    # Unpack and validate widget state
    if not validate_widget_state(widget_state):
        return None

    # Extract necessary paths
    next_img_path = get_image_path(function_type, widget_state[0])
    if not next_img_path:
        return None

    # Create the Job instance
    job = create_job(widget_state, img_path_str, function_type)

    # Process the image
    pic_array = ut.open_pic(next_img_path, face_detection_tools, exposure=job.fix_exposure_job, tilt=job.auto_tilt_job)
    return None if pic_array is None else handle_face_detection(pic_array, job)


@cached(cache)
def validate_widget_state(widget_state: WidgetState) -> bool:
    input_line_edit_text, width_line_edit_text, height_line_edit_text, *_ = widget_state
    return all([input_line_edit_text, width_line_edit_text, height_line_edit_text])


def get_image_path(function_type: FunctionType, input_line_edit_text: str) -> Optional[Path]:
    img_path = Path(input_line_edit_text)
    if function_type == FunctionType.PHOTO:
        return img_path if img_path.is_file() else None
    return path_iterator(img_path)


def create_job(widget_state: WidgetState, img_path_str: str, function_type: FunctionType) -> Job:
    return Job(
        width=int(widget_state[1]),
        height=int(widget_state[2]),
        fix_exposure_job=widget_state[3],
        multi_face_job=widget_state[4],
        auto_tilt_job=widget_state[5],
        sensitivity=widget_state[6],
        face_percent=widget_state[7],
        gamma=widget_state[8],
        top=widget_state[9],
        bottom=widget_state[10],
        left=widget_state[11],
        right=widget_state[12],
        radio_buttons=widget_state[13],
        photo_path=img_path_str if function_type == FunctionType.PHOTO else None,
        folder_path=img_path_str if function_type != FunctionType.PHOTO else None,
    )


def handle_face_detection(pic_array: cvt.MatLike, job: Job) -> Optional[QImage]:
    if job.multi_face_job:
        pic = ut.multi_box(pic_array, job)
        return matlike_to_qimage(pic)
    else:
        bounding_box = ut.box_detect(pic_array, job)
        return crop_to_qimage(pic_array, bounding_box, job.gamma) if bounding_box else None
