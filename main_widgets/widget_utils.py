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
def display_image_on_widget(image: cvt.MatLike) -> QImage:
    height, width, channel = image.shape
    bytes_per_line = channel * width
    return QImage(
        image.data.tobytes(),
        width,
        height,
        bytes_per_line,
        QImage.Format.Format_BGR888
    )

def crop_and_set(input_image: cvt.MatLike, bounding_box: Box, gamma_value: int) -> QImage:
    try:
        cropped_image = ut.numpy_array_crop(input_image, bounding_box)
        adjusted_image = ut.adjust_gamma(cropped_image, gamma_value)
        final_image = ut.convert_color_space(adjusted_image)
        return display_image_on_widget(final_image)
    except (cv2.error, Image.DecompressionBombError):
        return QImage()

def perform_crop_helper(
    function_type: FunctionType,
    widget_state: WidgetState,
    img_path_str: str,
    face_detection_tools: FaceToolPair
) -> Optional[QImage]:
    (
        input_line_edit_text,
        width_line_edit_text,
        height_line_edit_text,
        exposure_checkbox_checked,
        mface_checkbox_checked,
        tilt_checkbox_checked,
        sensitivity_dial_value,
        fpct_dial_value,
        gamma_dial_value,
        top_dial_value,
        bottom_dial_value,
        left_dial_value,
        right_dial_value,
        radio_tuple,
    ) = widget_state

    # Check if necessary fields are filled
    if not input_line_edit_text or not width_line_edit_text or not height_line_edit_text:
        return None

    img_path = Path(input_line_edit_text)
    if function_type == FunctionType.PHOTO:
        if not img_path.is_file():
            return None
        next_img_path = img_path
    else:
        next_img_path = path_iterator(img_path)
        if not next_img_path:
            return None

    # Create the Job instance
    job = Job(
        width=int(width_line_edit_text),
        height=int(height_line_edit_text),
        fix_exposure_job=exposure_checkbox_checked,
        multi_face_job=mface_checkbox_checked,
        auto_tilt_job=tilt_checkbox_checked,
        sensitivity=sensitivity_dial_value,
        face_percent=fpct_dial_value,
        gamma=gamma_dial_value,
        top=top_dial_value,
        bottom=bottom_dial_value,
        left=left_dial_value,
        right=right_dial_value,
        radio_buttons=radio_tuple,
        photo_path=img_path_str if function_type == FunctionType.PHOTO else None,
        folder_path=img_path_str if function_type != FunctionType.PHOTO else None,
    )

    # Open the image using your utility function
    pic_array = ut.open_pic(
        next_img_path,
        face_detection_tools,
        exposure=job.fix_exposure_job,
        tilt=job.auto_tilt_job
    )
    if pic_array is None:
        return None

    if job.multi_face_job:
        pic = ut.multi_box(pic_array, job)
        return display_image_on_widget(pic)
    else:
        bounding_box = ut.box_detect(pic_array, job)
        if bounding_box is None:
            return None
        return crop_and_set(pic_array, bounding_box, job.gamma)
