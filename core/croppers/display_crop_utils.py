from pathlib import Path
from typing import Optional

import cv2
from PyQt6.QtGui import QImage
from cachetools import cached, TTLCache

from core import processing as prc
from core.enums import FunctionType
from core.face_tools import FaceToolPair
from core.job import Job
from file_types import file_manager, FileCategory

RadioButtonTuple = tuple[bool, bool, bool, bool, bool, bool]
WidgetState = tuple[str, str, str, bool, bool, bool, int, int, int, int, int, int, int, RadioButtonTuple]

cache = TTLCache(maxsize=128, ttl=60)  # Entries expire after 60 seconds


@cached(cache)
def path_iterator(path: Path) -> Optional[Path]:
    if not path or not path.is_dir():
        return None
    
    return next(
        filter(
            lambda f: f.is_file() and file_manager.is_valid_type(f, FileCategory.PHOTO), path.iterdir()
        ),
        None
    )


def matlike_to_qimage(image: cv2.Mat) -> QImage:
    """
    Convert a BGR NumPy array (shape = [height, width, channels])
    to a QImage using QImage.Format_BGR888.
    """
    return QImage(image, image.shape[1], image.shape[0], image.strides[0], QImage.Format.Format_BGR888)


def perform_crop_helper(function_type: FunctionType,
                        widget_state: WidgetState,
                        img_path_str: str,
                        face_detection_tools: FaceToolPair) -> Optional[QImage]:
    # Unpack and validate widget state
    if not validate_widget_state(widget_state):
        return None

    # Extract the necessary paths
    next_img_path = get_image_path(function_type, widget_state[0])
    if not next_img_path:
        return None

    # Create the Job instance
    job = create_job(widget_state, img_path_str, function_type)

    # Process the image
    pic_array = prc.load_and_prepare_image(next_img_path, face_detection_tools, job)
    return None if pic_array is None else handle_face_detection(pic_array, job, face_detection_tools)


@cached(cache)
def validate_widget_state(widget_state: WidgetState) -> bool:
    # input_line_edit_text, width_line_edit_text, height_line_edit_text
    return all(widget_state[:3])


def get_image_path(function_type: FunctionType, input_line_edit_text: str) -> Optional[Path]:
    img_path = Path(input_line_edit_text)
    match function_type:
        case FunctionType.PHOTO:
            return img_path if img_path.is_file() else None
        case _:
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


def handle_face_detection(pic_array: cv2.Mat, job: Job, face_detection_tools: FaceToolPair) -> Optional[QImage]:
    if job.multi_face_job:
        pic = prc.annotate_faces(pic_array, job, face_detection_tools)
        # final_image = prc.convert_colour_space(pic) # Uncomment if needed
        return None if pic is None else matlike_to_qimage(pic)
    else:
        bounding_box = prc.detect_face_box(pic_array, job, face_detection_tools)
        if not bounding_box:
            return None

        # Create a pipeline with bounding box
        pipeline = prc.build_processing_pipeline(job, face_detection_tools, bounding_box, True)

        # Apply pipeline to original image
        processed = prc.run_processing_pipeline(pic_array, pipeline)

        return matlike_to_qimage(processed)
