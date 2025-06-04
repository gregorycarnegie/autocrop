import os
import random
from collections.abc import Callable, Iterator
from contextlib import suppress
from functools import cache, partial, reduce, singledispatch
from multiprocessing.managers import ListProxy
from pathlib import Path
from typing import TypeVar

import autocrop_rs.face_detection as r_face  # type: ignore
import autocrop_rs.image_processing as r_img  # type: ignore
import cv2
import cv2.typing as cvt
import numpy as np
import numpy.typing as npt
import polars as pl
import tifffile as tiff
from rawpy import ColorSpace  # type: ignore
from rawpy._rawpy import LibRawError

from core.colour_utils import adjust_gamma, ensure_rgb, normalize_image, to_grayscale
from core.crop_instruction import CropInstruction
from file_types import FileCategory, SignatureChecker, file_manager

from .config import config, logger
from .face_tools import (
    L_EYE_END,
    L_EYE_START,
    R_EYE_END,
    R_EYE_START,
    FaceToolPair,
    YuNetFaceDetector,
)
from .job import Job
from .operation_types import Box, CropFunction, Pipeline
from .protocols import ImageLoader, ImageOpener, ImageWriter, SimpleImageOpener, TableLoader

T = TypeVar('T', bound=cvt.MatLike)

def generate_crop_instructions(
        image_paths: list[Path],
        job: Job,
        face_detection_tools: FaceToolPair,
        output_paths: list[Path] | None,
        rejected_list: ListProxy
) -> list[CropInstruction]:
    """
    Phase 1: Detect faces and generate crop instructions without performing crops

    Args:
        image_paths: List of paths to images
        job: Job parameters
        face_detection_tools: Face detection tools
        output_paths: Optional list of output paths (for mapping operations)
        rejected_list: Optional shared list proxy to record rejected files

    Returns:
        List of CropInstruction objects
    """
    instructions: list[CropInstruction] = []
    job_params = serialize_job_parameters(job)

    for i, img_path in enumerate(image_paths):
        # Process each image
        output_path = determine_output_path(img_path, job, output_paths, i)
        image_instructions = process_single_image(
            img_path, output_path, job, face_detection_tools, job_params, rejected_list
        )
        instructions.extend(image_instructions)

    return instructions


def serialize_job_parameters(job: Job) -> dict:
    """Extract job parameters into a dictionary for serialization."""
    return {
        'width': job.width,
        'height': job.height,
        'fix_exposure_job': job.fix_exposure_job,
        'multi_face_job': job.multi_face_job,
        'auto_tilt_job': job.auto_tilt_job,
        'sensitivity': job.sensitivity,
        'face_percent': job.face_percent,
        'gamma': job.gamma,
        'top': job.top,
        'bottom': job.bottom,
        'left': job.left,
        'right': job.right,
        'radio_buttons': job.radio_tuple()
    }


def determine_output_path(
        img_path: Path,
        job: Job,
        output_paths: list[Path] | None,
        index: int
) -> str:
    """Determine the output path for an image."""
    if output_paths is None:
        # Standard folder operation
        return get_output_path(img_path, job.safe_destination, None, job.radio_choice()).as_posix()
    else:
        # Mapping operation
        return output_paths[index].as_posix()


def process_single_image(
        img_path: Path,
        output_path: str,
        job: Job,
        face_detection_tools: FaceToolPair,
        job_params: dict,
        rejected_list: ListProxy
) -> list[CropInstruction]:
    """Process a single image and return crop instructions."""
    # Load the image and detect faces
    image_array = load_and_prepare_image(img_path, face_detection_tools, job)
    if image_array is None:
        # Reject the file if it can't be loaded
        handle_rejected_image(img_path, job, rejected_list)
        return []

    if job.multi_face_job:
        return process_multi_face_image(
            img_path, output_path, image_array, job, face_detection_tools, job_params, rejected_list
        )
    else:
        return process_single_face_image(
            img_path, output_path, image_array, job, face_detection_tools, job_params, rejected_list
        )



def handle_rejected_image(img_path: Path, job: Job, rejected_list: ListProxy) -> None:
    """Handle rejected images by adding to rejected list if available, or moving them to the reject folder."""
    if job.safe_destination:
        reject(img_path, rejected_list)


def process_multi_face_image(
        img_path: Path,
        output_path: str,
        image_array: cvt.MatLike,
        job: Job,
        face_detection_tools: FaceToolPair,
        job_params: dict,
        rejected_list: ListProxy
) -> list[CropInstruction]:
    """Process an image in multi-face mode."""
    instructions = []

    # Get all faces in multi-face mode
    results = get_face_boxes(image_array, job, face_detection_tools)
    if not results:
        # Reject if no faces detected
        handle_rejected_image(img_path, job, rejected_list)
        return []

    # Create an instruction for each detected face
    for face_idx, (confidence, bounding_box) in enumerate(results):
        if confidence <= job.threshold:
            continue

        face_output_path = generate_face_output_path(img_path, output_path, job, face_idx)

        instructions.append(CropInstruction(
            file_path=img_path.as_posix(),
            output_path=face_output_path,
            bounding_box=bounding_box,
            job_params=job_params,
            multi_face=True,
            face_index=face_idx
        ))

    return instructions


def generate_face_output_path(img_path: Path, output_path: str, job: Job, face_idx: int) -> str:
    """Generate output path for a specific face in an image."""
    output_path_obj = Path(output_path)

    if job.safe_destination and not output_path_obj.is_relative_to(job.safe_destination):
        # Standard folder operation, add face index to output path
        return get_output_path(
            img_path, job.safe_destination, face_idx, job.radio_choice()
        ).as_posix()
    else:
        # Mapping operation or path already in destination
        return output_path_obj.with_stem(f"{output_path_obj.stem}_{face_idx}").as_posix()

def get_rotation_matrix(image: cvt.MatLike,
               face_detection_tools: FaceToolPair,
               job: Job) -> npt.NDArray[np.float64] | None:
    """
    Performs face alignment using OpenCV's Facemark model.

    Args:
        job: Job parameters
        image: Input image
        face_detection_tools: Tuple of (detector, facemark)

    Returns:
        Aligned image or original if no face detected
    """
    if not job.auto_tilt_job:
        return None

    detector, facemark = face_detection_tools

    # Optimize for smaller images for faster processing
    height, width = image.shape[:2]
    scale_factor = r_face.determine_scale_factor(width, height, config.face_scale_divisor)

    if scale_factor > 1:
        # Resize image for faster processing
        small_img = cv2.resize(image, (int(width / scale_factor), int(height / scale_factor)))
        faces = detector(small_img, job.threshold)
    else:
        small_img = image
        faces = detector(image, job.threshold)

    if not faces:
        return None

    # Find face with the highest confidence
    face = max(faces, key=lambda f: f.confidence)

    # Convert the detected face to OpenCV rect format required by Facemark
    faces_rect = np.array([[
        face.left,
        face.top,
        face.width,
        face.height
    ]])

    # Detect landmarks
    success, landmarks = facemark.fit(small_img, faces_rect)

    if not success or len(landmarks) == 0:
        return None  # Return the original image if landmark detection fails

    # Extract eye landmarks
    landmarks = landmarks[0][0]  # First face, first set of landmarks

    # Get left and right eye landmarks (indices 36-41 for left eye, 42-47 for right eye in the 68-point model)
    l_eye = np.ascontiguousarray(landmarks[L_EYE_START:L_EYE_END], dtype=np.float64)
    r_eye = np.ascontiguousarray(landmarks[R_EYE_START:R_EYE_END], dtype=np.float64)

    return r_img.get_rotation_matrix(l_eye, r_eye, scale_factor)

def process_single_face_image(
        img_path: Path,
        output_path: str,
        image_array: cvt.MatLike,
        job: Job,
        face_detection_tools: FaceToolPair,
        job_params: dict,
        rejected_list: ListProxy
) -> list[CropInstruction]:
    """Process an image in single-face mode."""
    # Single face mode
    rotation_matrix = get_rotation_matrix(image_array, face_detection_tools, job)

    bounding_box = detect_face_box(image_array, job, face_detection_tools)
    if bounding_box is None:
        # Reject if no face detected
        handle_rejected_image(img_path, job, rejected_list)
        return []

    instruction = CropInstruction(
        file_path=img_path.as_posix(),
        output_path=output_path,
        bounding_box=bounding_box,
        job_params=job_params,
        multi_face=False,
        face_index=None,
        rotation_matrix=rotation_matrix
    )

    return [instruction]

def execute_crop_instruction(instruction: CropInstruction) -> bool:
    """
    Phase 2: Execute a single crop instruction

    Args:
        instruction: The crop instruction to execute

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load the source image
        image = simple_opener(instruction.file_path)
        if image is None:
            return False

        # Recreate job parameters
        job_params = instruction.job_params
        job = Job(
            width=job_params['width'],
            height=job_params['height'],
            fix_exposure_job=job_params['fix_exposure_job'],
            multi_face_job=job_params['multi_face_job'],
            auto_tilt_job=job_params['auto_tilt_job'],
            sensitivity=job_params['sensitivity'],
            face_percent=job_params['face_percent'],
            gamma=job_params['gamma'],
            top=job_params['top'],
            bottom=job_params['bottom'],
            left=job_params['left'],
            right=job_params['right'],
            radio_buttons=job_params['radio_buttons'],
            destination=Path(os.path.dirname(instruction.output_path))
        )

        # Create the pipeline using just the bounding box (no face detection needed)
        pipeline = build_crop_instruction_pipeline(
            job, instruction.bounding_box, rotation_matrix=instruction.rotation_matrix
        )

        # Process the image
        processed_image = run_processing_pipeline(image, pipeline)

        # Save the result
        output_path = Path(instruction.output_path)
        is_tiff = output_path.suffix.lower() in {'.tif', '.tiff'}
        save(processed_image, output_path, job.gamma, is_tiff)

        return True
    except Exception as e:
        logger.exception(f"Error executing crop instruction: {e}")
        return False

def _open_standard_simple(
    file: Path
) -> cvt.MatLike | None:
    img = ImageLoader.loader('standard')(file.as_posix())
    return None if img is None else ensure_rgb(img)


def _open_raw_simple(
    file: Path
) -> cvt.MatLike | None:
    with suppress(
        LibRawError,
        MemoryError,
        ValueError,
        TypeError,
    ):
        with ImageLoader.loader('raw')(file.as_posix()) as raw:
            return raw.postprocess(
                use_camera_wb=True,
                no_auto_bright=True,
                output_color=ColorSpace.sRGB
            )

    return None


# Map each FileCategory to its opener strategy
_SIMPLE_OPENER_STRATEGIES: dict[FileCategory, SimpleImageOpener] = {
    FileCategory.PHOTO: _open_standard_simple,
    FileCategory.TIFF: _open_standard_simple,
    FileCategory.RAW: _open_raw_simple,
}

def simple_opener(
    file_str: str
) -> cvt.MatLike | None:
    """
    Open an image file using the appropriate strategy based on its FileCategory.
    Includes content verification for security.
    """
    file = Path(file_str)
    with suppress(IsADirectoryError):
        category = next(
            (
                cat
                for cat in [
                    FileCategory.PHOTO,
                    FileCategory.TIFF,
                    FileCategory.RAW,
                ]
                if file_manager.is_valid_type(file, cat)
            ),
            None,
        )
        # Verify file content before opening
        if not category:
            return None

        # Use the optimized file verification
        if not SignatureChecker.verify_file_type(file, category):
            return None

        if opener := _SIMPLE_OPENER_STRATEGIES.get(category):
            return opener(file)

    return None

def rotation_helper(
    image: cvt.MatLike,
    rotation_matrix: npt.NDArray[np.float64]
) -> cvt.MatLike:
    height, width = image.shape[:2]
    return cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=config.border_type
    )


def build_crop_instruction_pipeline(
        job: Job,
        bounding_box: Box,
        display: bool = False,
        video: bool = False,
        rotation_matrix: npt.NDArray[np.float64] | None = None
) -> Pipeline:
    """
    Creates a processing pipeline for executing a crop instruction.

    This function is specifically designed for Phase 2 of the two-phase approach,
    where face detection has already been performed and we have a bounding box.

    Args:
        job: Job parameters
        bounding_box: Pre-computed bounding box from Phase 1
        display: Whether the output is for display
        video: Whether the input is a video frame

    Returns:
        List of image processing functions to be applied in sequence
    """
    pipeline: Pipeline = []

    if rotation_matrix is not None:
        pipeline.append(partial(rotation_helper, rotation_matrix=rotation_matrix))

    pipeline.append(
        partial(crop_to_bounding_box, bounding_box=bounding_box)
    )

    # Add exposure correction if requested
    if job.fix_exposure_job:
        pipeline.append(partial(r_img.correct_exposure, exposure=True, video=video))

    # Add standard processing steps
    pipeline.extend(
        (
            partial(adjust_gamma, gam=job.gamma),
            partial(cv2.resize, dsize=job.size, interpolation=config.interpolation),
        )
    )

    # Add colour space conversion if needed
    if display or job.radio_choice() in ['.jpg', '.png', '.bmp', '.webp', 'No']:
        pipeline.append(ensure_rgb)

    return pipeline


def build_processing_pipeline(
        job: Job,
        face_detection_tools: FaceToolPair,
        bounding_box: Box | None=None,
        display=False,
        video=False
) -> Pipeline:
    """
    Creates a pipeline of image processing functions based on job parameters.
    """
    pipeline: Pipeline = []
    # Add alignment if requested
    if job.auto_tilt_job:
        pipeline.append(partial(align_face, face_detection_tools=face_detection_tools, job=job))

    # Add cropping first if a bounding box is provided
    if bounding_box is not None:
        pipeline.append(partial(crop_to_bounding_box, bounding_box=bounding_box))

    # Add exposure correction if requested
    if job.fix_exposure_job:
        pipeline.append(partial(r_img.correct_exposure, exposure=True, video=video))

    pipeline.extend(
        (
            partial(adjust_gamma, gam=job.gamma),
            partial(cv2.resize, dsize=job.size, interpolation=config.interpolation),
        )
    )
    # Add colour space conversion if needed
    if display or job.radio_choice() in ['.jpg', '.png', '.bmp', '.webp', 'No']:
        pipeline.append(ensure_rgb)

    return pipeline


def run_processing_pipeline(
        image: T,
        pipeline: list[Callable[[T], T]]
) -> T:
    """
    Apply a sequence of image processing functions to an image.
    """
    return reduce(lambda img, func: func(img), pipeline, image)


def crop_to_bounding_box(
        image: cvt.MatLike,
        bounding_box: Box
) -> cvt.MatLike:
    x0, y0, x1, y1 = bounding_box
    h, w = image.shape[:2]

    if x0 >= 0 and y0 >= 0 and x1 <= w and y1 <= h:
        return image[y0:y1, x0:x1]

    # Crop the valid region first
    cropped_valid = image[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    # Calculate padding needed
    if not any(padding := (max(0, -y0), max(0, y1 - h), max(0, -x0), max(0, x1 - w))):
        return cv2.Mat(cropped_valid)
    # Pad the image with black
    return cv2.copyMakeBorder(
        cropped_valid,
        *padding,
        config.border_type,
        value=config.border_colour,
    )


def prepare_visualisation_image(image: cvt.MatLike) -> tuple[cvt.MatLike, float]:
    """
    Resizes an image to 256 px height, returns grayscale if >2 channels, plus the scaling factor.
    """
    output_height = config.default_preview_height
    height, width = image.shape[:2]
    output_width, scaling_factor = r_img.calculate_dimensions(height, width, output_height)
    image_array = cv2.resize(image, (output_width, output_height), interpolation=config.interpolation)
    return to_grayscale(image_array) if len(image_array.shape) >= 3 else image_array, scaling_factor


def colour_and_align_face(
        image: cvt.MatLike,
        face_detection_tools: FaceToolPair,
        job: Job
) -> cvt.MatLike:
    # Convert BGR -> RGB for consistency
    return align_face(ensure_rgb(image), face_detection_tools, job)


def align_face(image: T,
               face_detection_tools: FaceToolPair,
               job: Job) -> T:
    """
    Performs face alignment using OpenCV's Facemark model.

    Args:
        job: Job parameters
        image: Input image
        face_detection_tools: Tuple of (detector, facemark)

    Returns:
        Aligned image or original if no face detected
    """
    if not job.auto_tilt_job:
        return image

    detector, facemark = face_detection_tools

    # Optimize for smaller images for faster processing
    height, width = image.shape[:2]
    scale_factor = r_face.determine_scale_factor(width, height, config.face_scale_divisor)

    if scale_factor > 1:
        # Resize image for faster processing
        small_img = cv2.resize(image, (int(width / scale_factor), int(height / scale_factor)))
        faces = detector(small_img, job.threshold)
    else:
        small_img = image
        faces = detector(image, job.threshold)

    if not faces:
        return image

    # Find face with the highest confidence
    face = max(faces, key=lambda f: f.confidence)

    # Convert the detected face to OpenCV rect format required by Facemark
    faces_rect = np.array([[
        face.left,
        face.top,
        face.width,
        face.height
    ]])

    # Detect landmarks
    success, landmarks = facemark.fit(small_img, faces_rect)

    if not success or len(landmarks) == 0:
        return image  # Return the original image if landmark detection fails

    # Extract eye landmarks
    landmarks = landmarks[0][0]  # First face, first set of landmarks

    # Get left and right eye landmarks (indices 36-41 for left eye, 42-47 for right eye in the 68-point model)
    l_eye = np.ascontiguousarray(landmarks[L_EYE_START:L_EYE_END], dtype=np.float64)
    r_eye = np.ascontiguousarray(landmarks[R_EYE_START:R_EYE_END], dtype=np.float64)

    rotation_matrix = r_img.get_rotation_matrix(l_eye, r_eye, scale_factor)
    return cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=config.border_type
    )


def _open_standard(
    file: Path,
    face_detection_tools: FaceToolPair,
    job: Job
) -> cvt.MatLike | None:
    img = ImageLoader.loader('standard')(file.as_posix())
    if img is None:
        return None
    return colour_and_align_face(img, face_detection_tools, job)


def _open_raw(
    file: Path,
    face_detection_tools: FaceToolPair,
    job: Job
) -> cvt.MatLike | None:
    with suppress(
        LibRawError,
        MemoryError,
        ValueError,
        TypeError,
    ):
        with ImageLoader.loader('raw')(file.as_posix()) as raw:
            img = raw.postprocess(
                use_camera_wb=True,
                no_auto_bright=True,
                output_color=ColorSpace.sRGB
            )
            return align_face(img, face_detection_tools, job)

    return None


# Map each FileCategory to its opener strategy
_OPENER_STRATEGIES: dict[FileCategory, ImageOpener] = {
    FileCategory.PHOTO: _open_standard,
    FileCategory.TIFF: _open_standard,
    FileCategory.RAW: _open_raw,
}


def load_and_prepare_image(
    file: Path,
    face_detection_tools: FaceToolPair,
    job: Job
) -> cvt.MatLike | None:
    """
    Open an image file using the appropriate strategy based on its FileCategory.
    Includes content verification for security.
    """
    with suppress(IsADirectoryError):
        category = next(
            (
                cat
                for cat in [
                    FileCategory.PHOTO,
                    FileCategory.TIFF,
                    FileCategory.RAW,
                ]
                if file_manager.is_valid_type(file, cat)
            ),
            None,
        )
        # Verify file content before opening
        if not category:
            return None

        # Use the optimized file verification
        if not SignatureChecker.verify_file_type(file, category):
            return None

        if opener := _OPENER_STRATEGIES.get(category):
            return opener(file, face_detection_tools, job)

    return None


def _load_csv(file: Path) -> pl.DataFrame:
    """
    Load a CSV file with header validation.
    """
    try:
        # First peek at the file to validate headers
        with file.open(mode='r', encoding='utf-8') as f:
            header_line = f.readline().strip()
            if not header_line:
                return pl.DataFrame()

        # If headers look valid, load the full file
        return pl.read_csv(file, infer_schema_length=config.infer_schema_length)
    except (pl.exceptions.PolarsError, UnicodeDecodeError, OSError):
        # Try with different encoding if the initial attempt fails
        try:
            return pl.read_csv(file, encoding='latin-1', infer_schema_length=config.infer_schema_length)
        except pl.exceptions.PolarsError:
            return pl.DataFrame()


def _load_excel(file: Path) -> pl.DataFrame:
    """
    Load an Excel file with validation.
    """
    with suppress(pl.exceptions.PolarsError):
        # Use read_excel with error handling
        return pl.read_excel(file)
    return pl.DataFrame()


def _load_parquet(file: Path) -> pl.DataFrame:
    """
    Load a Parquet file with validation.
    """
    with suppress(pl.exceptions.PolarsError):
        return pl.read_parquet(file)
    return pl.DataFrame()

# Map each extension to its loader strategy
_LOADER_STRATEGIES: dict[str, TableLoader] = {
    '.csv': _load_csv,
    '.xlsx': _load_excel,
    '.xlsm': _load_excel,
    '.xltx': _load_excel,
    '.xltm': _load_excel,
    '.parquet': _load_parquet,
}


def load_table(file: Path) -> pl.DataFrame:
    """
    Opens a tabular data file using the appropriate strategy based on the file type.
    Validates file headers and structure before loading.

    Args:
        file: Path to the table file

    Returns:
        pl.DataFrame: Loaded data frame or empty data frame if loading fails
    """
    if not file_manager.is_valid_type(file, FileCategory.TABLE):
        return pl.DataFrame()

    with suppress(IsADirectoryError):
        return next(
            (
                loader(file)
                for ext, loader in _LOADER_STRATEGIES.items()
                if file.suffix.lower() == ext
            ),
            pl.DataFrame()
        )
    # Return an empty DataFrame if loading fails
    return pl.DataFrame()


def draw_bounding_box_with_confidence(image: cvt.MatLike,
                                      confidence: np.float64,
                                      *,
                                      x0: int,
                                      y0: int,
                                      x1: int,
                                      y1: int) -> cvt.MatLike:
    """
    Draws a bounding box and confidence text onto an image.
    """
    colours = (255, 0, 0), (0, 255, 0), (0, 0, 255)
    colour = random.choice(colours)
    line_width = config.bbox_line_width
    text_offset = config.bbox_text_offset

    text = f"{confidence:.2f}%"
    y_text = y0 - text_offset if y0 > 20 else y0 + text_offset
    cv2.rectangle(image, (x0, y0), (x1, y1), colour, line_width)
    cv2.putText(image, text, (x0, y_text), cv2.FONT_HERSHEY_SIMPLEX, config.bbox_font_scale, colour, line_width)
    return image


def annotate_faces(image: cvt.MatLike,
                   job: Job,
                   face_detection_tools: FaceToolPair) -> cvt.MatLike | None:
    """
    Draws bounding boxes for all detected faces above a given threshold.

    Args:
        image: Input image
        job: Job parameters
        face_detection_tools: Tuple of (detector, landmark predictor)

    Returns:
        Image with bounding boxes drawn
    """
    # Get confidences and boxes
    results = get_face_boxes(image, job, face_detection_tools)
    if not results:
        return None
    # Adjust gamma and convert colour space for visualization
    rgb_image = run_processing_pipeline(image, [partial(adjust_gamma, gam=job.gamma), ensure_rgb])
    # Draw rectangle and confidence text
    for confidence, box in results:
        x0, y0, x1, y1 = box
        draw_bounding_box_with_confidence(rgb_image, np.float64(confidence), x0=x0, y0=y0, x1=x1, y1=y1)

    return rgb_image


def get_face_boxes(image: cvt.MatLike,
                   job: Job,
                   face_detection_tools: FaceToolPair) -> Iterator[tuple[float, Box]] | None:
    """
    Returns confidences and bounding boxes for all detected faces above the threshold.

    Args:
        image: Input image
        job: Job parameters
        face_detection_tools: Tuple of (detector, landmark predictor)

    Returns:
        Tuple of (confidences, boxes)
    """
    detector, _ = face_detection_tools

    # Use our optimized detector
    faces = detector(image, job.threshold)

    if not faces:
        return None

    # Extract confidences
    confidences = [face.confidence * 100 for face in faces]

    # Generate bounding boxes
    boxes: list[Box] = []
    for face in faces:
        if box := r_img.crop_positions(
            (face.left, face.top, face.width, face.height),
            job.face_percent,
            (job.width, job.height),
            (job.top, job.bottom, job.left, job.right),
        ):
            boxes.append(box)

    return zip(confidences, boxes)


def detect_faces(image: cvt.MatLike,
                 threshold: int,
                 detector: YuNetFaceDetector,
                 scale_factor: float) -> list[r_face.Rectangle]:
    """
    Detect faces in an image, with optional resizing for performance.

    Args:
        image: Input image
        threshold: Detection confidence threshold
        detector: Face detector object
        scale_factor: Scale factor for resizing

    Returns:
        List of detected faces
    """
    if scale_factor <= 1:
        # Small image, detect directly
        return detector(image, threshold)

    # Large image, resize for faster detection
    height, width = image.shape[:2]
    small_img = cv2.resize(image, (int(width / scale_factor), int(height / scale_factor)))
    return detector(small_img, threshold)


def detect_face_box(image: cvt.MatLike,
                    job: Job,
                    face_detection_tools: FaceToolPair) -> Box | None:
    """
    Detect face in an image with optimized performance.

    Args:
        image: Input image in OpenCV format
        job: Job configuration containing detection parameters
        face_detection_tools: Tuple of (detector, landmark predictor)

    Returns:
        Box coordinates if a face is detected, None otherwise
    """
    # with suppress(AttributeError, IndexError):
    height, width = image.shape[:2]
    detector, _ = face_detection_tools

    # Determine optimal scale factor for performance
    scale_factor = r_face.determine_scale_factor(width, height, config.face_scale_divisor)
    # Detect faces with appropriate scaling
    faces = detect_faces(image, job.threshold, detector, scale_factor)
    face = r_face.find_best_face(faces)
    # Exit early if no faces detected
    if not faces:
        return None

    # Find the face with the highest confidence
    face = max(faces, key=lambda f: f.confidence)
    # Calculate crop_from_path box using Rust module
    return r_img.crop_positions(
        face * scale_factor,
        job.face_percent,
        (job.width, job.height),
        (job.top, job.bottom, job.left, job.right)
    )

    # return None


def mask_extensions(file_list: npt.NDArray[np.str_], extensions: set[str]) -> tuple[npt.NDArray[np.bool_], int]:
    """
    Masks the file list based on supported extensions, returning the mask and its count.
    """
    if len(file_list) == 0:
        return np.array([], dtype=np.bool_), 0

    file_suffixes = np.array([Path(file).suffix.lower() for file in file_list])
    mask = np.isin(file_suffixes, list(extensions))
    return mask, np.count_nonzero(mask)


def split_by_cpus(mask: npt.NDArray[np.bool_],
                  core_count: int,
                  *file_lists: npt.NDArray[np.str_]) -> Iterator[list[npt.NDArray[np.str_]]]:
    """
    Splits the file list(s) based on a mask and the number of cores.
    """
    return map(lambda x: np.array_split(x[mask], core_count), file_lists)


def join_path_suffix(file_str: str, destination: Path) -> tuple[Path, bool]:
    """
    Joins a filename with a destination path and determines if it's a TIFF file.
    """

    path = destination.joinpath(file_str)
    return path, file_manager.should_use_tiff_save(path)


@cache
def set_filename(radio_options: tuple[str, ...],
                 *,
                 image_path: Path,
                 destination: Path,
                 radio_choice: str,
                 new: Path | str | None = None) -> tuple[Path, bool]:
    """
    Sets the output filename based on radio choice, RAW or non-RAW input, and optional custom filename.
    """
    suffix = image_path.suffix.lower()
    if file_manager.is_valid_type(image_path, FileCategory.RAW):
        selected_ext = radio_options[2] if radio_choice == radio_options[0] else radio_choice
    else:
        selected_ext = suffix if radio_choice == radio_options[0] else radio_choice

    final_name = f"{(new or image_path.stem)}{selected_ext}"
    return join_path_suffix(final_name, destination)


def reject(path: Path, rejected_list: ListProxy) -> None:
    """
    Records rejected file paths in a shared list proxy if provided,
    otherwise moves (copies) the file to 'rejects' folder under the given destination.

    Args:
        path: The path to the rejected file
        destination: The destination folder
        rejected_list: Optional shared list proxy to record rejects
    """
    rejected_list.append(path.as_posix())


def write_rejected_files_to_csv(rejected_list, destination: Path | None) -> Path | None:
    """
    Writes the list of rejected file paths to a CSV file.

    Args:
        rejected_list: Shared list proxy containing rejected file paths
        destination: The destination folder where the CSV should be saved

    Returns:
        Path to the CSV file or None if there was an error
    """
    if not rejected_list:
        return None

    if destination is None:
        destination = Path.home()

    try:
        # Create a unique filename with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = destination / f"rejected_files_{timestamp}.csv"

        # Create DataFrame and save to CSV
        df = pl.DataFrame({
            "rejected_file_path": list(rejected_list)
        })
        df.write_csv(csv_path)

        return csv_path
    except Exception as e:
        logger.exception(f"Error writing rejected files to CSV: {e}")
        return None

def make_frame_filepath(destination: Path,
                        file_enum: str,
                        job: Job) -> tuple[Path, bool]:
    """
    Constructs a filename for saving frames from a video.
    If radio_choice is 'original' (index 0), uses .jpg, otherwise uses the user-chosen extension.
    """

    file_str = f'{file_enum}.jpg' if job.radio_choice() == job.radio_options[0] else file_enum + job.radio_choice()
    return join_path_suffix(file_str, destination)


def _save_standard(image: cvt.MatLike,
      file_path: Path,
      _user_gam: float,
      _is_tiff: bool = False) -> None:
        lut = cv2.LUT(image, r_img.gamma(_user_gam * config.gamma_threshold))
        if file_path.suffix.lower() not in file_manager.get_save_formats(FileCategory.PHOTO):
            file_path = file_path.with_suffix('.jpg')
        cv2.imwrite(file_path.as_posix(), lut)


def _save_tiff(image: cvt.MatLike,
      file_path: Path,
      _user_gam: float,
      _is_tiff: bool = False) -> None:

        if image.dtype != np.uint8:
            image  = normalize_image(image)
        tiff.imwrite(file_path, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


_WRITER_STRATEGIES: dict[FileCategory, ImageWriter] = {
    FileCategory.PHOTO: _save_standard,
    FileCategory.TIFF: _save_tiff,
}


@singledispatch
def save(a0: Iterator | cvt.MatLike | np.ndarray | Path,
         *args,
         **kwargs) -> None:
    raise NotImplementedError(f"Unsupported input type: {type(a0)}")


@save.register
def _(image: cvt.MatLike | np.ndarray,
      file_path: Path,
      gamma_value: float,
      is_tiff: bool = False) -> None:
    """
    Saves an image to disk. If TIFF, uses tifffile; otherwise uses OpenCV.
    """
    image_type = FileCategory.TIFF if is_tiff else FileCategory.PHOTO
    strategy = _WRITER_STRATEGIES.get(image_type, _save_standard)
    strategy(image, file_path, gamma_value, is_tiff)


@save.register
def _(images: Iterator,
      file_path: Path,
      gamma_value: int,
      is_tiff: bool) -> None:
    """
    Saves multiple cropped images, enumerating filenames by appending '_0', '_1', etc.
    """

    for i, img in enumerate(images):
        new_file_path = file_path.with_stem(f'{file_path.stem}_{i}')
        save(img, new_file_path, gamma_value, is_tiff)


@save.register
def _(image: Path,
      job: Job,
      face_detection_tools: FaceToolPair,
      crop_function: CropFunction,
      new: Path | str | None = None) -> None:
    """
    Orchestrates face detection, cropping, and saving. Falls back to 'reject' if no faces are found.
    """

    if (destination_path := job.get_destination()) is None:
        return None

    if (cropped_images := crop_function(image, job, face_detection_tools)) is None:
        return None

    file_path, is_tiff = set_filename(
        job.radio_tuple(),
        image_path=image,
        destination=destination_path,
        radio_choice=job.radio_choice(),
        new=new
    )

    save(cropped_images, file_path, job.gamma, is_tiff)


def save_cropped_face(processed_image: cvt.MatLike,
                      output_path: Path,
                      gamma_value: int) -> None:
    """Save a processed face to the given output path"""
    is_tiff = output_path.suffix.lower() in {'.tif', '.tiff'}
    save(processed_image, output_path, gamma_value, is_tiff=is_tiff)


def save_video_frame(image: cvt.MatLike | np.ndarray,
                     file_enum_str: str,
                     destination: Path,
                     job: Job) -> None:
    """
    Saves a single frame from a video capture.
    """

    file_path, is_tiff = make_frame_filepath(destination, file_enum_str, job)
    save(image, file_path, job.gamma, is_tiff)


def process_image(image: cvt.MatLike,
                  job: Job,
                  bounding_box: Box,
                  face_detection_tools: FaceToolPair,
                  video: bool) -> cvt.MatLike:
    """
    Crops an image according to 'bounding_box', applies the processing pipeline, and resizes.
    """
    # Create and apply the processing pipeline
    pipeline = build_processing_pipeline(job, face_detection_tools, bounding_box, video=video)
    return run_processing_pipeline(image, pipeline)


@singledispatch
def crop_single_face(a0: cvt.MatLike | np.ndarray | Path, *args, **kwargs) -> cvt.MatLike | None:
    """
    Single-face cropping function. Returns the cropped face if found and resizes to `job.size`.
    """
    raise NotImplementedError(f"Unsupported input type: {type(a0)}")


@crop_single_face.register
def _(image: cvt.MatLike | np.ndarray,
      job: Job,
      face_detection_tools: FaceToolPair,
      video: bool=False) -> cvt.MatLike | None:
    if (bounding_box := detect_face_box(image, job, face_detection_tools)) is None:
        return None
    return process_image(image, job, bounding_box, face_detection_tools, video)


@crop_single_face.register
def _(image: Path,
      job: Job,
      face_detection_tools: FaceToolPair) -> cvt.MatLike | None:
    pic_array = load_and_prepare_image(image, face_detection_tools, job)
    if pic_array is None:
        return None
    return crop_single_face(pic_array, job, face_detection_tools)


@singledispatch
def crop_all_faces(a0: cvt.MatLike | np.ndarray | Path, *args, **kwargs) -> Iterator[cvt.MatLike] | None:
    """
    Multi-face cropping function. Yields cropped faces above the threshold, resized to `job.size`.
    """
    raise NotImplementedError(f"Unsupported input type: {type(a0)}")


@crop_all_faces.register
def _(image: cvt.MatLike | np.ndarray,
      job: Job,
      face_detection_tools: FaceToolPair,
      video: bool) -> Iterator[cvt.MatLike] | None:
    """
    Optimized multi-face cropping function using the pipeline approach.
    Yields cropped faces above the threshold, resized to `job.size`.
    """
    # Step 1: Detect faces and get bounding boxes
    results = get_face_boxes(image, job, face_detection_tools)

    if results is None:
        return None

    # Step 2: Filter faces based on the confidence threshold
    valid_faces = [
        (confidence, bounding_box) for confidence, bounding_box in results
        if confidence > job.threshold
    ]

    # If no valid faces were found, return None
    if not valid_faces:
        return None

    # Step 4: Process each face
    def process_face_box(bounding_box: Box) -> cvt.MatLike:
        # Create a pipeline specific to this face with its bounding box
        # This ensures alignment happens before cropping
        face_pipeline = build_processing_pipeline(job, face_detection_tools, bounding_box, video=video)

        # Apply the pipeline to the original image
        return run_processing_pipeline(image, face_pipeline)

    # Return a generator that processes each face on-demand
    return (process_face_box(bounding_box) for _, bounding_box in valid_faces)


@crop_all_faces.register
def _(image: Path,
      job: Job,
      face_detection_tools: FaceToolPair) -> Iterator[cvt.MatLike] | None:
    img = load_and_prepare_image(image, face_detection_tools, job)
    return None if img is None else crop_all_faces(img, job, face_detection_tools, video=False)


def get_output_path(input_path: Path,
                    destination: Path | None,
                    face_index: int | None,
                    radio_choice: str) -> Path:
    """Helper function to generate output paths."""
    suffix = input_path.suffix if radio_choice == 'No' else radio_choice
    if face_index is not None:
        # Multi-face output path
        stem = f"{input_path.stem}_{face_index}"
    else:
        # Single face output path
        stem = input_path.stem

    return (destination or Path.home()) / f"{stem}{suffix}"


@singledispatch
def crop_from_path(a0: Path | str, *args, **kwargs) -> None:
    """Applies cropping to an image based on the job configuration."""
    raise NotImplementedError(f"Unsupported input type: {type(a0)}")


@crop_from_path.register
def _(image: Path,
      job: Job,
      face_detection_tools: FaceToolPair,
      new: Path | str | None = None) -> None:
    crop_fn = crop_all_faces if job.multi_face_job else crop_single_face
    if all(x is not None for x in [job.table, job.safe_folder_path, new]):
        save(image, job, face_detection_tools, crop_fn, new)
    elif job.safe_folder_path is not None:
        save(image, job, face_detection_tools, crop_fn)
    else:
        save(image, job, face_detection_tools, crop_fn)


@crop_from_path.register
def _(image: str,
      job: Job,
      face_detection_tools: FaceToolPair,
      new: Path | str | None = None) -> None:
    crop_from_path(Path(image), job, face_detection_tools, new)
