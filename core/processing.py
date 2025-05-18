import random
import shutil
import threading
from collections.abc import Callable, Iterator
from contextlib import suppress
from functools import cache, lru_cache, partial, singledispatch
from pathlib import Path

import autocrop_rs.image_processing as r_img
import cv2
import cv2.typing as cvt
import numpy as np
import numpy.typing as npt
import polars as pl
import tifffile as tiff
from PyQt6 import QtWidgets
from rawpy import ColorSpace  # type: ignore
from rawpy._rawpy import LibRawError

from core.colour_utils import adjust_gamma, ensure_rgb, normalize_image, to_grayscale
from file_types import FileCategory, SignatureChecker, file_manager

from .config import Config
from .face_tools import (
    L_EYE_END,
    L_EYE_START,
    R_EYE_END,
    R_EYE_START,
    FaceToolPair,
    Rectangle,
    YuNetFaceDetector,
)
from .job import Job
from .operation_types import Box, CropFunction
from .protocols import ImageLoader, ImageOpener, ImageWriter, TableLoader


def build_processing_pipeline(job: Job,
                              face_detection_tools: FaceToolPair,
                              bounding_box: Box | None=None,
                              display=False,
                              video=False) -> list[Callable[[cvt.MatLike], cvt.MatLike]]:
    """
    Creates a pipeline of image processing functions based on job parameters.
    """
    pipeline: list[Callable[[cvt.MatLike], cvt.MatLike]] = []
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
            partial(cv2.resize, dsize=job.size, interpolation=Config.interpolation),
        )
    )
    # Add colour space conversion if needed
    if display or job.radio_choice() in ['.jpg', '.png', '.bmp', '.webp', 'No']:
        pipeline.append(ensure_rgb)

    return pipeline


def run_processing_pipeline(image: cvt.MatLike, pipeline: list[Callable[[cvt.MatLike], cvt.MatLike]]) -> cvt.MatLike:
    """
    Apply a sequence of image processing functions to an image.
    """
    result = image
    for func in pipeline:
        result = func(result)
    return result


def crop_to_bounding_box(image: cvt.MatLike, bounding_box: Box) -> cvt.MatLike:
    x0, y0, x1, y1 = bounding_box
    h, w = image.shape[:2]
    # Crop the valid region first
    cropped_valid = image[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    # Calculate padding needed
    if not any(padding := (max(0, -y0), max(0, y1 - h), max(0, -x0), max(0, x1 - w))):
        return cv2.Mat(cropped_valid)
    # Pad the image with black
    return cv2.copyMakeBorder(
        cropped_valid,
        *padding,
        Config.border_type,
        value=Config.border_colour,
    )


def prepare_visualisation_image(image: cvt.MatLike) -> tuple[cvt.MatLike, float]:
    """
    Resizes an image to 256 px height, returns grayscale if >2 channels, plus the scaling factor.
    """
    output_height = Config.default_preview_height
    height, width = image.shape[:2]
    output_width, scaling_factor = r_img.calculate_dimensions(height, width, output_height)
    image_array = cv2.resize(image, (output_width, output_height), interpolation=Config.interpolation)
    return to_grayscale(image_array) if len(image_array.shape) >= 3 else image_array, scaling_factor


def colour_and_align_face(image: cvt.MatLike,
                          face_detection_tools: FaceToolPair,
                          job: Job) -> cvt.MatLike:
    # Convert BGR -> RGB for consistency
    return align_face(ensure_rgb(image), face_detection_tools, job)


def align_face(image: cvt.MatLike,
               face_detection_tools: FaceToolPair,
               job: Job) -> cvt.MatLike:
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
    scale_factor = determine_scale_factor(width, height)

    if scale_factor > 1:
        # Resize image for faster processing
        small_img = cv2.resize(image, (width // scale_factor, height // scale_factor))
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
        borderMode=Config.border_type
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
        return pl.read_csv(file, infer_schema_length=Config.infer_schema_length)
    except (pl.exceptions.PolarsError, UnicodeDecodeError, OSError):
        # Try with different encoding if the initial attempt fails
        try:
            return pl.read_csv(file, encoding='latin-1', infer_schema_length=Config.infer_schema_length)
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
    line_width = Config.bbox_line_width
    text_offset = Config.bbox_text_offset

    text = f"{confidence:.2f}%"
    y_text = y0 - text_offset if y0 > 20 else y0 + text_offset
    cv2.rectangle(image, (x0, y0), (x1, y1), colour, line_width)
    cv2.putText(image, text, (x0, y_text), cv2.FONT_HERSHEY_SIMPLEX, Config.bbox_font_scale, colour, line_width)
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


@lru_cache(maxsize=32)
def determine_scale_factor(width: int, height: int) -> int:
    """
    Determine the scale factor for face detection based on image dimensions.

    Args:
        width: Image width
        height: Image height

    Returns:
        Scale factor (1 for small images, larger for big images)
    """
    return max(1, min(width, height) // Config.face_scale_divisor)



def detect_faces(image: cvt.MatLike,
                 threshold: int,
                 detector: YuNetFaceDetector,
                 scale_factor: int) -> list[Rectangle]:
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
    small_img = cv2.resize(image, (width // scale_factor, height // scale_factor))
    return detector(small_img, threshold)



def scale_face_coordinates(face: Rectangle, scale_factor: int) -> tuple[int, int, int, int]:
    """
    Scale face coordinates based on the scale factor.

    Args:
        face: Detected face object
        scale_factor: Scale factor that was used for detection

    Returns:
        Tuple of (x, y, width, height) with adjusted coordinates
    """
    if scale_factor > 1:
        return (
            face.left * scale_factor,
            face.top * scale_factor,
            face.width * scale_factor,
            face.height * scale_factor
        )
    return face.left, face.top, face.width, face.height



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
    with suppress(AttributeError, IndexError):
        height, width = image.shape[:2]
        detector, _ = face_detection_tools

        # Determine optimal scale factor for performance
        scale_factor = determine_scale_factor(width, height)

        # Detect faces with appropriate scaling
        faces = detect_faces(image, job.threshold, detector, scale_factor)

        # Exit early if no faces detected
        if not faces:
            return None

        # Find the face with the highest confidence
        face = max(faces, key=lambda f: f.confidence)

        # Scale coordinates if needed
        x0, y0, face_width, face_height = scale_face_coordinates(face, scale_factor)

        # Calculate crop_from_path box using Rust module
        return r_img.crop_positions(
            (x0, y0, face_width, face_height),
            job.face_percent,
            (job.width, job.height),
            (job.top, job.bottom, job.left, job.right)
        )

    return None


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


def reject(*,
           path: Path,
           destination: Path | None) -> None:
    """
    Moves (copies) the file to 'rejects' folder under the given destination.
    """

    reject_folder = (destination or Path.home()).joinpath('rejects')
    reject_folder.mkdir(exist_ok=True)
    shutil.copy(path, reject_folder.joinpath(path.name))


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
        lut = cv2.LUT(image, r_img.gamma(_user_gam * Config.gamma_threshold))
        if file_path.suffix.lower() not in file_manager.get_save_formats(FileCategory.PHOTO):
            file_path = file_path.with_suffix('.jpg')
        cv2.imwrite(file_path.as_posix(), lut)


def _save_tiff(image: cvt.MatLike,
      file_path: Path,
      _user_gam: float,
      _is_tiff: bool = False) -> None:

        if image.dtype != np.uint8:
            image  = normalize_image(image)
        tiff.imwrite(file_path, image)


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
        return

    if (cropped_images := crop_function(image, job, face_detection_tools)) is None:
        reject(path=image, destination=destination_path)
        return

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


def batch_process_with_pipeline(images: list[Path],
                                job: Job,
                                face_detection_tools: FaceToolPair,
                                cancel_event: threading.Event,
                                video: bool,
                                chunk_size: int = 10) -> list[Path]:
    """
    Process a batch of images with the same pipeline for efficiency with cancellation support.
    """
    pipeline = []
    all_output_paths = []
    total_images = len(images)

    # Check for cancellation before starting
    if cancel_event.is_set():
        return all_output_paths

    # Process images in smaller chunks
    for i in range(0, total_images, chunk_size):
        # Check for cancellation
        if cancel_event.is_set():
            return all_output_paths

        # Get current chunk
        chunk = images[i:min(i + chunk_size, total_images)]

        # Process each image in the chunk
        for img_path in chunk:
            # Check for cancellation BEFORE processing each image
            if cancel_event.is_set():
                return all_output_paths

            # Open the image
            image_array = load_and_prepare_image(img_path, face_detection_tools, job)

            # Create a function to get output paths for standard batch processing
            def get_output_path_fn(image_path: Path, face_index: int | None) -> Path:
                return get_output_path(image_path, job.safe_destination, face_index, job.radio_choice())

            # Process the image
            output_paths, pipeline = process_batch_item(
                image_array, job, face_detection_tools, pipeline,
                img_path, get_output_path_fn, video
            )

            all_output_paths.extend(output_paths)

            # Check for cancellation AFTER processing each image
            if cancel_event.is_set():
                return all_output_paths

        # Allow UI updates between chunks
        QtWidgets.QApplication.processEvents()

    return all_output_paths


def batch_process_with_mapping(images: list[Path],
                               output_paths: list[Path],
                               job: Job,
                               face_detection_tools: FaceToolPair,
                               cancel_event: threading.Event,
                               video: bool,
                               chunk_size: int = 10) -> list[Path]:
    """
    Process a batch of images with custom output paths using the same pipeline with cancellation support.
    """
    if len(images) != len(output_paths):
        raise ValueError("Input and output path lists must have same length")

    pipeline = []
    all_output_paths = []
    total_images = len(images)

    # Process images in chunks
    for i in range(0, total_images, chunk_size):
        # Check for cancellation BEFORE processing chunk
        if cancel_event.is_set():
            return all_output_paths

        # Get current chunk
        chunk_end = min(i + chunk_size, total_images)
        img_chunk = images[i:chunk_end]
        out_chunk = output_paths[i:chunk_end]

        # Process each image in the chunk with cancellation checks
        for img_path, out_path in zip(img_chunk, out_chunk):
            # Check for cancellation BEFORE processing each image
            if cancel_event.is_set():
                return all_output_paths

            # Open the image
            image_array = load_and_prepare_image(img_path, face_detection_tools, job)

            # Create a function to get output paths for mapping
            def get_output_path_fn(_image_path: Path, face_index: int | None) -> Path:
                if face_index is not None:
                    # Multi-face output path
                    return out_path.with_stem(f"{out_path.stem}_{face_index}")
                else:
                    # Single face output path
                    return out_path

            # Process the image
            output_paths_result, pipeline = process_batch_item(
                image_array, job, face_detection_tools, pipeline,
                img_path, get_output_path_fn, video
            )

            all_output_paths.extend(output_paths_result)

            # Check for cancellation AFTER processing each image
            if cancel_event.is_set():
                return all_output_paths

        # Allow UI to update between chunks
        QtWidgets.QApplication.processEvents()

        # Final cancellation check after UI updates
        if cancel_event.is_set():
            return all_output_paths

    return all_output_paths


def process_batch_item(image_array: cvt.MatLike | None,
                       job: Job,
                       face_detection_tools: FaceToolPair,
                       pipeline: list,
                       img_path: Path,
                       get_output_path_fn: Callable,
                       video: bool) -> tuple[list[Path], list]:
    """
    Process a single image from a batch with the given pipeline.
    Returns a tuple of (output_paths, pipeline)
    """
    if image_array is None:
        # If image loading fails, reject the file
        reject(path=img_path, destination=job.safe_destination)
        return [], pipeline
    output_paths = []

    def batch_helper(_bounding_box: Box,  face_index: int | None=None) -> None:
        # Create an output path using the provided function
        output_path = get_output_path_fn(img_path, face_index)
        # Create a pipeline specific to this face with its bounding box
        face_pipeline = build_processing_pipeline(job, face_detection_tools, _bounding_box, video=video)
        # Apply the pipeline to the original image
        processed = run_processing_pipeline(image_array, face_pipeline)
        # Save the processed image
        save_cropped_face(processed, output_path, job.gamma)
        output_paths.append(output_path)

    # Process based on multi-face setting
    if job.multi_face_job:
        # Multi-face processing
        results = get_face_boxes(image_array, job, face_detection_tools)

        if not results:
            reject(path=img_path, destination=job.safe_destination)
            return output_paths, pipeline

        valid_positions = [pos for confidence, pos in results if confidence > job.threshold]

        if not valid_positions:
            return output_paths, pipeline

        # Process each face
        for i, bounding_box in enumerate(valid_positions):
            batch_helper(bounding_box, i)
    else:
        # Single face processing
        if (bounding_box := detect_face_box(image_array, job, face_detection_tools)) is None:
            reject(path=img_path, destination=job.safe_destination)
            return output_paths, pipeline

        batch_helper(bounding_box)

    return output_paths, pipeline


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
