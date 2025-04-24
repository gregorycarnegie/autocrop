import cProfile
import pstats
import random
import shutil
import threading
from collections.abc import Callable, Iterator
from contextlib import suppress
from functools import cache, wraps, singledispatch, partial
from pathlib import Path
from typing import Any, Union, Optional

import autocrop_rs as rs
import cv2
import numpy as np
import numpy.typing as npt
import polars as pl
import tifffile as tiff
from PyQt6 import QtWidgets, QtCore
from rawpy import ColorSpace
from rawpy._rawpy import NotSupportedError, LibRawError, LibRawFatalError, LibRawNonFatalError

from core.colour_utils import ensure_rgb, to_grayscale, adjust_gamma, normalize_image
from file_types import file_manager, FileCategory
from .config import Config
from .face_tools import L_EYE_START, L_EYE_END, R_EYE_START, R_EYE_END, FaceToolPair, YuNetFaceDetector, Rectangle
from .protocols import ImageLoader, ImageOpener, ImageWriter, TableLoader
from .job import Job
from .operation_types import CropFunction, Box


def profile_it(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator to profile a function and print cumulative statistics.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any):
        with cProfile.Profile() as profile:
            func(*args, **kwargs)
        result = pstats.Stats(profile)
        result.sort_stats(pstats.SortKey.CUMULATIVE)
        result.print_stats()

    return wrapper


def build_processing_pipeline(job: Job,
                              face_detection_tools: FaceToolPair,
                              bounding_box: Optional[Box]=None,
                              display=False,
                              video=False) -> list[Callable[[cv2.Mat], cv2.Mat]]:
    """
    Creates a pipeline of image processing functions based on job parameters.
    """
    pipeline: list[Callable[[cv2.Mat], cv2.Mat]] = []
    # Add alignment if requested
    if job.auto_tilt_job:
        pipeline.append(partial(align_face, face_detection_tools=face_detection_tools, job=job))

    # Add cropping first if a bounding box is provided
    if bounding_box is not None:
        pipeline.append(partial(crop_to_bounding_box, bounding_box=bounding_box))

    # Add exposure correction if requested
    if job.fix_exposure_job:
        pipeline.append(partial(rs.correct_exposure, exposure=True, video=video))

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


def run_processing_pipeline(image: cv2.Mat, pipeline: list[Callable[[cv2.Mat], cv2.Mat]]) -> cv2.Mat:
    """
    Apply a sequence of image processing functions to an image.
    """
    result = image
    for func in pipeline:
        result = func(result)
    return result


def crop_to_bounding_box(image: cv2.Mat, bounding_box: Box) -> cv2.Mat:
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


def prepare_visualisation_image(image: cv2.Mat) -> tuple[cv2.Mat, float]:
    """
    Resizes an image to 256 px height, returns grayscale if >2 channels, plus the scaling factor.
    """
    output_height = Config.default_preview_height
    output_width, scaling_factor = rs.calculate_dimensions(*image.shape[:2], output_height)
    image_array = cv2.resize(image, (output_width, output_height), interpolation=Config.interpolation)
    return to_grayscale(image_array) if len(image_array.shape) >= 3 else image_array, scaling_factor


def colour_and_align_face(image: cv2.Mat,
                          face_detection_tools: FaceToolPair,
                          job: Job) -> cv2.Mat:
    # Convert BGR -> RGB for consistency
    return align_face(ensure_rgb(image), face_detection_tools, job)


def align_face(image: cv2.Mat,
               face_detection_tools: FaceToolPair,
               job: Job) -> cv2.Mat:
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
        return image  # Return original image if landmark detection fails
    
    # Extract eye landmarks
    landmarks = landmarks[0][0]  # First face, first set of landmarks
    
    # Get left and right eye landmarks (indices 36-41 for left eye, 42-47 for right eye in 68-point model)
    l_eye = np.ascontiguousarray(landmarks[L_EYE_START:L_EYE_END], dtype=np.float64)
    r_eye = np.ascontiguousarray(landmarks[R_EYE_START:R_EYE_END], dtype=np.float64)

    rotation_matrix = rs.get_rotation_matrix(l_eye, r_eye, scale_factor)
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
) -> Optional[cv2.Mat]:
    img = ImageLoader.loader('standard')(file.as_posix())
    if img is None:
        return None
    return colour_and_align_face(img, face_detection_tools, job)


def _open_raw(
    file: Path,
    face_detection_tools: FaceToolPair,
    job: Job
) -> Optional[cv2.Mat]:
    with suppress(
        NotSupportedError,
        LibRawFatalError,
        LibRawError,
        LibRawNonFatalError,
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
) -> Optional[cv2.Mat]:
    """
    Open an image file using the appropriate strategy based on its FileCategory.
    """
    with suppress(IsADirectoryError):
        return next(
            (
                opener(file, face_detection_tools, job)
                for category, opener in _OPENER_STRATEGIES.items()
                if file_manager.is_valid_type(file, category)
            ),
            None,
        )
    return None


def _load_csv(file: Path) -> Optional[pl.DataFrame]:
    """
    Load a CSV file with header validation.
    """
    try:
        # First peek at the file to validate headers
        with open(file, 'r', encoding='utf-8') as f:
            header_line = f.readline().strip()
            if not header_line:
                return None
        
        # If headers look valid, load the full file
        return pl.read_csv(file, infer_schema_length=1000)
    except (pl.NoDataError, UnicodeDecodeError, pl.ComputeError):
        # Try with different encoding if initial attempt fails
        try:
            return pl.read_csv(file, encoding='latin-1', infer_schema_length=1000)
        except Exception:
            return None
    except Exception:
        return None


def _load_excel(file: Path) -> Optional[pl.DataFrame]:
    """
    Load an Excel file with validation.
    """
    with suppress(Exception):
        # Use read_excel with error handling
        return pl.read_excel(file)


def _load_parquet(file: Path) -> Optional[pl.DataFrame]:
    """
    Load a Parquet file with validation.
    """
    with suppress(Exception):
        return pl.read_parquet(file)

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
    Opens a tabular data file using appropriate strategy based on file type.
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
    # Return empty DataFrame if loading fails
    return pl.DataFrame()


def draw_bounding_box_with_confidence(image: cv2.Mat,
                                      confidence: np.float64,
                                      *,
                                      x0: int,
                                      y0: int,
                                      x1: int,
                                      y1: int) -> cv2.Mat:
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


def annotate_faces(image: Union[cv2.Mat, np.ndarray],
                   job: Job,
                   face_detection_tools: FaceToolPair) -> Optional[cv2.Mat]:
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


def get_face_boxes(image: cv2.Mat,
                   job: Job,
                   face_detection_tools: FaceToolPair) -> Optional[Iterator[tuple[float, Box]]]:
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
        if box := rs.crop_positions(
            (face.left, face.top, face.width, face.height),
            job.face_percent,
            (job.width, job.height),
            (job.top, job.bottom, job.left, job.right),
        ):
            boxes.append(box)

    return zip(confidences, boxes)


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



def detect_faces(image: cv2.Mat,
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



def detect_face_box(image: cv2.Mat,
                    job: Job,
                    face_detection_tools: FaceToolPair) -> Optional[Box]:
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
        return rs.crop_positions(
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
                 new: Optional[Union[Path, str]] = None) -> tuple[Path, bool]:
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
           destination: Path) -> None:
    """
    Moves (copies) the file to 'rejects' folder under the given destination.
    """

    reject_folder = destination.joinpath('rejects')
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


def _save_standard(image: Union[cv2.Mat, np.ndarray],
      file_path: Path,
      user_gam: float,
      is_tiff: bool = False) -> None:
        lut = cv2.LUT(image, rs.gamma(user_gam * Config.gamma_threshold))
        cv2.imwrite(file_path.as_posix(), lut)


def _save_tiff(image: Union[cv2.Mat, np.ndarray],
      file_path: Path,
      user_gam: float,
      is_tiff: bool = False) -> None:
        if image.dtype != np.uint8:
            image  = normalize_image(image)
        tiff.imwrite(file_path, image)


_WRITER_STRATEGIES: dict[FileCategory, ImageWriter] = {
    FileCategory.PHOTO: _save_standard,
    FileCategory.TIFF: _save_tiff,
}


@singledispatch
def save(a0: Union[Iterator, cv2.Mat, np.ndarray, Path],
         *args,
         **kwargs) -> None:
    raise NotImplementedError(f"Unsupported input type: {type(a0)}")


@save.register
def _(image: Union[cv2.Mat, np.ndarray],
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
      new: Optional[Union[Path, str]] = None) -> None:
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


def save_cropped_face(processed_image: cv2.Mat,
                      output_path: Path,
                      gamma_value: int) -> None:
    """Save a processed face to the given output path"""
    is_tiff = output_path.suffix.lower() in {'.tif', '.tiff'}
    save(processed_image, output_path, gamma_value, is_tiff=is_tiff)


def save_video_frame(image: Union[cv2.Mat, np.ndarray],
                     file_enum_str: str,
                     destination: Path,
                     job: Job) -> None:
    """
    Saves a single frame from a video capture.
    """

    file_path, is_tiff = make_frame_filepath(destination, file_enum_str, job)
    save(image, file_path, job.gamma, is_tiff)


def process_image(image: cv2.Mat,
                  job: Job,
                  bounding_box: Box,
                  face_detection_tools: FaceToolPair,
                  video: bool) -> cv2.Mat:
    """
    Crops an image according to 'bounding_box', applies the processing pipeline, and resizes.
    """
    # Create and apply the processing pipeline
    pipeline = build_processing_pipeline(job, face_detection_tools, bounding_box, video=video)
    return run_processing_pipeline(image, pipeline)


@singledispatch
def crop_single_face(a0: Union[cv2.Mat, np.ndarray, Path], *args, **kwargs) -> Optional[cv2.Mat]:
    """
    Single-face cropping function. Returns the cropped face if found and resizes to `job.size`.
    """
    raise NotImplementedError(f"Unsupported input type: {type(a0)}")


@crop_single_face.register
def _(image: Union[cv2.Mat, np.ndarray],
      job: Job,
      face_detection_tools: FaceToolPair,
      video: bool=False) -> Optional[cv2.Mat]:
    if (bounding_box := detect_face_box(image, job, face_detection_tools)) is None:
        return None
    return process_image(image, job, bounding_box, face_detection_tools, video)


@crop_single_face.register
def _(image: Path,
      job: Job,
      face_detection_tools: FaceToolPair) -> Optional[cv2.Mat]:
    pic_array = load_and_prepare_image(image, face_detection_tools, job)
    if pic_array is None:
        return None
    return crop_single_face(pic_array, job, face_detection_tools)


@singledispatch
def crop_all_faces(a0: Union[cv2.Mat, np.ndarray, Path], *args, **kwargs) -> Optional[Iterator[cv2.Mat]]:
    """
    Multi-face cropping function. Yields cropped faces above the threshold, resized to `job.size`.
    """
    raise NotImplementedError(f"Unsupported input type: {type(a0)}")


@crop_all_faces.register
def _(image: Union[cv2.Mat, np.ndarray],
      job: Job,
      face_detection_tools: FaceToolPair,
      video: bool) -> Optional[Iterator[cv2.Mat]]:
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
    def process_face_box(bounding_box: Box) -> cv2.Mat:
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
      face_detection_tools: FaceToolPair) -> Optional[Iterator[cv2.Mat]]:
    img = load_and_prepare_image(image, face_detection_tools, job)
    return None if img is None else crop_all_faces(img, job, face_detection_tools, video=False)


def batch_process_with_pipeline(images: list[Path],
                                job: Job,
                                face_detection_tools: FaceToolPair,
                                cancel_event: threading.Event,
                                video: bool,
                                progress_bars: list[QtWidgets.QProgressBar] = None,
                                progress_count: int = 0,
                                chunk_size: int = 10) -> list[Path]:
    """
    Process a batch of images with the same pipeline for efficiency with cancellation support.
    """
    pipeline = None
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
            def get_output_path_fn(image_path: Path, face_index: Optional[int]) -> Path:
                return get_output_path(image_path, job.destination, face_index, job.radio_choice())

            # Process the image
            output_paths, pipeline = process_batch_item(
                image_array, job, face_detection_tools, pipeline, 
                img_path, get_output_path_fn, video
            )
            
            all_output_paths.extend(output_paths)
            
            # Check for cancellation AFTER processing each image
            if cancel_event.is_set():
                return all_output_paths
        
        progress_count = invoke_progress_by_chunk(
            len(chunk),  # Size of the current chunk 
            progress_count,      # Current progress count
            total_images,        # Total number of images
            progress_bars        # Progress bars to update
        )
            
        # Allow UI updates between chunks
        QtWidgets.QApplication.processEvents()
    
    return all_output_paths


def batch_process_with_mapping(images: list[Path],
                               output_paths: list[Path],
                               job: Job,
                               face_detection_tools: FaceToolPair,
                               cancel_event: threading.Event,
                               video: bool,
                               progress_bars: list[QtWidgets.QProgressBar] = None,
                               progress_count: int = 0,
                               chunk_size: int = 10) -> list[Path]:
    """
    Process a batch of images with custom output paths using the same pipeline with cancellation support.
    
    Args:
        images: List of image paths to process
        output_paths: List of output paths for processed images
        job: Job parameters
        face_detection_tools: Tools for face detection
        cancel_event: Event to check for cancellation requests
        video: bool
        progress_bars: Optional list of progress bars to update directly
        progress_count: int
        chunk_size: Number of images to process in each chunk
        
    Returns:
        List of output image paths that were successfully processed
    """
    if len(images) != len(output_paths):
        raise ValueError("Input and output path lists must have same length")

    pipeline = None
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
            def get_output_path_fn(image_path: Path, face_index: Optional[int]) -> Path:
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
        
        progress_count = invoke_progress_by_chunk(
            len(img_chunk),  # Size of the current chunk 
            progress_count,      # Current progress count
            total_images,        # Total number of images
            progress_bars        # Progress bars to update
        )
        
        # Allow UI to update between chunks AND process cancellation events
        QtWidgets.QApplication.processEvents()
        
        # Final cancellation check after UI updates
        if cancel_event.is_set():
            return all_output_paths
        
    return all_output_paths


def invoke_progress_by_chunk(chunk_size: int,
                             progress_count: int,
                             total_images: int,
                             progress_bars: list[QtWidgets.QProgressBar]) -> int:
    """
    Updates progress bars based on completed chunks rather than individual files.
    
    Args:
        chunk_size: Size of the current chunk
        progress_count: Current progress count before this chunk
        total_images: Total number of images to process
        progress_bars: List of progress bars to update
        
    Returns:
        Updated progress count
    """
    # Update progress count with entire chunk size
    progress_count += chunk_size
    
    # Make sure we don't exceed the total
    progress_count = min(progress_count, total_images)
    
    if progress_bars:
        percentage = min(100.0, (progress_count / total_images) * 100.0)
        value = int(10 * percentage)
        
        # Use safe thread approach to update UI
        for bar in progress_bars:
            # Use invokeMethod to update progress bars on the main thread
            QtCore.QMetaObject.invokeMethod(
                bar, 
                "setValue", 
                QtCore.Qt.ConnectionType.QueuedConnection,
                QtCore.Q_ARG(int, value)
            )
            
        # Process events for more responsive UI, but only after chunk completion
        QtWidgets.QApplication.processEvents()
            
    return progress_count


def process_batch_item(image_array: cv2.Mat,
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
    output_paths = []

    def batch_helper(_bounding_box: Box,  face_index: Optional[int]=None) -> None:
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
            reject(path=img_path, destination=job.destination)
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
            reject(path=img_path, destination=job.destination)
            return output_paths, pipeline

        batch_helper(bounding_box)
    
    return output_paths, pipeline


def get_output_path(input_path: Path,
                    destination: Path,
                    face_index: Optional[int],
                    radio_choice: str) -> Path:
    """Helper function to generate output paths."""
    suffix = input_path.suffix if radio_choice == 'No' else radio_choice
    if face_index is not None:
        # Multi-face output path
        stem = f"{input_path.stem}_{face_index}"
    else:
        # Single face output path
        stem = input_path.stem

    return destination / f"{stem}{suffix}"


@singledispatch
def crop_from_path(a0: Union[Path, str], *args, **kwargs) -> None:
    """Applies cropping to an image based on the job configuration."""
    raise NotImplementedError(f"Unsupported input type: {type(a0)}")


@crop_from_path.register
def _(image: Path,
      job: Job,
      face_detection_tools: FaceToolPair,
      new: Optional[Union[Path, str]] = None) -> None:
    crop_fn = crop_all_faces if job.multi_face_job else crop_single_face
    if all(x is not None for x in [job.table, job.folder_path, new]):
        save(image, job, face_detection_tools, crop_fn, new)
    elif job.folder_path is not None:
        save(image, job, face_detection_tools, crop_fn)
    else:
        save(image, job, face_detection_tools, crop_fn)


@crop_from_path.register
def _(image: str,
      job: Job,
      face_detection_tools: FaceToolPair,
      new: Optional[Union[Path, str]] = None) -> None:
    crop_from_path(Path(image), job, face_detection_tools, new)
