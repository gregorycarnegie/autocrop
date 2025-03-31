import cProfile
import collections.abc as c
import pstats
import random
import shutil
from functools import cache, wraps
from pathlib import Path
from typing import Any, Union
from typing import Optional

import autocrop_rs as rs
import cv2
import cv2.typing as cvt
import numpy as np
import numpy.typing as npt
import polars as pl
import tifffile as tiff
from rawpy._rawpy import NotSupportedError, LibRawError, LibRawFatalError, LibRawNonFatalError

from file_types import registry
from .face_tools import L_EYE_START, L_EYE_END, R_EYE_START, R_EYE_END, FaceToolPair
from .image_loader import ImageLoader
from .job import Job
from .operation_types import SaveFunction, Box

# Define constants
GAMMA_THRESHOLD = .001

CropFunction = c.Callable[
    [Union[cvt.MatLike, Path], Job, tuple[Any, Any]],
    Optional[Union[cvt.MatLike, c.Iterator[cvt.MatLike]]]
]


def profile_it(func: c.Callable[..., Any]) -> c.Callable[..., Any]:
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


def adjust_gamma(image: cvt.MatLike, gam: int) -> cvt.MatLike:
    """
    Adjusts image gamma using a precomputed lookup table.
    """

    return cv2.LUT(image, rs.gamma(gam * GAMMA_THRESHOLD))


def convert_color_space(image: cvt.MatLike) -> cvt.MatLike:
    """
    Converts the color space from BGR to RGB.
    """

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def numpy_array_crop(image: cvt.MatLike, bounding_box: Box) -> cvt.MatLike:
    x0, y0, x1, y1 = bounding_box
    h, w = image.shape[:2]

    # Calculate padding needed
    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - w)
    pad_bottom = max(0, y1 - h)

    # Source region in original image
    src_x0, src_y0 = max(0, x0), max(0, y0)
    src_x1, src_y1 = min(w, x1), min(h, y1)

    # Crop the valid region first
    cropped_valid = image[src_y0:src_y1, src_x0:src_x1]

    # Apply padding if needed
    if any(p > 0 for p in [pad_top, pad_bottom, pad_left, pad_right]):
        return cv2.copyMakeBorder(
            cropped_valid,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
    else:
        # No padding was required
        return cropped_valid

def correct_exposure(image: cvt.MatLike,
                     exposure: bool) -> cvt.MatLike:
    """
    Optionally corrects exposure by performing histogram-based scaling.
    """

    if not exposure:
        return image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim > 2 else image
    # Grayscale histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    # Calculate alpha and beta
    alpha, beta = rs.calc_alpha_beta(hist)
    return cv2.convertScaleAbs(src=image, alpha=alpha, beta=beta)


def format_image(image: cvt.MatLike) -> tuple[cvt.MatLike, float]:
    """
    Resizes an image to 256px height, returns grayscale if >2 channels, plus the scaling factor.
    """
    output_height = 256
    h, w = image.shape[:2]
    output_width, scaling_factor = rs.calculate_dimensions(h, w, output_height)
    
    image_array = cv2.resize(image, (output_width, output_height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY) if len(image_array.shape) >= 3 else image_array, scaling_factor


def align_head(image: cvt.MatLike,
               face_detection_tools: FaceToolPair,
               tilt: bool) -> cvt.MatLike:
    """
    Performs face alignment by detecting face, computing tilt, and rotating the image.
    
    Args:
        image: Input image
        face_detection_tools: Tuple of (detector, landmark predictor)
        tilt: Whether to perform tilt correction
        
    Returns:
        Aligned image or original if no face detected
    """
    if not tilt:
        return image

    detector, predictor = face_detection_tools

    # Optimize for smaller images for faster processing
    height, width = image.shape[:2]
    scale_factor = max(1, min(width, height) // 500)

    if scale_factor > 1:
        # Resize image for faster processing
        small_img = cv2.resize(image, (width // scale_factor, height // scale_factor))
        faces = detector(small_img)
    else:
        small_img = image
        faces = detector(image)

    if not faces:
        return image

    # Find face with highest confidence
    face = max(faces, key=lambda f: f.confidence)

    # Get facial landmarks
    landmarks = predictor(small_img, face)

    # Extract eye landmarks
    left_eye_landmarks = np.array([(landmarks.part(i).x, landmarks.part(i).y) 
                                 for i in range(L_EYE_START, L_EYE_END)])
    right_eye_landmarks = np.array([(landmarks.part(i).x, landmarks.part(i).y) 
                                  for i in range(R_EYE_START, R_EYE_END)])
# New Python code
    left_x = left_eye_landmarks[:, 0]  # First column (x coordinates)
    left_y = left_eye_landmarks[:, 1]  # Second column (y coordinates)
    right_x = right_eye_landmarks[:, 0]
    right_y = right_eye_landmarks[:, 1]

    center, angle = rs.get_rotation_matrix(left_x, left_y, right_x, right_y, scale_factor)
    center = int(center[0]), int(center[1])
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    return cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

def colour_expose_allign(image: cvt.MatLike, face_detection_tools: FaceToolPair, exposure:bool, tilt:bool) -> cvt.MatLike:
    # Convert BGR -> RGB for consistency
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = correct_exposure(image, exposure)
    return align_head(image, face_detection_tools, tilt)


def open_pic(file: Path,
             face_detection_tools: FaceToolPair,
             *,
             exposure: bool,
             tilt: bool) -> Optional[cvt.MatLike]:
    """
    Opens a non-RAW image using OpenCV, corrects exposure (optional), aligns head (optional).
    """

    img_path = file.as_posix()
    if registry.is_valid_type(file, "photo"):
        img = ImageLoader.loader('standard')(img_path)
        if img is None:
            return None

        return colour_expose_allign(img, face_detection_tools, exposure, tilt)

    elif registry.is_valid_type(file, "raw"):
        try:
            with ImageLoader.loader('raw')(img_path) as raw:
                try:
                    # Post-processing can also raise exceptions
                    img = raw.postprocess(use_camera_wb=True)

                    # Process the image further
                    if exposure:
                        img = correct_exposure(img, exposure)

                    return align_head(img, face_detection_tools, tilt)

                except (MemoryError, ValueError, TypeError) as e:
                    # Log more specific post-processing errors
                    print(f"Error post-processing RAW image {img_path}: {str(e)}")
                    return None

        except (NotSupportedError, LibRawFatalError, LibRawError, LibRawNonFatalError) as e:
            print(f"Error reading RAW file {img_path}: {str(e)}")
            return None

        except Exception as e:
            # Catch any other unexpected exceptions to ensure resources are released
            print(f"Unexpected error processing RAW file {img_path}: {str(e)}")
            return None
    else:
        return None


def open_animation(file: Path,
                   face_detection_tools: FaceToolPair,
                   *,
                   exposure: bool,
                   tilt: bool):
    
    retval, animation = cv2.imreadanimation(file.as_posix())
    if retval:
        return animation.frames
    else:
        return (colour_expose_allign(frame, face_detection_tools, exposure, tilt) for frame in animation.frames)


def open_table(file: Path) -> pl.DataFrame:
    """
    Opens a CSV or Excel file using Polars.
    """
    try:
        return pl.read_csv(file) if file.suffix.lower() == '.csv' else pl.read_excel(file)
    except IsADirectoryError:
        return pl.DataFrame()


def rs_crop_positions(box_outputs: npt.NDArray[np.int_],
                      job: Job) -> Box:
    """
    Wrapper around `rs.crop_positions` to compute final bounding box.
    """

    x0, y0, x1, y1 = box_outputs
    return rs.crop_positions(x0, y0, x1 - x0, y1 - y0, job.face_percent, job.width,
                             job.height, job.top, job.bottom, job.left, job.right)


def get_box_coordinates(output: Union[cvt.MatLike, npt.NDArray[np.generic]],
                        job: Job,
                        *,
                        width: int,
                        height: int,
                        x: Optional[npt.NDArray[np.generic]] = None) -> Box:
    """
    Vectorized version of box coordinate calculation.
    """
    scale = np.array([width, height, width, height])
    box_outputs = output * scale
    
    _box = box_outputs.astype(np.int_) if x is None else box_outputs[x.argmax()]
    
    return rs_crop_positions(_box, job)

def _draw_box_with_text(image: cvt.MatLike,
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
    font_scale, line_width, text_offset = .45, 2, 10

    text = f"{confidence:.2f}%"
    y_text = y0 - text_offset if y0 > 20 else y0 + text_offset
    cv2.rectangle(image, (x0, y0), (x1, y1), colour, line_width)
    cv2.putText(image, text, (x0, y_text), cv2.FONT_HERSHEY_SIMPLEX, font_scale, colour, line_width)
    return image

def multi_box(image: cvt.MatLike, job: Job, face_detection_tools: FaceToolPair) -> cvt.MatLike:
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
    confidences, boxes = multi_box_positions(image, job, face_detection_tools)

    # Adjust gamma and convert color space for visualization
    adjusted_image = adjust_gamma(image.copy(), job.gamma)
    rgb_image = convert_color_space(adjusted_image)

    # Draw rectangle and confidence text
    for confidence, box in zip(confidences, boxes):
        x0, y0, x1, y1 = box
        _draw_box_with_text(rgb_image, np.float64(confidence), x0=x0, y0=y0, x1=x1, y1=y1)

    return rgb_image


def multi_box_positions(image: cvt.MatLike, job: Job, face_detection_tools: FaceToolPair) -> tuple[list[float], list[Box]]:
    """
    Returns confidences and bounding boxes for all detected faces above threshold.
    
    Args:
        image: Input image
        job: Job parameters
        face_detection_tools: Tuple of (detector, landmark predictor)
        
    Returns:
        Tuple of (confidences, boxes)
    """
    detector, _ = face_detection_tools

    # Use our optimized detector
    faces = detector(image)

    # Filter by threshold (should already be done in detector, but double-check)
    faces = [face for face in faces if face.confidence * 100 >= job.threshold]

    if not faces:
        return [], []

    # Extract confidences
    confidences = [face.confidence * 100 for face in faces]

    # Generate bounding boxes
    boxes = []
    for face in faces:
        x0, y0 = face.left, face.top
        width, height = face.right - face.left, face.bottom - face.top

        if box := rs.crop_positions(
            x0,
            y0,
            width,
            height,
            job.face_percent,
            job.width,
            job.height,
            job.top,
            job.bottom,
            job.left,
            job.right,
        ):
            boxes.append(box)

    return confidences, boxes


def box_detect(image: cvt.MatLike, job: Job, face_detection_tools: FaceToolPair) -> Optional[Box]:
    """
    Detect face in an image with optimized performance.
    
    Args:
        image: Input image in OpenCV format
        job: Job configuration containing detection parameters
        face_detection_tools: Tuple of (detector, landmark predictor)
        
    Returns:
        Box coordinates if a face is detected, None otherwise
    """
    try:
        height, width = image.shape[:2]
        detector, _ = face_detection_tools
        
        # Determine if we need to resize for performance
        scale_factor = max(1, min(width, height) // 500)
        
        if scale_factor <= 1:
            # Small image, detect directly
            faces = detector(image)
        else:
            # Large image, resize for faster detection
            small_img = cv2.resize(image, (width // scale_factor, height // scale_factor))
            faces = detector(small_img)
        
        # If no faces or faces below threshold confidence, return None
        # The threshold check is now in the detector itself
        if not faces:
            return None
            
        # Find the face with highest confidence
        face = max(faces, key=lambda f: f.confidence)
        
        # For resized images, scale the coordinates back
        if scale_factor > 1:
            x0 = face.left * scale_factor
            y0 = face.top * scale_factor
            width = (face.right - face.left) * scale_factor
            height = (face.bottom - face.top) * scale_factor
        else:
            x0 = face.left
            y0 = face.top
            width = face.right - face.left
            height = face.bottom - face.top
        
        # Use the crop_positions function from Rust
        return rs.crop_positions(
            x0, y0, width, height, 
            job.face_percent, job.width, job.height,
            job.top, job.bottom, job.left, job.right
        )
    except (AttributeError, IndexError) as e:
        print(f"Error in box_detect: {e}")
        return None


def mask_extensions(file_list: npt.NDArray[np.str_], extentions: set[str]) -> tuple[npt.NDArray[np.bool_], int]:
    """
    Masks the file list based on supported extensions, returning the mask and its count.
    """
    if len(file_list) == 0:
        return np.array([], dtype=np.bool_), 0

    file_suffixes = np.array([Path(file).suffix.lower() for file in file_list])
    # x = registry.get_extensions("photo") | registry.get_extensions("raw")
    mask = np.isin(file_suffixes, list(extentions))
    return mask, np.count_nonzero(mask)


def split_by_cpus(mask: npt.NDArray[np.bool_],
                  core_count: int,
                  *file_lists: npt.NDArray[np.str_]) -> c.Iterator[list[npt.NDArray[np.str_]]]:
    """
    Splits the file list(s) based on a mask and the number of cores.
    """
    return map(lambda x: np.array_split(x[mask], core_count), file_lists)


def join_path_suffix(file_str: str, destination: Path) -> tuple[Path, bool]:
    """
    Joins a filename with a destination path and determines if it's a TIFF file.
    """

    path = destination.joinpath(file_str)
    return path, registry.should_use_tiff_save(path)


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
    if registry.is_valid_type(image_path, "raw"):
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


def save_image(image: cvt.MatLike,
               file_path: Path,
               user_gam: Union[int, float],
               is_tiff: bool = False) -> None:
    """
    Saves an image to disk. If TIFF, uses tifffile; otherwise uses OpenCV.
    """

    if is_tiff:
        # Only convert if actually needed
        if image.dtype != np.uint8:
            image = cv2.convertScaleAbs(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tiff.imwrite(file_path, image)
    else:
        lut = cv2.LUT(image, rs.gamma(user_gam * GAMMA_THRESHOLD))
        cv2.imwrite(file_path.as_posix(), lut)


def multi_save_image(images: c.Iterator[cvt.MatLike],
                     file_path: Path,
                     gamma_value: int,
                     is_tiff: bool) -> None:
    """
    Saves multiple cropped images, enumerating filenames by appending '_0', '_1', etc.
    """

    for i, img in enumerate(images):
        new_file_path = file_path.with_stem(f'{file_path.stem}_{i}')
        save_image(img, new_file_path, gamma_value, is_tiff)


def get_frame_path(destination: Path,
                   file_enum: str,
                   job: Job) -> tuple[Path, bool]:
    """
    Constructs a filename for saving frames from a video. 
    If radio_choice is 'original' (index 0), uses .jpg, otherwise uses the user-chosen extension.
    """

    file_str = f'{file_enum}.jpg' if job.radio_choice() == job.radio_options[0] else file_enum + job.radio_choice()
    return join_path_suffix(file_str, destination)


def save_detection(image: Path,
                   job: Job,
                   face_detection_tools: FaceToolPair,
                   crop_function: CropFunction,
                   save_function: SaveFunction,
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

    save_function(cropped_images, file_path, job.gamma, is_tiff)

def crop_image(image: Union[Path, cvt.MatLike],
               job: Job,
               face_detection_tools: FaceToolPair) -> Optional[cvt.MatLike]:
    """
    Single-face cropping function. Returns the cropped face if found and resizes to `job.size`.
    """

    pic_array = open_pic(image, face_detection_tools, exposure=job.fix_exposure_job,
                         tilt=job.auto_tilt_job) \
        if isinstance(image, Path) else image

    if pic_array is None:
        return None

    if (bounding_box := box_detect(pic_array, job, face_detection_tools)) is None:
        return None
    
    cropped_pic = numpy_array_crop(pic_array, bounding_box)
    result = convert_color_space(cropped_pic) if len(cropped_pic.shape) >= 3 else cropped_pic
    return cv2.resize(result, job.size, interpolation=cv2.INTER_AREA)


def process_image(image: cvt.MatLike,
                  job: Job,
                  crop_position: Box) -> cvt.MatLike:
    """
    Crops an image according to 'crop_position', converts color, and resizes to `job.size`.
    """

    cropped_image = numpy_array_crop(image, crop_position)
    result = convert_color_space(cropped_image) if len(cropped_image.shape) >= 3 else cropped_image
    return cv2.resize(result, job.size, interpolation=cv2.INTER_AREA)


def multi_crop(image: Union[cvt.MatLike, Path],
               job: Job,
               face_detection_tools: FaceToolPair) -> Optional[c.Iterator[cvt.MatLike]]:
    """
    Multi-face cropping function. Yields cropped faces above threshold, resized to `job.size`.
    """

    img = open_pic(image, face_detection_tools, exposure=job.fix_exposure_job, tilt=job.auto_tilt_job) \
        if isinstance(image, Path) else image

    if img is None:
        return None

    confidences, crop_positions = multi_box_positions(img, job, face_detection_tools)
    confidences = np.array(confidences) > job.threshold
    # Check if any faces were detected
    if np.any(confidences):
        # Cropped images
        return map(lambda pos: process_image(img, job, pos), crop_positions)
    else:
        return None


def get_crop_save_functions(job: Job) -> tuple[CropFunction, SaveFunction]:
    """
    Determines the correct crop and save functions based on whether multi-face detection is enabled.
    """

    return (multi_crop, multi_save_image) if job.multi_face_job else (crop_image, save_image)


def crop(image: Union[Path, str],
         job: Job,
         face_detection_tools: FaceToolPair,
         new: Optional[Union[Path, str]] = None) -> None:
    """
    High-level API for performing cropping and saving the result.
    """

    crop_fn, save_fn = get_crop_save_functions(job)

    # If working with table/folder logic
    if all(x is not None for x in [job.table, job.folder_path, new]):
        source = Path(image) if isinstance(image, str) else image
        save_detection(source, job, face_detection_tools, crop_fn, save_fn, new)
    elif job.folder_path is not None:
        source = job.folder_path / image.name
        save_detection(source, job, face_detection_tools, crop_fn, save_fn)
    else:
        save_detection(image, job, face_detection_tools, crop_fn, save_fn)


def frame_save(image: cvt.MatLike,
               file_enum_str: str,
               destination: Path,
               job: Job) -> None:
    """
    Saves a single frame from a video capture.
    """

    file_path, is_tiff = get_frame_path(destination, file_enum_str, job)
    save_image(image, file_path, job.gamma, is_tiff)
