import cProfile
import collections.abc as c
import pstats
import random
import shutil
import threading
from functools import cache, wraps, singledispatch, partial
from pathlib import Path
from typing import Any, Union, Optional

import autocrop_rs as rs
import cv2
import cv2.typing as cvt
import numpy as np
import numpy.typing as npt
import polars as pl
from PyQt6 import QtWidgets
import tifffile as tiff
from rawpy._rawpy import NotSupportedError, LibRawError, LibRawFatalError, LibRawNonFatalError

from file_types import registry
from .face_tools import L_EYE_START, L_EYE_END, R_EYE_START, R_EYE_END, FaceToolPair
from .image_loader import ImageLoader
from .job import Job
from .operation_types import CropFunction, SaveFunction, Box


# Define constants
GAMMA_THRESHOLD = .001

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

def create_image_pipeline(job: Job, face_detection_tools: FaceToolPair) -> list[c.Callable[[cvt.MatLike], cvt.MatLike]]:
    """
    Creates a pipeline of image processing functions based on job parameters.
    """
    pipeline = []

    # Add alignment if requested
    if job.auto_tilt_job:
        pipeline.append(partial(align_head, face_detection_tools=face_detection_tools, job=job))

    # Add exposure correction if requested
    if job.fix_exposure_job:
        pipeline.append(partial(correct_exposure, exposure=True))

    # Always add gamma adjustment
    pipeline.append(partial(adjust_gamma, gam=job.gamma))

    # Add color space conversion if needed
    if job.radio_choice() in ['.jpg', '.png', '.bmp', '.webp', 'No']:
        pipeline.append(convert_color_space)

    return pipeline

def apply_pipeline(image: cvt.MatLike, pipeline: list[c.Callable[[cvt.MatLike], cvt.MatLike]]) -> cvt.MatLike:
    """
    Apply a sequence of image processing functions to an image.
    """
    result = image
    for func in pipeline:
        result = func(result)
    return result

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
    scale_factor = max(1, min(width, height) // 500)

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
    left_eye = np.ascontiguousarray(landmarks[L_EYE_START:L_EYE_END], dtype=np.float64)
    right_eye = np.ascontiguousarray(landmarks[R_EYE_START:R_EYE_END], dtype=np.float64)
    
    center, angle = rs.get_rotation_matrix(left_eye, right_eye, scale_factor)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Apply rotation to align the face
    return cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT
    )

def colour_expose_align(image: cvt.MatLike, face_detection_tools: FaceToolPair, job: Job) -> cvt.MatLike:
    # Convert BGR -> RGB for consistency
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) >= 3 else image
    return align_head(image, face_detection_tools, job)

def open_pic(file: Path,
             face_detection_tools: FaceToolPair,
             job: Job) -> Optional[cvt.MatLike]:
    """
    Opens a non-RAW image using OpenCV, corrects exposure (optional), aligns head (optional).
    """

    img_path = file.as_posix()
    if registry.is_valid_type(file, "photo") or registry.is_valid_type(file, "tiff"):
        img = ImageLoader.loader('standard')(img_path)
        if img is None:
            return None

        return colour_expose_align(img, face_detection_tools, job)

    elif registry.is_valid_type(file, "raw"):
        try:
            with ImageLoader.loader('raw')(img_path) as raw:
                try:
                    # Post-processing can also raise exceptions
                    img = raw.postprocess(use_camera_wb=True)

                    return align_head(img, face_detection_tools, job)

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

def open_table(file: Path) -> pl.DataFrame:
    """
    Opens a CSV or Excel file using Polars.
    """
    if registry.is_valid_type(file, "table"):
        try:
            return pl.read_csv(file) if file.suffix.lower() == '.csv' else pl.read_excel(file)
        except IsADirectoryError:
            return pl.DataFrame()
    return pl.DataFrame()

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
    results = multi_box_positions(image, job, face_detection_tools)

    # Adjust gamma and convert color space for visualization
    adjusted_image = adjust_gamma(image.copy(), job.gamma)
    rgb_image = convert_color_space(adjusted_image)

    # Draw rectangle and confidence text
    for confidence, box in results:
        x0, y0, x1, y1 = box
        _draw_box_with_text(rgb_image, np.float64(confidence), x0=x0, y0=y0, x1=x1, y1=y1)

    return rgb_image

def multi_box_positions(image: cvt.MatLike, job: Job, face_detection_tools: FaceToolPair) -> Optional[c.Iterator[tuple[float, Box]]]:
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
    faces = detector(image, job.threshold)

    if not faces:
        return None

    # Extract confidences
    confidences = [face.confidence * 100 for face in faces]

    # Generate bounding boxes
    boxes: list[Box] = []
    for face in faces:
        if box := rs.crop_positions(
            face.left,
            face.top,
            face.width,
            face.height,
            job.face_percent,
            job.width,
            job.height,
            job.top,
            job.bottom,
            job.left,
            job.right,
        ):
            boxes.append(box)

    return zip(confidences, boxes)

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
            faces = detector(image, job.threshold)
        else:
            # Large image, resize for faster detection
            small_img = cv2.resize(image, (width // scale_factor, height // scale_factor))
            faces = detector(small_img, job.threshold)
        
        # If no faces or faces below threshold confidence, return None
        # The threshold check is now in the detector itself
        if not faces:
            return None
            
        # Find the face with the highest confidence
        face = max(faces, key=lambda f: f.confidence)
        
        # For resized images, scale the coordinates back
        if scale_factor > 1:
            x0 = face.left * scale_factor
            y0 = face.top * scale_factor
            width = face.width * scale_factor
            height = face.height * scale_factor
        else:
            x0 = face.left
            y0 = face.top
            width = face.width
            height = face.height
        
        # Use the crop_positions function from Rust
        return rs.crop_positions(
            x0, y0, width, height, 
            job.face_percent, job.width, job.height,
            job.top, job.bottom, job.left, job.right
        )
    except (AttributeError, IndexError) as e:
        print(f"Error in box_detect: {e}")
        return None

def mask_extensions(file_list: npt.NDArray[np.str_], extensions: set[str]) -> tuple[npt.NDArray[np.bool_], int]:
    """
    Masks the file list based on supported extensions, returning the mask and its count.
    """
    if len(file_list) == 0:
        return np.array([], dtype=np.bool_), 0

    file_suffixes = np.array([Path(file).suffix.lower() for file in file_list])
    # x = registry.get_extensions("photo") | registry.get_extensions("raw")
    mask = np.isin(file_suffixes, list(extensions))
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

def process_image(image: cvt.MatLike, job: Job, bounding_box: Box, face_detection_tools: FaceToolPair) -> cvt.MatLike:
    """
    Crops an image according to 'bounding_box', applies processing pipeline, and resizes.
    """
    # Crop the image
    cropped_image = numpy_array_crop(image, bounding_box)
    
    # Create and apply the processing pipeline
    pipeline = create_image_pipeline(job, face_detection_tools)
    processed = apply_pipeline(cropped_image, pipeline)
    
    # Resize to final dimensions
    return cv2.resize(processed, job.size, interpolation=cv2.INTER_AREA)

@singledispatch
def crop_image(image: Union[cvt.MatLike, Path],
               job: Job,
               face_detection_tools: FaceToolPair) -> Optional[cvt.MatLike]:
    """
    Single-face cropping function. Returns the cropped face if found and resizes to `job.size`.
    """
    raise NotImplementedError(f"Unsupported input type: {type(image)}")

@crop_image.register
def _(image: cvt.MatLike, job: Job, face_detection_tools: FaceToolPair) -> Optional[cvt.MatLike]:
    if (bounding_box := box_detect(image, job, face_detection_tools)) is None:
        return None
    return process_image(image, job, bounding_box, face_detection_tools)

@crop_image.register
def _(image: Path, job: Job, face_detection_tools: FaceToolPair) -> Optional[cvt.MatLike]:
    pic_array = open_pic(image, face_detection_tools, job)
    if pic_array is None:
        return None
    return crop_image(pic_array, job, face_detection_tools)

@singledispatch
def multi_crop(image: Union[cvt.MatLike, Path],
               job: Job,
               face_detection_tools: FaceToolPair) -> Optional[c.Iterator[cvt.MatLike]]:
    """
    Multi-face cropping function. Yields cropped faces above threshold, resized to `job.size`.
    """
    raise TypeError(f"Unsupported input type: {type(image)}")

@multi_crop.register
def _(image: cvt.MatLike, job: Job, face_detection_tools: FaceToolPair) -> Optional[c.Iterator[cvt.MatLike]]:
    """
    Optimized multi-face cropping function using the pipeline approach.
    Yields cropped faces above threshold, resized to `job.size`.
    """
    # Step 1: Detect faces and get bounding boxes
    results = multi_box_positions(image, job, face_detection_tools)

    if results is None:
        return None
    
    # Step 2: Filter faces based on confidence threshold
    valid_faces = [
        (confidence, position) for confidence, position in results
        if confidence > job.threshold
    ]
    
    # If no valid faces were found, return None
    if not valid_faces:
        return None
    
    # Step 3: Create the processing pipeline once (reused for all faces)
    pipeline = create_image_pipeline(job, face_detection_tools)
    
    # Step 4: Process each face
    def process_face_box(confidence: float, position: Box) -> cvt.MatLike:
        # Crop the face
        cropped = numpy_array_crop(image, position)
        
        # Apply the processing pipeline
        processed = apply_pipeline(cropped, pipeline)
        
        # Resize to final dimensions
        return cv2.resize(processed, job.size, interpolation=cv2.INTER_AREA)
    
    # Return a generator that processes each face on-demand
    return (process_face_box(confidence, position) for confidence, position in valid_faces)

@multi_crop.register
def _(image: Path, job: Job, face_detection_tools: FaceToolPair) -> Optional[c.Iterator[cvt.MatLike]]:
    img = open_pic(image, face_detection_tools, job)
    return None if img is None else multi_crop(img, job, face_detection_tools)

def process_face(image_array: cvt.MatLike, bounding_box: Box, pipeline: list, job: Job) -> cvt.MatLike:
    """Process a single face with the given pipeline"""
    cropped = numpy_array_crop(image_array, bounding_box)
    processed = apply_pipeline(cropped, pipeline)
    return cv2.resize(processed, job.size, interpolation=cv2.INTER_AREA)

def save_processed_face(processed_image: cvt.MatLike, output_path: Path, gamma: int) -> None:
    """Save a processed face to the given output path"""
    is_tiff = output_path.suffix.lower() in {'.tif', '.tiff'}
    save_image(processed_image, output_path, gamma, is_tiff=is_tiff)

def process_batch_item(image_array: cvt.MatLike, job: Job, face_detection_tools: FaceToolPair, 
                      pipeline: list, progress_callback: c.Callable, img_path: Path,
                      get_output_path_fn: c.Callable) -> tuple[list[Path], list]:
    """
    Process a single image from a batch with the given pipeline.
    Returns a tuple of (output_paths, pipeline)
    """
    output_paths = []
    
    # Process based on multi-face setting
    if job.multi_face_job:
        # Multi-face processing
        results = multi_box_positions(image_array, job, face_detection_tools)

        if not results:
            reject(path=img_path, destination=job.destination)
            progress_callback()
            return output_paths, pipeline

        valid_positions = [pos for confidence, pos in results if confidence > job.threshold]

        if not valid_positions:
            progress_callback()
            return output_paths, pipeline

        # Create pipeline if not already created
        if pipeline is None:
            pipeline = create_image_pipeline(job, face_detection_tools)

        # Process each face
        for i, bounding_box in enumerate(valid_positions):
            # Create output path using the provided function
            output_path = get_output_path_fn(img_path, i)
            
            # Process and save
            processed = process_face(image_array, bounding_box, pipeline, job)
            save_processed_face(processed, output_path, job.gamma)
            output_paths.append(output_path)
    else:
        # Single face processing
        if (bounding_box := box_detect(image_array, job, face_detection_tools)) is None:
            reject(path=img_path, destination=job.destination)
            progress_callback()
            return output_paths, pipeline

        # Create pipeline if not already created
        if pipeline is None:
            pipeline = create_image_pipeline(job, face_detection_tools)

        # Create output path using the provided function
        output_path = get_output_path_fn(img_path, None)
        
        # Process and save
        processed = process_face(image_array, bounding_box, pipeline, job)
        save_processed_face(processed, output_path, job.gamma)
        output_paths.append(output_path)
    
    progress_callback()
    return output_paths, pipeline

def batch_process_with_pipeline(images: list[Path], job: Job, face_detection_tools: FaceToolPair, 
                               progress_callback: c.Callable, cancel_event: threading.Event,
                               chunk_size: int = 10) -> list[Path]:
    """
    Process a batch of images with the same pipeline for efficiency with cancellation support.
    
    Args:
        images: List of image paths to process
        job: Job parameters
        face_detection_tools: Tools for face detection
        progress_callback: Callback to update progress
        cancel_event: Event to check for cancellation requests
        chunk_size: Number of images to process in each chunk
        
    Returns:
        List of output image paths
    """
    pipeline = None
    all_output_paths = []
    total_images = len(images)
    
    # Process images in smaller chunks to maintain UI responsiveness
    for i in range(0, total_images, chunk_size):
        # Check for cancellation BEFORE processing chunk
        if cancel_event.is_set():
            break
            
        # Get current chunk
        chunk = images[i:min(i + chunk_size, total_images)]
        
        # Process each image in the chunk with cancellation checks
        for img_path in chunk:
            # Check for cancellation BEFORE processing each image
            if cancel_event.is_set():
                break
                
            # Open the image
            image_array = open_pic(img_path, face_detection_tools, job)
            if image_array is None:
                progress_callback()
                continue

            # Create a function to get output paths for standard batch processing
            def get_output_path_fn(image_path: Path, face_index: Optional[int]) -> Path:
                return get_output_path(image_path, job.destination, face_index, job.radio_choice())

            # Process the image
            output_paths, pipeline = process_batch_item(
                image_array, job, face_detection_tools, pipeline, 
                progress_callback, img_path, get_output_path_fn
            )
            
            all_output_paths.extend(output_paths)
            
            # Check for cancellation AFTER processing each image
            if cancel_event.is_set():
                break
        
        # Allow UI to update between chunks AND process cancellation events
        QtWidgets.QApplication.processEvents()
        
        # Final cancellation check after UI updates
        if cancel_event.is_set():
            break
        
    return all_output_paths

def batch_process_with_mapping(images: list[Path], output_paths: list[Path], job: Job, 
                              face_detection_tools: FaceToolPair, progress_callback: c.Callable,
                              cancel_event: threading.Event, chunk_size: int = 10) -> list[Path]:
    """
    Process a batch of images with custom output paths using the same pipeline with cancellation support.
    
    Args:
        images: List of image paths to process
        output_paths: List of output paths for processed images
        job: Job parameters
        face_detection_tools: Tools for face detection
        progress_callback: Callback to update progress
        cancel_event: Event to check for cancellation requests
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
            break
            
        # Get current chunk
        chunk_end = min(i + chunk_size, total_images)
        img_chunk = images[i:chunk_end]
        out_chunk = output_paths[i:chunk_end]
        
        # Process each image in the chunk with cancellation checks
        for img_path, out_path in zip(img_chunk, out_chunk):
            # Check for cancellation BEFORE processing each image
            if cancel_event.is_set():
                break
                
            # Open the image
            image_array = open_pic(img_path, face_detection_tools, job)
            if image_array is None:
                progress_callback()
                continue

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
                progress_callback, img_path, get_output_path_fn
            )
            
            all_output_paths.extend(output_paths_result)
            
            # Check for cancellation AFTER processing each image
            if cancel_event.is_set():
                break
        
        # Allow UI to update between chunks AND process cancellation events
        QtWidgets.QApplication.processEvents()
        
        # Final cancellation check after UI updates
        if cancel_event.is_set():
            break
        
    return all_output_paths

def get_output_path(input_path: Path, destination: Path, face_index: Optional[int], radio_choice: str) -> Path:
    """Helper function to generate output paths."""
    suffix = input_path.suffix if radio_choice == 'No' else radio_choice
    if face_index is not None:
        # Multi-face output path
        stem = f"{input_path.stem}_{face_index}"
    else:
        # Single face output path
        stem = input_path.stem

    return destination / f"{stem}{suffix}"

def get_crop_save_functions(job: Job) -> tuple[CropFunction, SaveFunction]:
    """
    Determines the correct crop and save functions based on whether multi-face detection is enabled.
    """
    return (multi_crop, multi_save_image) if job.multi_face_job else (crop_image, save_image)

@singledispatch
def crop(image: Union[Path, str],
         job: Job,
         face_detection_tools: FaceToolPair,
         new: Optional[Union[Path, str]] = None) -> None:
    """Applies cropping to an image based on the job configuration."""
    raise NotImplementedError(f"Unsupported input type: {type(image)}")

@crop.register
def _(image: Path, job: Job, face_detection_tools: FaceToolPair, new: Optional[Union[Path, str]] = None) -> None:
    crop_fn, save_fn = get_crop_save_functions(job)
    if all(x is not None for x in [job.table, job.folder_path, new]):
        save_detection(image, job, face_detection_tools, crop_fn, save_fn, new)
    elif job.folder_path is not None:
        save_detection(image, job, face_detection_tools, crop_fn, save_fn)
    else:
        save_detection(image, job, face_detection_tools, crop_fn, save_fn)

@crop.register
def _(image: str, job: Job, face_detection_tools: FaceToolPair, new: Optional[Union[Path, str]] = None) -> None:
    crop(Path(image), job, face_detection_tools, new)

def frame_save(image: cvt.MatLike,
               file_enum_str: str,
               destination: Path,
               job: Job) -> None:
    """
    Saves a single frame from a video capture.
    """

    file_path, is_tiff = get_frame_path(destination, file_enum_str, job)
    save_image(image, file_path, job.gamma, is_tiff)
