import cProfile
import pstats
import shutil
from functools import cache, lru_cache, wraps
from pathlib import Path
from typing import Any, Callable, Generator, List, Optional, Tuple, Union

import autocrop_rs
import cv2
import cv2.typing as cvt
import numpy as np
import numpy.typing as npt
import pandas as pd
import rawpy
import tifffile as tiff
from PIL import Image

from . import window_functions as wf
from file_types import Photo
from .face_worker import FaceWorker
from .image_widget import ImageWidget
from .job import Job

# Define constants
GAMMA_THRESHOLD = .001
L_EYE_START, L_EYE_END = 42, 48
R_EYE_START, R_EYE_END = 36, 42

def profile_it(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any):
        with cProfile.Profile() as profile:
            func(*args, **kwargs)
        result = pstats.Stats(profile)
        result.sort_stats(pstats.SortKey.CUMULATIVE)
        result.print_stats()
    return wrapper

def pillow_to_numpy(image: Image.Image) -> npt.NDArray[np.uint8]:
    """Convert PIL image to numpy array"""
    return np.frombuffer(image.tobytes(), dtype=np.uint8).reshape((image.size[1], image.size[0], len(image.getbands())))

@cache
def gamma(gam: Union[int, float] = 1.0) -> npt.NDArray[np.generic]:
    """
    The function calculates a gamma correction curve, which is a nonlinear transformation used to correct the
    brightness of an image. The gamma value passed in through the gam argument determines the shape of the correction
    curve. If the gam argument is not provided or is set to 1.0, the function simply returns an array containing the
    values 0 through 255.

    A gamma correction curve with a value greater than 1 will increase the contrast and make the dark regions of the
    image darker and the light regions of the image lighter. On the other hand, a gamma correction curve with a value
    less than 1 will decrease the contrast and make the dark and light regions of the image less distinct.
    """
    return np.power(np.arange(256) / 255, 1.0 / gam) * 255 if gam != 1.0 else np.arange(256)

def adjust_gamma(image: Union[cvt.MatLike, npt.NDArray[np.uint8]], gam: int) -> cvt.MatLike:
    return cv2.LUT(image, gamma(gam * GAMMA_THRESHOLD).astype('uint8'))

def convert_color_space(image: Union[cvt.MatLike, npt.NDArray[np.uint8]]) -> cvt.MatLike:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def numpy_array_crop(image: cvt.MatLike, bounding_box: Tuple[int, int, int, int]) -> npt.NDArray[np.uint8]:
    # Load and crop image using PIL
    picture = Image.fromarray(image).crop(bounding_box)
    return pillow_to_numpy(picture)

def crop_and_set(image: cvt.MatLike,
                 bounding_box: Tuple[int, int, int, int],
                 gamma_value: int,
                 image_widget: ImageWidget) -> None:
    """
    Crop the given image using the bounding box, adjust its exposure and gamma, and set it to an image widget.

    Parameters:
        image: The input image as a numpy array.
        bounding_box: The bounding box coordinates to crop the image.
        gamma_value: The gamma value for gamma correction.
        image_widget: The image widget to display the processed image.
    
    Returns: None
    """
    try:
        cropped_image = numpy_array_crop(image, bounding_box)
        adjusted_image = adjust_gamma(cropped_image, gamma_value)
        final_image = convert_color_space(adjusted_image)
    except (cv2.error, Image.DecompressionBombError):
        return None
    wf.display_image_on_widget(final_image, image_widget)

def correct_exposure(image: Union[cvt.MatLike, npt.NDArray[np.uint8]],
                     exposure: bool) -> cvt.MatLike:
    """
    Adjust the exposure of an input image based on the alpha and beta values calculated from its grayscale histogram.

    Parameters:
        image (Union[cv2.Mat, Image.Image, np.ndarray]): The input image to correct. This can be an OpenCV matrix, a PIL image, or a numpy array.
        exposure (Optional[bool], default=False): A flag to indicate whether exposure correction should be performed. If False or not provided, the function simply returns the original image.

    Returns:
        (Union[cv2.Mat, np.ndarray]): The exposure corrected image if exposure=True, otherwise the original image. The returned image will be a numpy array if the input was a PIL Image, otherwise, the same type as the input image.
    """
    if not exposure: return image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
    # Grayscale histogram
    hist: npt.NDArray[np.generic] = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    # Calculate alpha and beta
    alpha, beta = autocrop_rs.calc_alpha_beta(hist)
    return cv2.convertScaleAbs(src=image, alpha=alpha, beta=beta)

def open_image(image: Path,
               exposure: bool,
               tilt: bool,
               face_worker: FaceWorker) -> cvt.MatLike:
    """
    Opens an image using OpenCV, performs exposure correction if needed, and optionally aligns the head in the image.

    Parameters:
        image (Path): The path to the image file to open.
        exposure (bool): A flag to indicate whether exposure correction should be performed.
        tilt (bool): A flag to indicate whether to align the head in the image.
        face_worker: tuple of face detector and predictor for current thread.

    Returns:
        (Union[cv2.Mat, np.ndarray]): The processed image. If head alignment is performed, the image will be aligned; otherwise, it is the original or exposure-corrected image.
    """
    img = cv2.imread(image.as_posix())
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = correct_exposure(img, exposure)
    return align_head(img, face_worker, tilt)

def open_raw(image: Path,
             exposure: bool,
             tilt: bool,
             face_worker: FaceWorker) -> cvt.MatLike:
    """
    Opens a raw image file, post-processes the raw image data, performs exposure correction if needed, and optionally aligns the head in the image.

    Parameters:
        image (Path): The path to the raw image file to open.
        exposure (bool): A flag to indicate whether exposure correction should be performed.
        tilt (bool): A flag to indicate whether to align the head in the image.
        face_worker: tuple of face detector and predictor for current thread.

    Returns:
        (Union[cv2.Mat, np.ndarray]): The processed image. If head alignment is performed, the image will be aligned; otherwise, it is the original or exposure-corrected image.
    """
    with rawpy.imread(image.as_posix()) as raw:
        # Post-process the raw image data
        img = raw.postprocess(use_camera_wb=True)
        img = correct_exposure(img, exposure)
        return align_head(img, face_worker, tilt)

@cache
def open_table(input_file: Path) -> pd.DataFrame:
    return pd.read_csv(input_file) if input_file.suffix.lower() == '.csv' else pd.read_excel(input_file)

@lru_cache(5, True)
def open_pic(input_file: Union[Path, str],
             exposure: bool,
             tilt: bool,
             face_worker: FaceWorker) -> Optional[cvt.MatLike]:
    """Given a filename, returns a numpy array or a pandas dataframe"""
    input_file = Path(input_file) if isinstance(input_file, str) else input_file
    match input_file.suffix.lower():
        case extension if extension in Photo.CV2_TYPES:
            return open_image(input_file, exposure, tilt, face_worker)
        case extension if extension in Photo.RAW_TYPES:
            return open_raw(input_file, exposure, tilt, face_worker)
        case _: return None

def align_head(image: Union[cvt.MatLike, npt.NDArray[np.uint8]],
               face_worker: FaceWorker,
               tilt: bool) -> cvt.MatLike:
    """
    Aligns the head in an image using dlib and shape_predictor_68_face_landmarks.dat.

    Parameters:
        image: A numpy array representing the image.
        face_worker: tuple of face detector and predictor for current thread.
        tilt: boolean value. determines weather or not to perform alignment

    Returns:
        A numpy array representing the aligned image.
    """
    if not tilt: return image
    height, _ = image.shape[:2]
    scaling_factor = 256 / height
    image_array = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    # Detect the faces in the image.
    if len(image_array.shape) >= 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    
    face_detector, predictor = face_worker.worker_tuple

    faces = face_detector(image_array, 1)
    # If no faces are detected, return the original image.
    if len(faces) == 0: return image

    # Find the face with the highest confidence score.
    face = max(faces, key=lambda x: x.area())
    # Get the 68 facial landmarks for the face.
    landmarks = predictor(image_array, face)
    landmarks_array = np.array([(p.x, p.y) for p in landmarks.parts()])
    # Find the angle of the tilt and the center of the face
    angle, center_x, center_y = get_angle_of_tilt(landmarks_array, scaling_factor)
    return rotate_image(image, angle, (center_x, center_y))

def get_angle_of_tilt(landmarks_array: npt.NDArray[np.int_], scaling_factor: float) -> Tuple[float, float, float]:
    """
    Gets the angle of tilt of a face in an image using dlib.

    Parameters:
        landmarks_array: The 68 facial landmarks for the face.
        scaling_factor: scaling factor

    Returns:
        The angle of tilt in degrees.
    """
    # Find the eyes in the image (l_eye - r_eye).
    eye_diff = np.mean(landmarks_array[L_EYE_START:L_EYE_END], axis=0) - \
               np.mean(landmarks_array[R_EYE_START:R_EYE_END], axis=0)
    # Find the center of the face.
    center_x, center_y = np.mean(landmarks_array[R_EYE_START:L_EYE_END], axis=0) / scaling_factor
    return np.arctan2(eye_diff[1], eye_diff[0]) * 180 / np.pi, center_x, center_y

def rotate_image(image: Union[cvt.MatLike, npt.NDArray[np.uint8]],
                 angle: float,
                 center: Tuple[float, float]) -> cvt.MatLike:
    """
    Rotates an image by a specified angle around a center point.

    Parameters:
        image: A numpy array representing the image.
        angle: The angle to rotate the image by in degrees.
        center: The point to rotate the image around.

    Returns:
        A numpy array representing the rotated image.
    """
    # Get the rotation matrix.
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    # Apply the rotation to the image.
    return cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

def prepare_detections(image: cvt.MatLike,
                       face_worker: FaceWorker) -> npt.NDArray[np.float_]:
    # Create blob from image
    # We standardize the image by scaling it and then subtracting the mean RGB values
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)
    # Set the input for the neural network
    caffe = face_worker.caffe_model()
    caffe.setInput(blob)
    # Forward pass through the network to get detections
    return np.array(caffe.forward())

def get_box_coordinates(output: Union[cvt.MatLike, npt.NDArray[np.generic]],
                        width: int,
                        height: int,
                        job: Job,
                        x: Optional[npt.NDArray[np.generic]] = None) -> Tuple[int, int, int, int]:
    box_outputs = output * np.array([width, height, width, height])
    x0, y0, x1, y1 = box_outputs.astype(np.int_) if x is None else box_outputs[np.argmax(x)]
    return autocrop_rs.crop_positions(x0, y0, x1 - x0, y1 - y0, job.face_percent, job.width,
                                      job.height, job.top, job.bottom, job.left, job.right)

def box(img: cvt.MatLike,
        width: int,
        height: int,
        job: Job,
        face_worker: FaceWorker,) -> Optional[Tuple[int, int, int, int]]:
    # preprocess the image: resize and performs mean subtraction
    detections = prepare_detections(img, face_worker)
    output = np.squeeze(detections)
    # get the confidence
    confidence_list = output[:, 2]
    if np.max(confidence_list) * 100 < 100 - job.sensitivity: return None
    return get_box_coordinates(output[:, 3:7], width, height, job, confidence_list)

def multi_box(image: cvt.MatLike,
              job: Job,
              face_worker: FaceWorker) -> cvt.MatLike:
    """
    Draw bounding boxes around detected faces in an image based on a given sensitivity threshold.
    
    Args:
    - image (cvt.MatLike): Input image with potential faces.
    - job (Job): Object containing detection sensitivity parameter.
    - face_worker (FaceWorker): Worker object responsible for face detection.
    
    Returns:
    - cvt.MatLike: Image with bounding boxes drawn around detected faces.
    """
    
    height, width = image.shape[:2]
    detections = prepare_detections(image, face_worker)
    threshold = 100 - job.sensitivity

    x = range(detections.shape[2])
    conf_array: Generator[np.float64 , None, None] = (detections[0, 0, i, 2] * 100 for i in x)
    output_array: Generator[npt.NDArray[np.float_] , None, None] = (detections[0, 0, i, 3:7] for i in x)

    for confidence, output in zip(conf_array, output_array):
        if confidence > threshold:
            x0, y0, x1, y1 = get_box_coordinates(output, width, height, job)
            image = _draw_box_with_text(image, x0, y0, x1, y1, confidence)
    return image

def _draw_box_with_text(image: cvt.MatLike, x0: int, y0: int, x1: int, y1: int, confidence: np.float64) -> cvt.MatLike:
    """
    Draw a bounding box and confidence text on an image.
    
    Args:
    - image (cvt.MatLike): Image on which to draw.
    - x0, y0, x1, y1 (int): Bounding box coordinates.
    - confidence (float): Detection confidence.
    
    Returns:
    - cvt.MatLike: Image with bounding box and text.
    """
    RED_COLOR = 0, 0, 255
    FONT_SCALE = .45
    LINE_WIDTH = 2
    TEXT_OFFSET = 10

    text = "{:.2f}%".format(confidence)
    y_text = y0 - TEXT_OFFSET if y0 > 20 else y0 + TEXT_OFFSET
    cv2.rectangle(image, (x0, y0), (x1, y1), RED_COLOR, LINE_WIDTH)
    cv2.putText(image, text, (x0, y_text), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, RED_COLOR, LINE_WIDTH)
    return image

def multi_box_positions(image: cvt.MatLike,
                        job: Job,
                        face_worker: FaceWorker) -> Tuple[npt.NDArray[np.float_], npt.NDArray[np.int_]]:
    height, width = image.shape[:2]
    detections = prepare_detections(image, face_worker)
    crop_positions = [get_box_coordinates(detections[0, 0, i, 3:7], width, height, job)
                      for i in range(detections.shape[2])
                      if detections[0, 0, i, 2] * 100 > 100 - job.sensitivity]
    return np.array(detections), np.array(crop_positions)

def box_detect(img: cvt.MatLike,
               job: Job,
               face_worker: FaceWorker) -> Optional[Tuple[int, int, int, int]]:
    try:
        # get width and height of the image
        height, width = img.shape[:2]
    except AttributeError:
        return None
    return box(img, width, height, job, face_worker)

def get_first_file(img_path: Path) -> Optional[Path]:
    files = np.fromiter(img_path.iterdir(), Path)
    file: Generator[Path, None, None] = (pic for pic in files if pic.suffix.lower() in Photo.file_types)
    return next(file, None)

def mask_extensions(file_list: npt.NDArray[np.str_]) -> Tuple[npt.NDArray[np.bool_], int]:
    """Get the extensions of the file names and Create a mask that indicates which files have supported extensions."""
    mask: npt.NDArray[np.bool_] = np.in1d(np.char.lower([Path(file).suffix for file in file_list]), Photo.file_types)
    return mask, file_list[mask].size

def split_by_cpus(mask: npt.NDArray[np.bool_],
                  core_count: int,
                  *file_lists: npt.NDArray[np.str_]) -> Generator[List[npt.NDArray[np.str_]], None, None]:
    """
    Splits each file list into chunks based on a boolean mask and the given CPU count.
    
    Args:
    - mask (np.ndarray): A boolean mask used to filter each file list.
    - core_count (int): Number of CPUs to determine the number of chunks.
    - *file_lists (np.ndarray): Multiple lists of files to be split.
    
    Returns:
    - Generator: Yields chunks of files for each file list.
    """
    return (np.array_split(file_list[mask], core_count) for file_list in file_lists)

@cache
def set_filename(image_path: Path,
                 destination: Path,
                 radio_choice: str,
                 radio_options: Tuple[str, ...],
                 new: Optional[str] = None) -> Tuple[Path, bool]:
    if (suffix := image_path.suffix.lower()) in Photo.RAW_TYPES:
        selected_extension = radio_options[2] if radio_choice == radio_options[0] else radio_choice
    else:
        selected_extension = suffix if radio_choice == radio_options[0] else radio_choice
    final_path = destination.joinpath(f'{new or image_path.stem}{selected_extension}')
    return final_path, final_path.suffix in {'.tif', '.tiff'}

def reject(path: Path,
           destination: Path,
           image: Path) -> None:
    reject_folder = destination.joinpath('rejects')
    reject_folder.mkdir(exist_ok=True)
    shutil.copy(path, reject_folder.joinpath(image.name))

def save_image(image: cvt.MatLike,
               file_path: Path,
               user_gam: Union[int, float],
               is_tiff: bool = False) -> None:
    if is_tiff:
        image = cv2.convertScaleAbs(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tiff.imwrite(file_path, image)
    else:
        cv2.imwrite(file_path.as_posix(), cv2.LUT(image, gamma(user_gam * GAMMA_THRESHOLD)))

def multi_save_image(cropped_images: List[cvt.MatLike],
                     file_path: Path,
                     gamma_value: int,
                     is_tiff: bool) -> None:
    for i, image in enumerate(cropped_images):
        new_file_path = file_path.with_stem(f'{file_path.stem}_{i}')
        save_image(image, new_file_path, gamma_value, is_tiff)

def get_frame_path(destination: Path,
                   file_enum: str,
                   job: Job) -> Tuple[Path, bool]:
    file_str = f'{file_enum}.jpg' if job.radio_choice() == job.radio_options[0] else file_enum + job.radio_choice()
    file_path = destination.joinpath(file_str)
    return file_path, file_path.suffix in {'.tif', '.tiff'}

def save_detection(source_image: Path,
                   job: Job,
                   face_worker: FaceWorker,
                   crop_function: Callable[[Union[cvt.MatLike, Path], Job, FaceWorker], Optional[Union[cvt.MatLike, Generator[cvt.MatLike, None, None]]]],
                   save_function: Callable[[Any, Path, int, bool], None],
                   image_name: Optional[Path] = None,
                   new: Optional[str] = None) -> None:
    if (destination_path := job.get_destination()) is None: return None

    image_name = source_image if image_name is None else image_name

    if (cropped_images := crop_function(source_image, job, face_worker)) is None:
        reject(source_image, destination_path, image_name)
        return None
    
    file_path, is_tiff = set_filename(image_name, destination_path, job.radio_choice(), job.radio_tuple(), new)
    
    save_function(cropped_images, file_path, job.gamma, is_tiff)

def crop_image(image: Union[Path, cvt.MatLike],
               job: Job,
               face_worker: FaceWorker) -> Optional[cvt.MatLike]:
    pic_array = open_pic(image, job.fix_exposure_job.isChecked(), job.auto_tilt_job.isChecked(), face_worker) \
        if isinstance(image, Path) else image
    if pic_array is None: return None
    if (bounding_box := box_detect(pic_array, job, face_worker)) is None: return None
    cropped_pic = numpy_array_crop(pic_array, bounding_box)
    result = convert_color_space(cropped_pic) if len(cropped_pic.shape) >= 3 else cropped_pic
    return cv2.resize(result, job.size, interpolation=cv2.INTER_AREA)

def multi_crop(source_image: Union[cvt.MatLike, Path],
               job: Job,
               face_worker: FaceWorker) -> Optional[Generator[cvt.MatLike, None, None]]:
    img = open_pic(source_image, job.fix_exposure_job.isChecked(), job.auto_tilt_job.isChecked(), face_worker) \
                    if isinstance(source_image, Path) else source_image
    if img is None:
        return None
    detections, crop_positions = multi_box_positions(img, job, face_worker)
    # Check if any faces were detected
    if not np.any(100 * detections[0, 0, :, 2] > (100 - job.sensitivity)): return None
    # Cropped images
    images = (Image.fromarray(img).crop(crop_position) for crop_position in crop_positions)
    # images as numpy arrays
    image_array: Generator[npt.NDArray[np.uint8], None, None] = (pillow_to_numpy(image) for image in images)
    # numpy arrays with collour space converted
    results: Generator[cvt.MatLike, None, None] = (convert_color_space(array) for array in image_array)
    # return resized results
    return (cv2.resize(src=result, dsize=job.size, interpolation=cv2.INTER_AREA) for result in results)

def crop(image: Path,
         job: Job,
         face_worker: FaceWorker,
         new: Optional[str] = None) -> None:
    if job.table is not None and job.folder_path is not None and new is not None:
        # Data cropping
        path = job.folder_path.joinpath(image)
        if job.multi_face_job.isChecked():
            save_detection(path, job, face_worker, multi_crop, multi_save_image, image, new)
        else:
            save_detection(path, job, face_worker, crop_image, save_image, image, new)
    elif job.folder_path is not None:
        # Folder cropping
        source, image_name = job.folder_path, image.name
        path = source.joinpath(image_name)
        if job.multi_face_job.isChecked():
            save_detection(path, job, face_worker, multi_crop, multi_save_image, Path(image_name))
        else:
            save_detection(path, job, face_worker, crop_image, save_image, Path(image_name))
    elif job.multi_face_job.isChecked():
        save_detection(image, job, face_worker, multi_crop, multi_save_image)
    else:
        save_detection(image, job, face_worker, crop_image, save_image)
