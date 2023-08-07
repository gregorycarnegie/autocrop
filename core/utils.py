import cProfile
import pstats
import shutil
from functools import cache, lru_cache, wraps
from pathlib import Path
from typing import Any, Callable, List, Optional, Union, Generator, Tuple

import autocrop_rs
import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd
import rawpy
import tifffile as tiff
from PIL import Image
from PyQt6.QtGui import QImage, QPixmap

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
def gamma(gam: Union[int, float] = 1.0) -> Union[npt.NDArray[np.float_], npt.NDArray[np.uint8]]:
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

def adjust_gamma(image: Union[cv2.Mat, npt.NDArray[np.uint8]], gam: int) -> npt.NDArray[np.uint8]:
    return cv2.LUT(image, gamma(gam * GAMMA_THRESHOLD).astype('uint8'))

def convert_color_space(image: Union[cv2.Mat, npt.NDArray[np.uint8]]) -> cv2.Mat:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def display_image_on_widget(image: Union[cv2.Mat, npt.NDArray[np.uint8]], image_widget: ImageWidget) -> None:
    height, width, channel = image.shape
    bytes_per_line = channel * width
    q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_BGR888)
    image_widget.setImage(QPixmap.fromImage(q_image))

def correct_exposure(image: Union[cv2.Mat, Image.Image, npt.NDArray[np.uint8]],
                     exposure: bool) -> Union[cv2.Mat, npt.NDArray[np.uint8]]:
    """
    Adjust the exposure of an input image based on the alpha and beta values calculated from its grayscale histogram.

    Parameters:
        image (Union[cv2.Mat, Image.Image, np.ndarray]): The input image to correct. This can be an OpenCV matrix, a PIL image, or a numpy array.
        exposure (Optional[bool], default=False): A flag to indicate whether exposure correction should be performed. If False or not provided, the function simply returns the original image.

    Returns:
        (Union[cv2.Mat, np.ndarray]): The exposure corrected image if exposure=True, otherwise the original image. The returned image will be a numpy array if the input was a PIL Image, otherwise, the same type as the input image.
    """
    image_array = pillow_to_numpy(image) if isinstance(image, Image.Image) else image
    if not exposure:
        return image_array
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY) if len(image_array.shape) > 2 else image_array
    # Grayscale histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    # Calculate alpha and beta
    alpha, beta = autocrop_rs.calc_alpha_beta(hist)
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def open_image(image: Path,
               exposure: bool,
               tilt: bool,
               face_worker: FaceWorker) -> Union[cv2.Mat, npt.NDArray[np.uint8]]:
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
             face_worker: FaceWorker) -> Union[cv2.Mat, npt.NDArray[np.uint8]]:
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
             face_worker: FaceWorker) -> Optional[Union[cv2.Mat, npt.NDArray[np.uint8]]]:
    """Given a filename, returns a numpy array or a pandas dataframe"""
    input_file = Path(input_file) if isinstance(input_file, str) else input_file
    match input_file.suffix.lower():
        case extension if extension in Photo().CV2_TYPES:
            return open_image(input_file, exposure, tilt, face_worker)
        case extension if extension in Photo().RAW_TYPES:
            return open_raw(input_file, exposure, tilt, face_worker)
        case _: return None

def align_head(image: Union[cv2.Mat, npt.NDArray[np.uint8]],
               face_worker: FaceWorker,
               tilt: bool) -> Union[cv2.Mat, npt.NDArray[np.uint8]]:
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

def rotate_image(image: Union[cv2.Mat, npt.NDArray[np.uint8]],
                 angle: float,
                 center: Tuple[float, float]) -> Union[cv2.Mat, npt.NDArray[np.uint8]]:
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

def prepare_detections(image: Union[cv2.Mat, npt.NDArray[np.uint8]],
                       face_worker: FaceWorker) -> npt.NDArray[np.float_]:
    # Create blob from image
    # We standardize the image by scaling it and then subtracting the mean RGB values
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)
    # Set the input for the neural network
    caffe = face_worker.caffe_model()
    caffe.setInput(blob)
    # Forward pass through the network to get detections
    return caffe.forward()

def get_box_coordinates(output: npt.NDArray[np.float_],
                        width: int,
                        height: int,
                        job: Job,
                        x: Optional[npt.NDArray[np.float_]] = None) -> Tuple[int, int, int, int]:
    box_outputs = output * np.array([width, height, width, height])
    x0, y0, x1, y1 = box_outputs.astype(np.int_) if x is None else box_outputs[np.argmax(x)]
    return autocrop_rs.crop_positions(
        x0, y0, x1 - x0, y1 - y0, job.face_percent.value(), job.width_value(),
        job.height_value(), job.top.value(), job.bottom.value(), job.left.value(),
        job.right.value())

def box(img: Union[cv2.Mat, npt.NDArray[np.uint8]],
        width: int,
        height: int,
        job: Job,
        face_worker: FaceWorker,) -> Optional[Tuple[int, int, int, int]]:
    # preprocess the image: resize and performs mean subtraction
    detections = prepare_detections(img, face_worker)
    output = np.squeeze(detections)
    # get the confidence
    confidence_list = output[:, 2]
    if np.max(confidence_list) * 100 < job.sensitivity.value(): return None
    return get_box_coordinates(output[:, 3:7], width, height, job, confidence_list)

def multi_box(image: Union[cv2.Mat, npt.NDArray[np.uint8]],
              job: Job,
              face_worker: FaceWorker) -> Union[cv2.Mat, npt.NDArray[np.uint8]]:
    height, width = image.shape[:2]
    detections = prepare_detections(image, face_worker)
    for i in range(detections.shape[2]):
        # Confidence in the detection
        if (confidence := detections[0, 0, i, 2]) * 100 > job.sensitivity.value(): # Threshold
            x0, y0, x1, y1 = get_box_coordinates(detections[0, 0, i, 3:7], width, height, job)
            text = "{:.2f}%".format(confidence)
            y = y0 - 10 if y0 > 20 else y0 + 10
            cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 255), 2)
            cv2.putText(image, text, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    return image

def multi_box_positions(image: Union[cv2.Mat, npt.NDArray[np.uint8]],
                        job: Job,
                        face_worker: FaceWorker) -> Tuple[npt.NDArray[np.float_], npt.NDArray[np.int_]]:
    height, width = image.shape[:2]
    detections = prepare_detections(image, face_worker)
    crop_positions = [get_box_coordinates(detections[0, 0, i, 3:7], width, height, job)
                      for i in range(detections.shape[2])
                      if detections[0, 0, i, 2] * 100 > job.sensitivity.value()]
    return detections, np.array(crop_positions)

def box_detect(img: Union[cv2.Mat, npt.NDArray[np.uint8]],
               job: Job,
               face_worker: FaceWorker) -> Optional[Tuple[int, int, int, int]]:
    try:
        # get width and height of the image
        height, width = img.shape[:2]
    except AttributeError:
        return None
    return box(img, width, height, job, face_worker)

@cache
def get_first_file(img_path: Path) -> Optional[Path]:
    files = np.fromiter(img_path.iterdir(), Path)
    file = np.array([pic for pic in files if pic.suffix.lower() in Photo().file_types])
    return file[0] if file.size > 0 else None

def mask_extensions(file_list: npt.NDArray[np.str_]) -> Tuple[npt.NDArray[np.bool_], int]:
    """Get the extensions of the file names and Create a mask that indicates which files have supported extensions."""
    mask = np.in1d(np.char.lower([Path(file).suffix for file in file_list]), Photo().file_types)
    return mask, file_list[mask].size

def split_by_cpus(mask: npt.NDArray[np.bool_],
                  cpu_count: int,
                  *file_lists: npt.NDArray[np.str_]) -> Generator[List[npt.NDArray[np.str_]], None, None]:
    """Split the file list and the mapping data into chunks."""
    return (np.array_split(file_list[mask], cpu_count) for file_list in file_lists)

@cache
def set_filename(image_path: Path,
                 destination: Path,
                 radio_choice: str,
                 radio_options: Tuple[str, ...],
                 new: Optional[str] = None) -> Tuple[Path, bool]:
    if (suffix := image_path.suffix.lower()) in Photo().RAW_TYPES:
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

def save_image(image: Union[cv2.Mat, npt.NDArray[np.uint8]],
               file_path: str,
               user_gam: Union[int, float],
               is_tiff: bool = False) -> None:
    if is_tiff:
        image = cv2.convertScaleAbs(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tiff.imwrite(file_path, image)
    else:
        cv2.imwrite(file_path, cv2.LUT(image, gamma(user_gam * GAMMA_THRESHOLD)))
