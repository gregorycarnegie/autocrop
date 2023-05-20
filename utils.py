import autocrop_rs
import contextlib
import cv2
import dlib
import math
import rawpy
import shutil
import numpy as np
import pandas as pd
import tifffile as tiff
from files import PIL_TYPES, CV2_TYPES, RAW_TYPES, PANDAS_TYPES
from custom_widgets import ImageWidget
from functools import cache, lru_cache, wraps
from pathlib import Path
from PIL import Image
from PyQt6 import QtGui
from typing import Optional, Union
import cProfile
import pstats


GAMMA_THRESHOLD = 0.001

def profileit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with cProfile.Profile() as profile:
            func(*args, **kwargs)
        result = pstats.Stats(profile)
        result.sort_stats(pstats.SortKey.CUMULATIVE)
        result.print_stats()

    return wrapper

def caffe_model() -> cv2.dnn.Net:
    return cv2.dnn.readNetFromCaffe("resources\\weights\\deploy.prototxt.txt",
                                    "resources\\models\\res10_300x300_ssd_iter_140000.caffemodel")

@cache
def gamma(gam: Union[int, float] = 1.0) -> np.ndarray:
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

def adjust_gamma(image: np.ndarray, gam: int) -> cv2.Mat:
    return cv2.LUT(image, gamma(gam * GAMMA_THRESHOLD).astype('uint8'))

def convert_color_space(image: Union[cv2.Mat, np.ndarray]) -> cv2.Mat:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def display_image_on_widget(image: Union[cv2.Mat, np.ndarray], image_widget: ImageWidget) -> None:
    height, width, channel = image.shape
    bytes_per_line = channel * width
    q_image = QtGui.QImage(image.data, width, height, bytes_per_line, QtGui.QImage.Format.Format_BGR888)
    image_widget.setImage(QtGui.QPixmap.fromImage(q_image))

def correct_exposure(image: Union[cv2.Mat, Image.Image, np.ndarray],
                     exposure: Optional[bool] = False) -> Union[cv2.Mat, np.ndarray]:
    if not exposure:
        return np.array(image) if isinstance(image, Image.Image) else image

    image_array = np.array(image) if isinstance(image, Image.Image) else image
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY) if len(image_array.shape) > 2 else image_array

    # Grayscale histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()

    # Calculate alpha and beta
    alpha, beta = autocrop_rs.calc_alpha_beta(hist)
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

@lru_cache(5, True)
def open_file(input_file: Union[Path, str], exposure: Optional[bool] = False) -> Union[np.ndarray, pd.DataFrame, None]:
    """Given a filename, returns a numpy array or a pandas dataframe"""
    input_file = Path(input_file) if isinstance(input_file, str) else input_file
    if (extension := input_file.suffix.lower()) in CV2_TYPES:
        # Try with cv2
        with cv2.imread(input_file.as_posix()) as img:
            # Convert the RGB image data to a NumPy array
            return correct_exposure(img, exposure)
    elif extension in PIL_TYPES:
        # Try with PIL
        with Image.open(input_file).convert('RGB') as img_orig:
            x = np.array(img_orig)
            return correct_exposure(x, exposure)
    elif extension in RAW_TYPES:
        # Try with rawpy
        with rawpy.imread(input_file.as_posix()) as raw:
            # Post-process the raw image data
            rgb_image = raw.postprocess()
            x = np.array(rgb_image)
            return correct_exposure(x, exposure)
    elif extension in PANDAS_TYPES:
        return pd.read_csv(input_file) if extension == '.csv' else pd.read_excel(input_file)
    return None

def box(img: Union[cv2.Mat, np.ndarray], conf: int, face_perc: int, width: int, height: int, wide: int, high: int,
        top: int, bottom: int, left: int, right: int) -> Optional[tuple[int]]:
    # preprocess the image: resize and performs mean subtraction
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
    # set the image into the input of the neural network
    caffe = caffe_model()
    caffe.setInput(blob)
    # perform inference and get the result
    output = np.squeeze(caffe.forward())
    # get the confidence
    confidence_list = output[:, 2]
    if np.max(confidence_list) < conf * .01:
        return None
    # get the surrounding box coordinates and upscale them to original image
    box_coords = output[:, 3:7] * np.array([width, height, width, height])
    x0, y0, x1, y1 = box_coords[np.argmax(confidence_list)]
    return autocrop_rs.crop_positions(x0, y0, x1 - x0, y1 - y0, face_perc, wide, high, top, bottom, left, right)

@cache
def dlib_predictor(path: str) -> dlib.shape_predictor:
    return dlib.shape_predictor(path)

def align_head(image: Union[cv2.Mat, np.ndarray]) -> Union[cv2.Mat, np.ndarray]:
    """
    Aligns the head in an image using dlib and shape_predictor_68_face_landmarks.dat.

    Args:
        image: A numpy array representing the image.

    Returns:
        A numpy array representing the aligned image.
    """

    # Load the face detector and shape predictor.
    # TODO: This is very slow
    face_detector = dlib.get_frontal_face_detector()
    predictor = dlib_predictor('resources\\models\\shape_predictor_68_face_landmarks.dat')


    height, _ = image.shape[:2]
    scaling_factor = 256 / height
    image_array = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    # Detect the faces in the image.

    if len(image_array.shape) >=3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    
    faces = face_detector(image_array, 1)

    # If no faces are detected, return the original image.
    if len(faces) == 0:
        return image

    # Find the face with the highest confidence score.
    face = max(faces, key=lambda face: face.area())

    # Get the 68 facial landmarks for the face.
    landmarks = predictor(image_array, face)

    # Find the center of the face.
    center = (sum(landmarks.part(i).x for i in range(36, 48)) // (12 * scaling_factor),
              sum(landmarks.part(i).y for i in range(36, 48)) // (12 * scaling_factor))
    
    # Find the angle of the tilt.
    angle = get_angle_of_tilt(landmarks)

    return rotate_image(image, angle, center)


def get_angle_of_tilt(landmarks: dlib.full_object_detection) -> float:
    """
    Gets the angle of tilt of a face in an image using dlib.

    Args:
        landmarks: The 68 facial landmarks for the face.

    Returns:
        The angle of tilt in degrees.
    """

    # Find the eyes in the image.
    left_eye = (
        sum(landmarks.part(i).x for i in range(42, 48)) // 6,
        sum(landmarks.part(i).y for i in range(42, 48)) // 6,
    )
    right_eye = (
        sum(landmarks.part(i).x for i in range(36, 42)) // 6,
        sum(landmarks.part(i).y for i in range(36, 42)) // 6,
    )

    return math.atan2(left_eye[1] - right_eye[1], left_eye[0] - right_eye[0]) * 180 / math.pi


def rotate_image(image: Union[cv2.Mat, np.ndarray], angle: float, center: tuple) -> cv2.Mat:
    """
    Rotates an image by a specified angle around a center point.

    Args:
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


@cache
def box_detect(img_path: Path, wide: int, high: int, conf: int, face_perc: int, top: int, bottom: int, left: int,
               right: int) -> Optional[tuple[int]]:
    img = open_file(img_path)
    # TODO: Add support for face alignment
    img = align_head(img)
    # get width and height of the image
    try:
        if isinstance(img, np.ndarray):
            height, width = img.shape[:2]
        else:
            return None
    except AttributeError:
        return None
    return box(img, conf, face_perc, width, height, wide, high, top, bottom, left, right)

def box_detect_frame(img: cv2.Mat, wide: int, high: int, conf: int, face_perc: int, top: int, bottom: int, left: int,
                     right: int) -> Optional[tuple[int]]:
    # get width and height of the image
    # TODO: Add support for face alignment
    img = align_head(img)
    try:
        height, width = img.shape[:2]
    except AttributeError:
        return None
    return box(img, conf, face_perc, width, height, wide, high, top, bottom, left, right)

def get_first_file(img_path: Path, file_types: np.ndarray) -> Optional[Path]:
    files = np.fromiter(img_path.iterdir(), Path)
    file = np.array([pic for pic in files if pic.suffix.lower() in file_types])
    return file[0] if file.size > 0 else None

def reorient_image_by_exif(im: Image.Image, exif_orientation: int) -> np.ndarray:
    try:
        orientations = {2: Image.FLIP_LEFT_RIGHT,
                        3: Image.ROTATE_180,
                        4: Image.FLIP_TOP_BOTTOM,
                        5: Image.ROTATE_90 | Image.FLIP_TOP_BOTTOM,
                        6: Image.ROTATE_270,
                        7: Image.ROTATE_270 | Image.FLIP_TOP_BOTTOM,
                        8: Image.ROTATE_90}
        return np.array(im.transpose(orientations[exif_orientation]))
    except (KeyError, AttributeError, TypeError, IndexError):
        return np.array(im)

def reorient_image_from_object(im_obj: Union[Image.Image, rawpy.RawPy]) -> np.ndarray:
    with contextlib.suppress(KeyError, AttributeError, TypeError, IndexError):
        if isinstance(im_obj, Image.Image):
            exif_orientation = im_obj.getexif()[274]
            return reorient_image_by_exif(im_obj, exif_orientation)
        elif isinstance(im_obj, rawpy.RawPy):
            rgb_image = im_obj.postprocess(use_camera_wb=True)
            im = Image.fromarray(rgb_image.raw_image_visible)
            exif_orientation = im_obj.exif_data.get('Orientation', 1)
            return reorient_image_by_exif(im, exif_orientation)
    return np.array(im_obj if isinstance(im_obj, Image.Image) else im_obj.postprocess(use_camera_wb=True))

def preprocess_image(image: Union[Image.Image, rawpy.RawPy], bounding_box: tuple[int], checkbox: bool) -> Image.Image:
    pic = reorient_image_from_object(image)
    pic_array = correct_exposure(pic, checkbox)
    ################################
    pic_array = align_head(pic_array)

    return Image.fromarray(pic_array).crop(bounding_box)

@cache
def set_filename(image_path: Path, destination: Path, radio_choice: str, radio_options: tuple,
                 new: Optional[str] = None) -> tuple[Path, bool]:
    if image_path.suffix.lower() in RAW_TYPES:
        selected_extension = radio_options[2] if radio_choice == radio_options[0] else radio_choice
    else:
        selected_extension = image_path.suffix if radio_choice == radio_options[0] else radio_choice
    final_path = destination.joinpath(f'{new or image_path.stem}{selected_extension}')
    return final_path, final_path.suffix in {'.tif', '.tiff'}

def reject(path: Path, destination: Path, image: Path) -> None:
    reject_folder = destination.joinpath('rejects')
    reject_folder.mkdir(exist_ok=True)
    shutil.copy(path, reject_folder.joinpath(image.name))

def save_image(image: cv2.Mat, file_path: str, user_gam: Union[int, float], gamma_threshold: Union[int, float],
               is_tiff: bool = False) -> None:
    if is_tiff:
        image = cv2.convertScaleAbs(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tiff.imwrite(file_path, image)
    else:
        cv2.imwrite(file_path, cv2.LUT(image, gamma(user_gam * gamma_threshold)))
