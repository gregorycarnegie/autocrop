import autocrop_rs
import contextlib
import cv2
import dlib
import rawpy
import shutil
import numpy as np
import pandas as pd
import tifffile as tiff
from files import IMAGE_TYPES, CV2_TYPES, RAW_TYPES, PANDAS_TYPES
from custom_widgets import ImageWidget
from functools import cache, lru_cache, wraps
from pathlib import Path
from PIL import Image
from PyQt6 import QtGui, QtWidgets
from typing import Optional, Union, NamedTuple, Generator
import cProfile
import pstats


GAMMA_THRESHOLD = 0.001

class Job(NamedTuple):
    width: QtWidgets.QLineEdit
    height: QtWidgets.QLineEdit
    fix_exposure_job: QtWidgets.QCheckBox
    multiface_job: QtWidgets.QCheckBox
    autotilt_job: QtWidgets.QCheckBox
    sensitivity: QtWidgets.QDial
    facepercent: QtWidgets.QDial
    gamma: QtWidgets.QDial
    top: QtWidgets.QDial
    bottom: QtWidgets.QDial
    left: QtWidgets.QDial
    right: QtWidgets.QDial
    radio_buttons: tuple[QtWidgets.QRadioButton, QtWidgets.QRadioButton, QtWidgets.QRadioButton,
                         QtWidgets.QRadioButton, QtWidgets.QRadioButton, QtWidgets.QRadioButton]
    radio_options: np.ndarray = np.array(['No', '.bmp', '.jpg', '.png', '.tiff', '.webp'])
    destination: Optional[QtWidgets.QLineEdit] = None
    photo_path: Optional[QtWidgets.QLineEdit] = None
    folder_path: Optional[QtWidgets.QLineEdit] = None
    video_path: Optional[QtWidgets.QLineEdit] = None
    start_position: Optional[float] = None
    stop_position: Optional[float] = None
    table: Optional[pd.DataFrame] = None
    column1: Optional[QtWidgets.QComboBox] = None
    column2: Optional[QtWidgets.QComboBox] = None

    def file_list(self):
        x = np.fromiter(Path(self.folder_path.text()).iterdir(), Path)
        y = np.array([pic.suffix.lower() in IMAGE_TYPES for pic in x])
        return x[y]
    
    def radio_choice(self) -> str:
        x = np.array([r.isChecked() for r in self.radio_buttons])
        return self.radio_options[x][0]
    
    def width_value(self) -> int:
        return int(self.width.text())
    
    def height_value(self) -> int:
        return int(self.height.text())
    
    def destination_path(self) -> Optional[Path]:
        if self.destination is None:
            return None
        x = Path(self.destination.text())
        x.mkdir(exist_ok=True)
        return x
    
    def file_list_to_numpy(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        if (
            not isinstance(self.table, pd.DataFrame)
            or not isinstance(self.column1, QtWidgets.QComboBox)
            or not isinstance(self.column2, QtWidgets.QComboBox)
        ):
            return None
        x = self.table[self.column1.currentText()].to_numpy().astype(str)
        y = self.table[self.column2.currentText()].to_numpy().astype(str)
        return x, y

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
    return cv2.dnn.readNetFromCaffe('resources\\weights\\deploy.prototxt.txt',
                                    'resources\\models\\res10_300x300_ssd_iter_140000.caffemodel')

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

def open_cv2(image: Path, exposure: bool, tilt: bool, face_detector = None,
             predictor = None) -> (cv2.Mat | np.ndarray):
    # if face_detector is None or predictor is None:
    #     return None
    img = cv2.imread(image.as_posix())
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = correct_exposure(img, exposure)
    return align_head(img, face_detector, predictor) if tilt else img

def open_raw(image: Path, exposure: bool, tilt: bool, face_detector = None,
             predictor = None) -> (cv2.Mat | np.ndarray):
    # if face_detector is None or predictor is None:
    #     return None
    with rawpy.imread(image.as_posix()) as raw:
        # Post-process the raw image data
        x = reorient_image_from_object(raw)
        x = correct_exposure(x, exposure)
        return align_head(x, face_detector, predictor) if tilt else x

def open_table(input_file: Path, extension: str) -> pd.DataFrame:
    return pd.read_csv(input_file) if extension == '.csv' else pd.read_excel(input_file)

@lru_cache(5, True)
def open_file(input_file: Union[Path, str], exposure: bool, tilt: bool, face_detector = None,
              predictor = None) -> Union[cv2.Mat, np.ndarray, pd.DataFrame, None]:
    """Given a filename, returns a numpy array or a pandas dataframe"""
    input_file = Path(input_file) if isinstance(input_file, str) else input_file
    if (extension := input_file.suffix.lower()) in CV2_TYPES:
        # Try with OpenCV
        return open_cv2(input_file, exposure, tilt, face_detector, predictor)
    elif extension in RAW_TYPES:
        # Try with rawpy
        return open_raw(input_file, exposure, tilt, face_detector, predictor)
    elif extension in PANDAS_TYPES:
        # Try pandas
        return open_table(input_file, extension)
    return None

@cache
def dlib_predictor(path: str) -> dlib.shape_predictor:
    return dlib.shape_predictor(path)

def align_head(image: Union[cv2.Mat, np.ndarray], face_detector, predictor) -> Union[cv2.Mat, np.ndarray]:
    """
    Aligns the head in an image using dlib and shape_predictor_68_face_landmarks.dat.

    Args:
        image: A numpy array representing the image.

    Returns:
        A numpy array representing the aligned image.
    """

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
    center = (sum(landmarks.part(i).x for i in range(36, 48)) / (12 * scaling_factor),
              sum(landmarks.part(i).y for i in range(36, 48)) / (12 * scaling_factor))
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
    left_eye = (sum(landmarks.part(i).x for i in range(42, 48)) / 6,
                sum(landmarks.part(i).y for i in range(42, 48)) / 6)
    right_eye = (sum(landmarks.part(i).x for i in range(36, 42)) / 6,
                 sum(landmarks.part(i).y for i in range(36, 42)) / 6)
    return np.arctan2(left_eye[1] - right_eye[1], left_eye[0] - right_eye[0]) * 180 / np.pi

def rotate_image(image: Union[cv2.Mat, np.ndarray], angle: float, center: tuple[float, float]) -> cv2.Mat:
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

def box(img: Union[cv2.Mat, np.ndarray], conf: int, face_perc: int, width: int, height: int, wide: int, high: int,
        top: int, bottom: int, left: int, right: int) -> Optional[tuple[int, int, int, int]]:
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


def multi_box(image: Union[cv2.Mat, np.ndarray], conf: int, face_perc: int, wide: int, high: int,
              top: int, bottom: int, left: int, right: int):
    height, width = image.shape[:2]
    # Create blob from image
    # We standardize the image by scaling it and then subtracting the mean RGB values
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)

    # Set the input for the neural network
    caffe = caffe_model()
    caffe.setInput(blob)

    # Forward pass through the network to get detections
    detections = caffe.forward()
    for i in range(detections.shape[2]):
        # Confidence in the detection
        if (confidence := detections[0, 0, i, 2]) > conf * .01: # Threshold
            # get the surrounding box coordinates and upscale them to original image
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            x0, y0, x1, y1 = box.astype("int")
            x0, y0, x1, y1 = autocrop_rs.crop_positions(x0, y0, x1 - x0, y1 - y0, face_perc, wide, high, top, bottom, left, right)

            text = "{:.2f}%".format(confidence * 100)
            y = y0 - 10 if y0 > 20 else y0 + 10
            
            cv2.rectangle(image, (x0, y0), (x1, y1),(0, 0, 255), 2)
            cv2.putText(image, text, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    
    return image

def multi_box_positions(image: Union[cv2.Mat, np.ndarray], conf: int, face_perc: int, wide: int, high: int,
                        top: int, bottom: int, left: int, right: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    # Create blob from image
    # We standardize the image by scaling it and then subtracting the mean RGB values
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)

    # Set the input for the neural network
    caffe = caffe_model()
    caffe.setInput(blob)

    # Forward pass through the network to get detections
    detections = caffe.forward()
    crop_positions = []
    for i in range(detections.shape[2]):
        # Confidence in the detection
        if detections[0, 0, i, 2] > conf * .01: # Threshold
            # get the surrounding box coordinates and upscale them to original image
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            x0, y0, x1, y1 = box.astype("int")

            # x0, y0, x1, y1 = autocrop_rs.crop_positions(x0, y0, x1 - x0, y1 - y0, face_perc, wide, high, top, bottom, left, right)
            # print(f'{x0}, {y0}, {x1}, {y1}')
            # np.append(crop_positions, (x0, y0, x1, y1))
            crop_positions.append(autocrop_rs.crop_positions(x0, y0, x1 - x0, y1 - y0, face_perc, wide, high, top, bottom, left, right))
    
    return detections, np.array(crop_positions)

@cache
def box_detect(img_path: Path, wide: int, high: int, conf: int, face_perc: int, top: int, bottom: int, left: int,
               right: int) -> Optional[tuple[int, int, int, int]]:
    img = open_file(img_path)
    # get width and height of the image
    try:
        if isinstance(img, np.ndarray):
            height, width = img.shape[:2]
        else:
            return None
    except AttributeError:
        return None
    return box(img, conf, face_perc, width, height, wide, high, top, bottom, left, right)

def box_detect_numpy(img: Union[cv2.Mat, np.ndarray], wide: int, high: int, conf: int, face_perc: int, top: int,
                     bottom: int, left: int, right: int) -> Optional[tuple[int, int, int, int]]:
    try:
        # get width and height of the image
        height, width = img.shape[:2]
    except AttributeError:
        return None
    return box(img, conf, face_perc, width, height, wide, high, top, bottom, left, right)

@cache
def get_first_file(img_path: Path) -> Optional[Path]:
    files = np.fromiter(img_path.iterdir(), Path)
    file = np.array([pic for pic in files if pic.suffix.lower() in IMAGE_TYPES])
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

# def reorient_image_by_exif(im: Image) -> Image:
#     try:
#         image_orientation = im.getexif()[274]
#         if image_orientation in {2, '2'}:
#             return im.transpose(Image.FLIP_LEFT_RIGHT)
#         elif image_orientation in {3, '3'}:
#             return im.transpose(Image.ROTATE_180)
#         elif image_orientation in {4, '4'}:
#             return im.transpose(Image.FLIP_TOP_BOTTOM)
#         elif image_orientation in {5, '5'}:
#             return im.transpose(Image.ROTATE_90).transpose(Image.FLIP_TOP_BOTTOM)
#         elif image_orientation in {6, '6'}:
#             return im.transpose(Image.ROTATE_270)
#         elif image_orientation in {7, '7'}:
#             return im.transpose(Image.ROTATE_270).transpose(Image.FLIP_TOP_BOTTOM)
#         elif image_orientation in {8, '8'}:
#             return im.transpose(Image.ROTATE_90)
#         else:
#             return im
#     except (KeyError, AttributeError, TypeError, IndexError):
#         return im

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

def mask_extensions(file_list: np.ndarray) -> tuple[np.ndarray, int]:
    # Get the extensions of the file names.
    extensions = np.char.lower([Path(file).suffix for file in file_list])
    # Create a mask that indicates which files have supported extensions.
    mask = np.in1d(extensions, IMAGE_TYPES)
    return mask, file_list[mask].size

def split_by_cpus(mask: np.ndarray, cpu_count: int, *file_lists: np.ndarray) -> Generator[list[np.ndarray], None, None]:
    """Split the file list and the mapping data into chunks."""
    return (np.array_split(file_list[mask], cpu_count) for file_list in file_lists)

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
