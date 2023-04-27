import contextlib
import cv2
import rawpy
import re
import shutil
import time
import numpy as np
import pandas as pd
import tifffile as tiff
from pathlib import Path
from PIL import Image
from PyQt6 import QtCore, QtGui, QtWidgets
from typing import Optional, Union


class ImageWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = None

    def setImage(self, image: QtGui.QPixmap) -> None:
        self.image = image
        self.update()

    def paintEvent(self, event) -> None:
        if self.image is not None:
            qp = QtGui.QPainter(self)
            qp.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
            scaled_image = self.image.scaled(self.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                                             QtCore.Qt.TransformationMode.SmoothTransformation)
            x_offset = (self.width() - scaled_image.width()) // 2
            y_offset = (self.height() - scaled_image.height()) // 2
            qp.drawPixmap(x_offset, y_offset, scaled_image)


class DataFrameModel(QtCore.QAbstractTableModel):
    def __init__(self, df: pd.DataFrame, parent=None):
        super(DataFrameModel, self).__init__(parent)
        self._df = df

    def rowCount(self, parent=None):
        if self._df is not None:
            return self._df.shape[0]

    def columnCount(self, parent=None):
        if self._df is not None:
            return self._df.shape[1]

    def data(self, index, role=QtCore.Qt.ItemDataRole.DisplayRole) -> Optional[str]:
        if index.isValid() and role == QtCore.Qt.ItemDataRole.DisplayRole:
            return str(self._df.iloc[index.row(), index.column()])
        return None

    def headerData(self, section, orientation, role=QtCore.Qt.ItemDataRole.DisplayRole) -> Optional[str]:
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            if orientation == QtCore.Qt.Orientation.Horizontal:
                return str(self._df.columns[section])
            if orientation == QtCore.Qt.Orientation.Vertical:
                return str(self._df.index[section])
        return None
    
    
PIL_TYPES = np.array(['.bmp', '.dib', '.jfif', '.jp2', '.jpe', '.jpeg', '.jpg', '.pbm',
                      '.pgm', '.png', '.ppm', '.ras', '.sr', '.tif', '.tiff', '.webp'])
CV2_TYPES = np.array(['.eps', '.icns', '.ico', '.im', '.msp', '.pcx', '.sgi', '.spi', '.xbm'])
RAW_TYPES = np.array(['.dng', '.arw', '.cr2', '.crw', '.erf', '.kdc', '.nef', '.nrw', 
                      '.orf', '.pef', '.raf', '.raw', '.sr2', '.srw', '.x3f'])
PANDAS_TYPES = np.array(['.csv', ".xlsx", ".xlsm", ".xltx", ".xltm"])
VIDEO_TYPES = np.array([".avi", ".m4v", ".mkv", ".mov", ".mp4", ".wmv"])

GAMMA_THRESHOLD = 0.001

def timeit(func):
    """
    A decorator that times how long a function takes to execute.

    Args:
        func: The function to time.

    Returns:
        A wrapper function that times the execution of the decorated function.
    """

    def wrapper(*args, **kwargs):
        start_time = time.perf_counter_ns()
        result = func(*args, **kwargs)
        end_time = time.perf_counter_ns()
        print(f"{func.__name__} took {end_time - start_time} seconds to execute.")
        return result

    return wrapper

def caffe_model():
    return cv2.dnn.readNetFromCaffe("resources\\weights\\deploy.prototxt.txt",
                                    "resources\\models\\res10_300x300_ssd_iter_140000.caffemodel")

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

def correct_exposure(image: Union[cv2.Mat, np.ndarray], exposure: Optional[bool] = False) -> Union[cv2.Mat, np.ndarray]:
    if not exposure:
        return image
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Calculate histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    average_pixel_value = np.sum(hist * np.arange(256)) / np.sum(hist)
    return cv2.convertScaleAbs(image,
                               alpha=1.5 if average_pixel_value < 127 else 0.5,
                               beta=50 if average_pixel_value < 127 else -50)

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

def crop_positions(x: int, y: int, width: int, height: int, percent_face: int, output_width: int,
                   output_height: int, top: int, bottom: int, left: int, right: int) -> Optional[np.ndarray]:
    """
    Returns the coordinates of the crop position centered around the detected face with extra margins. Tries to honor
    `self.face_percent` if possible, else uses the largest margins that comply with required aspect ratio given by
    `self.height` and `self.width`.

    Parameters:
    -----------
    Args:
        x (int): The x-coordinate of the top-left corner of the face.
        y (int): The y-coordinate of the top-left corner of the face.
        width (int): The width of the face.
        height (int): The height of the face.
        output_width (int): The width of the output image
        output_height (int): The height of the output image
        percent_face (int): The percentage of the image that the face occupies.
    """
    if 0 < percent_face <= 100 and output_height > 0:
        if output_height >= output_width:
            cropped_height = height * 100.0 / percent_face
            cropped_width = output_width * cropped_height / output_height
        else:
            cropped_width = width * 100.0 / percent_face
            cropped_height = output_height * cropped_width / output_width
        
        # left, top, right, bottom
        l = x + (width - cropped_width) / 2 - left / 100 * cropped_width
        t = y + (height - cropped_height) / 2 - top / 100 * cropped_height
        r = x + (width + cropped_width) / 2 + right / 100 * cropped_width
        b = y + (height + cropped_height) / 2 + bottom / 100 * cropped_height

        return np.array([l, t, r, b]).astype(int)
    else:
        return None


def box_detect(img_path: Union[cv2.Mat, Path], wide: int, high: int, conf: int, face_perc: int,
               top: int, bottom: int, left: int, right: int) -> Optional[np.ndarray]:
    img = open_file(img_path) if isinstance(img_path, Path) else img_path
    # get width and height of the image
    try:
        height, width = img.shape[:2]
    except AttributeError:
        return None

    # preprocess the image: resize and performs mean subtraction
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
    # set the image into the input of the neural network
    caffe = caffe_model()
    caffe.setInput(blob)
    # perform inference and get the result
    output = np.squeeze(caffe.forward())
    # get the confidence
    confidence_list = output[:, 2]
    if np.max(confidence_list) < conf * 0.01:
        return None
    # get the surrounding box coordinates and upscale them to original image
    box_coords = output[:, 3:7] * np.array([width, height, width, height])
    
    x0, y0, x1, y1 = box_coords[np.argmax(confidence_list)]

    return crop_positions(x0, y0, x1 - x0, y1 - y0, face_perc, wide, high, top, bottom, left, right)

def get_first_file(img_path: Path, file_types: np.ndarray) -> Optional[Path]:
    files = np.fromiter(img_path.iterdir(), Path)
    file = np.array([pic for pic in files if pic.suffix.lower() in file_types])
    return file[0] if file.size > 0 else None

def display_crop(img_path: Path, wide: QtWidgets.QLineEdit, high: QtWidgets.QLineEdit, conf: QtWidgets.QDial,
                 face_perc: QtWidgets.QDial, gam: QtWidgets.QDial, top: QtWidgets.QDial, bottom: QtWidgets.QDial,
                 left: QtWidgets.QDial, right: QtWidgets.QDial, image_widget: ImageWidget,
                 file_types: Optional[np.ndarray] = None) -> None:
    if not img_path or img_path.as_posix() in {'', '.', None}:
        return None

    if img_path.is_dir():
        if not isinstance(file_types, np.ndarray):
            return None

        img_path = get_first_file(img_path, file_types)
        if img_path is None:
            return None

    bounding_box = box_detect(img_path, int(wide.text()), int(high.text()), conf.value(), face_perc.value(),
                              top.value(), bottom.value(), left.value(), right.value())
    if bounding_box is None:
        return None

    photo_path = img_path.as_posix()
    
    if (extention := img_path.suffix.lower()) in CV2_TYPES or extention in PIL_TYPES:
        with Image.open(photo_path) as img:
            pic = reorient_image_from_object(img)
            crop_and_set(pic, bounding_box, gam.value(), image_widget)
    elif extention in RAW_TYPES:
        with rawpy.imread(photo_path) as raw:
            pic = reorient_image_from_object(raw)
            crop_and_set(pic, bounding_box, gam.value(), image_widget)
    else:
        return None

def crop_and_set(pic: Image.Image, bounding_box: np.ndarray, gam: int, image_widget: ImageWidget) -> None:
    try:
        cropped_pic = np.array(pic.crop(bounding_box))
        cropped_pic = cv2.LUT(cropped_pic, gamma(gam * GAMMA_THRESHOLD).astype('uint8'))
        pic_array = cv2.cvtColor(np.array(cropped_pic), cv2.COLOR_BGR2RGB)
    except (cv2.error, Image.DecompressionBombError):
        return None
    # Convert numpy array to QImage
    height, width, channel = pic_array.shape
    bytes_per_line = channel * width
    qImg = QtGui.QImage(pic_array.data, width, height, bytes_per_line, QtGui.QImage.Format.Format_BGR888)
    # Set image to the image widget
    image_widget.setImage(QtGui.QPixmap.fromImage(qImg))

def reorient_image_by_exif(im: Image.Image, exif_orientation: int) -> Image.Image:
    try:
        orientations = {
            2: Image.FLIP_LEFT_RIGHT,
            3: Image.ROTATE_180,
            4: Image.FLIP_TOP_BOTTOM,
            5: Image.ROTATE_90 | Image.FLIP_TOP_BOTTOM,
            6: Image.ROTATE_270,
            7: Image.ROTATE_270 | Image.FLIP_TOP_BOTTOM,
            8: Image.ROTATE_90,
        }
        return im.transpose(orientations[exif_orientation])
    except (KeyError, AttributeError, TypeError, IndexError):
        return im

def reorient_image_from_object(im_obj: Union[Image.Image, rawpy.RawPy]) -> Image.Image:
    with contextlib.suppress(KeyError, AttributeError, TypeError, IndexError):
        if isinstance(im_obj, Image.Image):
            exif_orientation = im_obj.getexif()[274]
            return reorient_image_by_exif(im_obj, exif_orientation)
        elif isinstance(im_obj, rawpy.RawPy):
            rgb_image = im_obj.postprocess(use_camera_wb=True)
            im = Image.fromarray(rgb_image)
            exif_orientation = im_obj.exif_data.get('Orientation', 1)
            return reorient_image_by_exif(im, exif_orientation)
    return im_obj if isinstance(im_obj, Image.Image) else Image.fromarray(im_obj.postprocess(use_camera_wb=True))

def crop_image(image: Union[Path, np.ndarray], bounding_box: np.ndarray, width: int, height: int) -> cv2.Mat:
    if isinstance(image, Path):
        if image.suffix.lower() in RAW_TYPES:
            raw = rawpy.imread(image.as_posix())
            pic = reorient_image_from_object(raw)
            cropped_pic = pic.crop(bounding_box)
        else:
            photo = Image.open(image.as_posix())
            pic = reorient_image_from_object(photo)
            cropped_pic = pic.crop(bounding_box)
            pic.close()
    else:
        pic = Image.fromarray(image)
        cropped_pic = pic.crop(bounding_box)

    pic_array = np.array(cropped_pic)
    if isinstance(image, Path) and image.suffix.lower() in RAW_TYPES:
        pic_array = pic_array[:, :, ::-1]
    else:
        pic_array = cv2.cvtColor(pic_array, cv2.COLOR_BGR2RGB)
    
    return cv2.resize(pic_array, (width, height), interpolation=cv2.INTER_AREA)

def reject(path: Path, destination: Path, image: Path) -> None:
    reject_folder = destination.joinpath('rejects')
    reject_folder.mkdir(exist_ok=True)
    shutil.copy(path, reject_folder.joinpath(image.name))

def save_image(image: cv2.Mat, file_path: str, user_gam: Union[int, float], gamma_threshold: Union[int, float],
               is_tiff: bool = False) -> None:
    if is_tiff:
        array = np.array(image).astype(np.uint8)
        image = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        tiff.imwrite(file_path, image)
    else:
        cv2.imwrite(file_path, cv2.LUT(image, gamma(user_gam * gamma_threshold)))

def save_detection(path: Path, destination: Path, image: Path, width: int, height: int, confidence: int, face: int,
                   user_gam: int, top: int, bottom: int, left: int, right: int, radio: str, 
                   r_choices: np.ndarray, new: Optional[str] = None) -> None:
    """
    This code first checks if bounding_box is not None, and if so, it proceeds to crop the image and create the
    destination directory if it doesn't already exist. It then constructs the file name using a ternary expression
    that appends the file extension to the file name if radio is equal to the first element in radio_choices,
    and appends radio itself if radio is not equal to the first element in radio_choices. The code then constructs
    the file path by joining the destination directory and the file name and saves the cropped image to the file
    using the imwrite() function. If bounding_box is None, the code calls the reject() function to reject the image.
    """
    # Save the cropped image if a face was detected
    if (bounding_box := box_detect(path, width, height, confidence, face,
                                   top, bottom, left, right)) is not None:
        cropped_image = crop_image(path, bounding_box, width, height)
        destination.mkdir(exist_ok=True)
        if image.suffix.lower() in RAW_TYPES:
            file = f'{new or image.stem}{r_choices[2] if radio == r_choices[0] else radio}'
        else:
            file = f'{new or image.stem}{image.suffix if radio == r_choices[0] else radio}'
        
        file_path = destination.joinpath(file)
        is_tiff = file_path.suffix in {'.tif', '.tiff'}
        save_image(cropped_image, file_path.as_posix(), user_gam, GAMMA_THRESHOLD, is_tiff=is_tiff)
    else:
        reject(path, destination, image)

def crop_frame(frame, video_line_edit: QtWidgets.QLineEdit, destination_line_edit_4: QtWidgets.QLineEdit, wide: int,
               high: int, conf: QtWidgets.QDial, face_perc: QtWidgets.QDial, gamma_dial: QtWidgets.QDial,
               top: QtWidgets.QDial, bottom: QtWidgets.QDial, left: QtWidgets.QDial, right: QtWidgets.QDial,
               position_label: QtWidgets.QLabel, radio: str, radio_options: np.ndarray) -> None:
    if (bounding_box := box_detect(frame, wide, high, conf.value(), face_perc.value(),
                                   top.value(), bottom.value(), left.value(), right.value())) is not None:
        destination = Path(destination_line_edit_4.text())
        base_name = Path(video_line_edit.text()).stem

        cropped_image = crop_image(frame, bounding_box, wide, high)
        destination.mkdir(exist_ok=True)
        position = re.sub(':', '_', position_label.text())
        file_path = destination.joinpath(
            f'{base_name} - ({position}){radio_options[2] if radio == radio_options[0] else radio}')
        is_tiff = file_path.suffix in {'.tif', '.tiff'}
        save_image(cropped_image, file_path.as_posix(), gamma_dial.value(), GAMMA_THRESHOLD, is_tiff=is_tiff)
    else:
        return None

def frame_extraction(video, output_dir: QtWidgets.QLineEdit, frame_number: int, frame_step: int,
                     width: int, height: int, confidence: QtWidgets.QDial,
                     face: QtWidgets.QDial, user_gam: QtWidgets.QDial, 
                     top: QtWidgets.QDial, bottom: QtWidgets.QDial, left: QtWidgets.QDial, right: QtWidgets.QDial,
                     radio: str, radio_options: np.ndarray) -> None:
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = video.read()

    if not ret:
        return None

    destination = Path(output_dir.text())
    destination.mkdir(exist_ok=True)

    if (bounding_box := box_detect(frame, width, height, confidence.value(), face.value(),
                                   top.value(), bottom.value(), left.value(), right.value())) is not None:
        cropped_image = crop_image(frame, bounding_box, width, height)
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        file_enum = f'frame_{frame_number:06d}'
    else:
        cropped_image = frame
        file_enum = f'failed_frame_{frame_number:06d}'

    file_string = f'{file_enum}.jpg' if radio == radio_options[0] else file_enum + radio
    file_path = destination.joinpath(file_string)

    is_tiff = file_path.suffix in {'.tif', '.tiff'}
    save_image(cropped_image, file_path.as_posix(), user_gam.value(), GAMMA_THRESHOLD, is_tiff=is_tiff)
    frame_number += frame_step

def crop(image: Path, destination: Path, width: QtWidgets.QLineEdit, height: QtWidgets.QLineEdit, confidence: QtWidgets.QDial,
         face: QtWidgets.QDial, user_gam: QtWidgets.QDial, top: QtWidgets.QDial, bottom: QtWidgets.QDial, left: QtWidgets.QDial, right: QtWidgets.QDial, 
         radio: str, radio_choices: np.ndarray,
         line_edit: Optional[str] = None, source_folder: Optional[Path] = None, new: Optional[str] = None) -> None:
    if line_edit is None and isinstance(new, str) and isinstance(source_folder, Path):
        # Data cropping
        path = source_folder.joinpath(image)
        save_detection(path, destination, image, int(width.text()), int(height.text()), confidence.value(), face.value(), user_gam.value(), 
                       top.value(), bottom.value(), left.value(), right.value(),
                       radio, radio_choices, new)
    elif isinstance(line_edit, str):
        # Folder cropping
        source, image_name = Path(line_edit), image.name
        path = source.joinpath(image_name)
        save_detection(path, destination, Path(image_name), int(width.text()), int(height.text()), confidence.value(), face.value(), user_gam.value(),
                       top.value(), bottom.value(), left.value(), right.value(),
                       radio, radio_choices)
    elif image.is_file():
        # File cropping
        save_detection(image, destination, image, int(width.text()), int(height.text()), confidence.value(), face.value(), user_gam.value(), 
                       top.value(), bottom.value(), left.value(), right.value(),
                       radio, radio_choices, new)
