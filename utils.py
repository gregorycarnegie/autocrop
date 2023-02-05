import os
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage
from cv2 import cvtColor, dnn, imread, COLOR_BGR2RGB, LUT, imwrite, resize, INTER_AREA
from settings import FileTypeList, SpreadSheet, GAMMA_THRESHOLD, proto_path, caffe_path


def caffe_model():
    return dnn.readNetFromCaffe(proto_path, caffe_path)


def gamma(gam=1.0) -> np.ndarray:
    """
    The function calculates a gamma correction curve, which is a nonlinear transformation used to correct the
    brightness of an image. The gamma value passed in through the gam argument determines the shape of the correction
    curve. If the gam argument is not provided or is set to 1.0, the function simply returns an array containing the
    values 0 through 255.

    A gamma correction curve with a value greater than 1 will increase the contrast and make the dark regions of the
    image darker and the light regions of the image lighter. On the other hand, a gamma correction curve with a value
    less than 1 will decrease the contrast and make the dark and light regions of the image less distinct.
    """
    if gam != 1.0:
        return np.power(np.arange(256) / 255, 1.0 / gam) * 255
    else:
        return np.arange(256)


def open_file(input_filename: str) -> (None | np.ndarray | pd.DataFrame):
    """Given a filename, returns a numpy array or a pandas dataframe"""
    extension = os.path.splitext(input_filename)[1].lower()
    if extension in FileTypeList().list1:
        """Try with cv2"""
        x = imread(input_filename)
        assert not isinstance(x, type(None)), 'image not found'
        return x
    if extension in FileTypeList().list2:
        """Try with PIL"""
        with Image.open(input_filename) as img_orig:
            return np.fromfile(img_orig)
    if extension.lower() in SpreadSheet().list1:
        try:
            return pd.read_csv(input_filename)
        except FileNotFoundError:
            return None
    if extension.lower() in SpreadSheet().list2:
        try:
            return pd.read_excel(input_filename)
        except FileNotFoundError:
            return None
    return None


def crop_positions(x: int, y: int, w: int, h: int, percent_face: int, wide: int, high: int) -> np.ndarray:
    """
    Returns the coordinates of the crop position centered around the detected face with extra margins. Tries to honor
    `self.face_percent` if possible, else uses the largest margins that comply with required aspect ratio given by
    `self.height` and `self.width`.

    Parameters:
    -----------
    Args:
        x (int): The x-coordinate of the top-left corner of the face.
        y (int): The y-coordinate of the top-left corner of the face.
        w (int): The width of the face.
        h (int): The height of the face.
        wide (int): The width of the output image
        high (int): The height of the output image
        percent_face (int): The percentage of the image that the face occupies.
    """

    """aspect: float | Aspect ratio"""
    aspect = wide / high
    """Adjust output height based on percent"""
    if high >= wide:
        height_crop = h * 100.0 / percent_face
        width_crop = aspect * height_crop
    else:
        width_crop = w * 100.0 / percent_face
        height_crop = width_crop / aspect

    """Calculate padding by centering face"""
    x_pad, y_pad = (width_crop - w) * 0.5, (height_crop - h) * 0.5
    """Calc. positions of crop h1, h2, v1, v2"""
    return np.array([x - x_pad, y - y_pad, x + w + x_pad, y + h + y_pad]).astype(int)


def box_detect(img_path: str, wide: int, high: int, conf: int, face_perc: int) -> (None | np.ndarray):
    img = open_file(img_path) if isinstance(img_path, str) else img_path
    """get width and height of the image"""
    try:
        height, width = img.shape[:2]
    except AttributeError:
        return None

    """preprocess the image: resize and performs mean subtraction"""
    blob = dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
    """set the image into the input of the neural network"""
    caffe = caffe_model()
    caffe.setInput(blob)
    """perform inference and get the result"""
    output = np.squeeze(caffe.forward())
    """get the confidence"""
    confidence_list = output[:, 2]
    if np.max(confidence_list) < conf * 0.01:
        return None
    """get the surrounding box coordinates and upscale them to original image"""
    box_coords = output[:, 3:7] * np.array([width, height, width, height])
    x0, y0, x1, y1 = box_coords[np.argmax(confidence_list)]
    return crop_positions(x0, y0, x1 - x0, y1 - y0, face_perc, wide, high)


def display_crop(img_path: str, wide: int, high: int, conf: int, face_perc: int, label, gam: int):
    bounding_box = box_detect(img_path, wide, high, conf, face_perc)
    label.setScaledContents(False)
    """Save the cropped image with PIL if a face was detected"""
    if bounding_box is not None:
        """Open image and check exif orientation and rotate accordingly"""
        with Image.open(img_path) as img:
            pic = reorient_image(img)
            """crop picture"""
            cropped_pic = np.array(pic.crop(bounding_box))
            cropped_pic = LUT(cropped_pic, gamma(gam * GAMMA_THRESHOLD).astype('uint8'))

            pic_array = cvtColor(np.array(cropped_pic), COLOR_BGR2RGB)
            height, width = pic_array.shape[:2]

            qImg = QImage(pic_array.data, width, height, 3 * width, QImage.Format.Format_BGR888)
            pixmap = QPixmap.fromImage(qImg)
            label.setPixmap(pixmap.scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio))


def reorient_image(im: Image) -> Image:
    try:
        image_orientation = im.getexif()[274]
        if image_orientation in {2, '2'}:
            return im.transpose(Image.FLIP_LEFT_RIGHT)
        elif image_orientation in {3, '3'}:
            return im.transpose(Image.ROTATE_180)
        elif image_orientation in {4, '4'}:
            return im.transpose(Image.FLIP_TOP_BOTTOM)
        elif image_orientation in {5, '5'}:
            return im.transpose(Image.ROTATE_90).transpose(Image.FLIP_TOP_BOTTOM)
        elif image_orientation in {6, '6'}:
            return im.transpose(Image.ROTATE_270)
        elif image_orientation in {7, '7'}:
            return im.transpose(Image.ROTATE_270).transpose(Image.FLIP_TOP_BOTTOM)
        elif image_orientation in {8, '8'}:
            return im.transpose(Image.ROTATE_90)
        else:
            return im
    except (KeyError, AttributeError, TypeError, IndexError):
        return im


def crop_image(path: str, bounding_box, width, height):
    pic = reorient_image(Image.open(path))
    cropped_pic = pic.crop(bounding_box)
    pic_array = cvtColor(np.array(cropped_pic), COLOR_BGR2RGB)
    return resize(pic_array, (int(width), int(height)), interpolation=INTER_AREA)


def reject(path: str, destination: str, image: str):
    reject_folder = os.path.join(destination, 'reject')
    os.makedirs(reject_folder, exist_ok=True)
    shutil.copy(path, os.path.join(reject_folder, image))


def save_detection(path: str, destination: str, image: str, width: int, height: int, confidence: int, face: int,
                   user_gam: int, radio: str, r_choices: np.ndarray, new: (None | str)):
    """
    This code first checks if bounding_box is not None, and if so, it proceeds to crop the image and create the
    destination directory if it doesn't already exist. It then constructs the file name using a ternary expression
    that appends the file extension to the file name if radio is equal to the first element in radio_choices,
    and appends radio itself if radio is not equal to the first element in radio_choices. The code then constructs
    the file path by joining the destination directory and the file name and saves the cropped image to the file
    using the imwrite() function. If bounding_box is None, the code calls the reject() function to reject the image.
    """
    bounding_box = box_detect(path, width, height, confidence, face)
    """Save the cropped image with PIL if a face was detected"""
    if bounding_box is not None:
        cropped_image = crop_image(path, bounding_box, width, height)
        os.makedirs(destination, exist_ok=True)
        file = f'{new or os.path.splitext(image)[0]}{os.path.splitext(image)[1] if radio == r_choices[0] else radio}'
        file_path = os.path.join(destination, file)
        imwrite(file_path, LUT(cropped_image, gamma(user_gam * GAMMA_THRESHOLD)))
    else:
        reject(path, destination, image)


def crop(image: str, file_bool: bool, destination: str, width: int, height: int, confidence: int, face: int,
         user_gam: int, radio: str, n: int, lines: dict, radio_choices: np.ndarray):
    source, image = os.path.split(image) if file_bool else (lines[n].text(), image)
    path = os.path.join(source, image)
    save_detection(path, destination, image, width, height, confidence, face, user_gam, radio, radio_choices, None)


def m_crop(source_folder: str, image: str, new: str, destination: str, width: int, height: int, confidence: int,
           face: int, user_gam: int, radio: str, radio_choices: np.ndarray):
    path = os.path.join(source_folder, image)
    save_detection(path, destination, image, width, height, confidence, face, user_gam, radio, radio_choices, new)
    