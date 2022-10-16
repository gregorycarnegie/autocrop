import itertools
import os
from functools import wraps
from pathlib import Path
from time import perf_counter as pc

import numpy as np
from PIL import Image
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage
from cv2 import cvtColor, dnn, imread, COLOR_BGR2RGB, LUT

GAMMA, GAMMA_THRESHOLD = 0.90, 0.001
CV2_FILETYPES, PILLOW_FILETYPES = (np.array(['.bmp', '.dib', '.jp2', '.jpe', '.jpeg', '.jpg', '.pbm', '.pgm', '.png',
                                             '.ppm', '.ras', '.sr', '.tif', '.tiff', '.webp']),
                                   np.array(['.eps', '.icns', '.ico', '.im', '.msp', '.pcx', '.sgi', '.spi', '.xbm']))
COMBINED_FILETYPES = np.append(CV2_FILETYPES, PILLOW_FILETYPES)
FILE_FILTER, INPUT_FILETYPES = (np.array([f'*{file}' for file in COMBINED_FILETYPES]),
                                np.append(COMBINED_FILETYPES, [s.upper() for s in COMBINED_FILETYPES]))
file_type_list = f"All Files (*){''.join(f';;{_} Files (*{_})' for _ in np.sort(COMBINED_FILETYPES))}"
default_dir, proto_txt_path, caffe_model_path = (f'{Path.home()}\\Pictures', 'resources\\weights\\deploy.prototxt.txt',
                                                 'resources\\models\\res10_300x300_ssd_iter_140000.caffemodel')
caffe_model = dnn.readNetFromCaffe(proto_txt_path, caffe_model_path)


def func_speed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = None
        for i in range(1000):
            start = pc()
            result = func(*args, **kwargs)
            print(pc() - start)
        return result

    return wrapper


def intersect(v1, v2):
    a1, a2 = v1
    b1, b2 = v2
    da, db, dp = a2 - a1, b2 - b1, a1 - b1

    dap = np.empty_like(da)
    dap[0], dap[1] = -da[1], da[0]

    numerator, denominator = np.dot(dap, dp), np.dot(dap, db)

    if float(denominator) == 0.0:
        return numerator * 100 * db + b1
    else:
        return numerator / denominator * db + b1


def distance(pt1, pt2):
    """Returns the euclidean distance in 2D between 2 pts."""
    return np.linalg.norm(pt2 - pt1)


def gamma(gam=1.0):
    if gam != 1.0:
        return np.power(np.arange(256) / 255, 1.0 / gam) * 255
    else:
        return np.arange(256)


def open_file(input_filename):
    """Given a filename, returns a numpy array"""
    extension = os.path.splitext(input_filename)[1].lower()
    if extension in CV2_FILETYPES:
        """Try with cv2"""
        x = imread(input_filename)
        assert not isinstance(x, type(None)), 'image not found'
        return x
    if extension in PILLOW_FILETYPES:
        """Try with PIL"""
        with Image.open(input_filename) as img_orig:
            return np.fromfile(img_orig)
    return None


def _determine_safe_zoom(img_h, img_w, x, y, w, h, percent_face):
    """
    Determines the safest zoom level with which to add margins
    around the detected face. Tries to honor `self.face_percent`
    when possible.

    Parameters:
    -----------
    img_h: int | Height (px) of the image to be cropped
    img_w: int | Width (px) of the image to be cropped
    x: int | Leftmost coordinates of the detected face
    y: int | Bottom-most coordinates of the detected face
    w: int | Width of the detected face
    h: int | Height of the detected face
    """
    """Find out what zoom factor to use given self.aspect_ratio"""
    center = np.array([x + w * 0.5, y + h * 0.5])
    """Image corners"""
    im = np.zeros((5, 2))
    im[1:2, 1], im[2:3, 0] = img_h, img_w
    image_sides = np.dstack([im[:-1], im[1:]])
    corners = np.array(list(itertools.product((x, x + w), (y, y + h))))
    corner_vectors = np.hstack([np.repeat(center, 4, axis=0).reshape((2, 4)).T, corners]).reshape((4, 2, 2))
    A = np.array([distance(*vector) for vector in corner_vectors])
    intersects = np.array([[intersect(vector, side) for side in image_sides] for vector in corner_vectors])
    Z = (intersects >= 0) * (intersects <= im[2])
    X = np.array([distance(center, inter) for inter in intersects])
    result = np.append(percent_face, [1e4 * A if (X == 0)[i] and Z.all() else 1e2 * A / X[i] for i in range(len(X))])
    return np.amax(result)


def _crop_positions(img_h, img_w, x, y, w, h, percent_face, wide, high):
    """
    Returns the coordinates of the crop position centered
    around the detected face with extra margins. Tries to
    honor `self.face_percent` if possible, else uses the
    largest margins that comply with required aspect ratio
    given by `self.height` and `self.width`.

    Parameters:
    -----------
    img_h: int | Height (px) of the image to be cropped
    img_w: int | Width (px) of the image to be cropped
    x: int | Leftmost coordinates of the detected face
    y: int | Bottom-most coordinates of the detected face
    w: int | Width of the detected face
    h: int | Height of the detected face
    """

    """aspect: float | Aspect ratio"""
    aspect = wide / high
    """zoom: float | Zoom factor"""
    zoom = _determine_safe_zoom(img_h, img_w, x, y, w, h, percent_face)

    """Adjust output height based on percent"""
    if high >= wide:
        height_crop = h * 100.0 / zoom
        width_crop = aspect * height_crop
    else:
        width_crop = w * 100.0 / zoom
        height_crop = width_crop / aspect

    """Calculate padding by centering face"""
    x_pad, y_pad = (width_crop - w) * 0.5, (height_crop - h) * 0.5
    """Calc. positions of crop h1, h2, v1, v2"""
    return np.array([y - y_pad, y + h + y_pad, x - x_pad, x + w + x_pad]).astype(int)


def box_detect(img_path, wide, high, conf, face_perc):
    img = open_file(img_path) if isinstance(img_path, str) else img_path
    """get width and height of the image"""
    try:
        height, width = img.shape[:2]
    except AttributeError:
        return None

    """preprocess the image: resize and performs mean subtraction"""
    blob = dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
    """set the image into the input of the neural network"""
    caffe_model.setInput(blob)
    """perform inference and get the result"""
    output = np.squeeze(caffe_model.forward())
    conf *= 0.01
    """get the confidence"""
    confidence_list = output[:, 2]
    if np.max(confidence_list) < conf:
        return None
    """get the surrounding box coordinates and upscale them to original image"""
    box = output[:, 3:7] * np.array([width, height, width, height])
    x0, y0, x1, y1 = box[np.argmax(confidence_list)]
    return _crop_positions(height, width, x0, y0, x1 - x0, y1 - y0, face_perc, wide, high)


def display_crop(img_path, wide, high, conf, face_perc, label, gam):
    bounding_box = box_detect(img_path, wide, high, conf, face_perc)
    label.setScaledContents(False)
    """Save the cropped image with PIL if a face was detected"""
    if bounding_box is not None:
        """Open image and check exif orientation and rotate accordingly"""
        with Image.open(img_path) as img:
            pic = reorient_image(img)
        """crop picture"""
        try:
            cropped_pic = pic.crop((bounding_box[2], bounding_box[0], bounding_box[3], bounding_box[1]))
            cropped_pic = np.array(cropped_pic)
            table = gamma(gam * GAMMA_THRESHOLD).astype('uint8')
            cropped_pic = LUT(cropped_pic, table)

            pic_array = cvtColor(np.array(cropped_pic), COLOR_BGR2RGB)
            height, width, channel = pic_array.shape
            bytesPerLine = 3 * width

            qImg = QImage(pic_array.data, width, height, bytesPerLine, QImage.Format.Format_BGR888)
            pixmap = QPixmap.fromImage(qImg)
            label.setPixmap(pixmap.scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio))
        except AttributeError:
            pass


def reorient_image(im):
    try:
        image_exif = im.getexif()
        image_orientation = image_exif[274]
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
