import itertools
import os
import numpy as np

from pathlib import Path
from PIL import Image
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage
from cv2 import cvtColor, dnn, imread, waitKey, COLOR_BGR2RGB, LUT

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


def intersect(v1, v2):
    a1, a2 = v1
    b1, b2 = v2
    da, db, dp = a2 - a1, b2 - b1, a1 - b1

    dap = np.empty_like(da)
    dap[0], dap[1] = -da[1], da[0]

    denominator = np.dot(dap, db).astype(float)
    num = np.dot(dap, dp)

    if float(denominator) == 0.0:
        return num / 0.01 * db + b1
    else:
        return num / denominator * db + b1


def distance(pt1, pt2):
    """Returns the euclidean distance in 2D between 2 pts."""
    return np.linalg.norm(pt2 - pt1)


def gamma(gam=1.0):
    if gam != 1.0:
        inv_gamma = 1.0 / gam
        return np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype('uint8')
    else:
        return np.arange(0, 256).astype('uint8')
    # return LUT(img, table)


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
            return np.asarray(img_orig)
    return None


def _determine_safe_zoom(img_h, img_w, x, y, w, h, percent_face):
    """
    Determines the safest zoom level with which to add margins
    around the detected face. Tries to honor `self.face_percent`
    when possible.

    Parameters:
    -----------
    img_h: int | Height (px) of the image to be cropped\\n
    img_w: int | Width (px) of the image to be cropped\\n
    x: int | Leftmost coordinates of the detected face\\n
    y: int | Bottom-most coordinates of the detected face\\n
    w: int | Width of the detected face\\n
    h: int | Height of the detected face\\n
    """
    """Find out what zoom factor to use given self.aspect_ratio"""
    corners = itertools.product((x, x + w), (y, y + h))
    center = np.array([x + w * 0.5, y + h * 0.5]).astype(int)
    """image_corners"""
    im = np.array([(0, 0), (0, img_h), (img_w, img_h), (img_w, 0), (0, 0)])
    image_sides = np.array([(im[n], im[n + 1]) for n in np.arange(0, 4)])
    """Hopefully we use this one"""
    corner_ratios = [percent_face]
    for c in corners:
        corner_vector = np.array([center, c])
        a = distance(*corner_vector)
        intersects = np.array([intersect(corner_vector, side) for side in image_sides])
        for pt in intersects:
            """if intersect within image"""
            if (pt >= 0).all() and (pt <= im[2]).all():
                dist_to_pt = distance(center, pt)
                if float(dist_to_pt) == 0.0:
                    np.append(corner_ratios, 10000 * a)
                else:
                    np.append(corner_ratios, 100 * a / dist_to_pt)
    return np.amax(corner_ratios)


def _crop_positions(img_h, img_w, x, y, w, h, percent_face, wide, high):
    """
    Returns the coordinates of the crop position centered
    around the detected face with extra margins. Tries to
    honor `self.face_percent` if possible, else uses the
    largest margins that comply with required aspect ratio
    given by `self.height` and `self.width`.

    Parameters:
    -----------
    img_h: int | Height (px) of the image to be cropped\\n
    img_w: int | Width (px) of the image to be cropped\\n
    x: int | Leftmost coordinates of the detected face\\n
    y: int | Bottom-most coordinates of the detected face\\n
    w: int | Width of the detected face\\n
    h: int | Height of the detected face\\n
    """

    """aspect: float | Aspect ratio"""
    aspect = wide / high
    """zoom: float | Zoom factor"""
    zoom = _determine_safe_zoom(img_h, img_w, x, y, w, h, percent_face)

    """Adjust output height based on percent"""
    if high >= wide:
        height_crop = h * 100.0 / zoom
        width_crop = aspect * float(height_crop)
    else:
        width_crop = w * 100.0 / zoom
        height_crop = float(width_crop) / aspect

    """Calculate padding by centering face"""
    x_pad, y_pad = (width_crop - w) * 0.5, (height_crop - h) * 0.5
    """Calc. positions of crop"""
    h1, h2 = x - x_pad, x + w + x_pad
    v1, v2 = y - y_pad, y + h + y_pad

    return np.array([v1, v2, h1, h2]).astype(int)


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

    for i in range(output.shape[0]):
        """get the confidence"""
        confidence = output[i, 2]
        """if confidence is above 50%, then draw the surrounding box"""
        if confidence > conf:
            """get the surrounding box coordinates and upscale them to original image"""
            box = output[i, 3:7] * np.array([width, height, width, height])
            """convert to integers"""
            startx, starty, endx, endy = box.astype(int)
            pos = _crop_positions(height, width, startx, starty, endx - startx, endy - starty, face_perc, wide, high)
            waitKey(0)
            return np.array([pos[0], pos[1], pos[2], pos[3]])
        else:
            return None


def display_crop(img_path, wide, high, conf, face_perc, label, gam):
    bounding_box = box_detect(img_path, wide, high, conf, face_perc)
    label.setScaledContents(False)
    """Save the cropped image with PIL if a face was detected"""
    if bounding_box is not None:
        """Open image and check exif orientation and rotate accordingly"""
        with Image.open(img_path) as img:
            pic = reorient_image(img)
        """crop picture"""
        cropped_pic = pic.crop((bounding_box[2], bounding_box[0], bounding_box[3], bounding_box[1]))
        cropped_pic = np.array(cropped_pic)
        table = gamma(gam * GAMMA_THRESHOLD)
        cropped_pic = LUT(cropped_pic, table)

        pic_array = cvtColor(np.array(cropped_pic), COLOR_BGR2RGB)
        height, width, channel = pic_array.shape
        bytesPerLine = 3 * width

        qImg = QImage(pic_array.data, width, height, bytesPerLine, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(qImg)
        label.setPixmap(pixmap.scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio))


def reorient_image(im):
    try:
        image_exif = im._getexif()
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
