import itertools
import os
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage
from cv2 import cvtColor, dnn, imread, COLOR_BGR2RGB, LUT, imwrite, resize, INTER_AREA
from settings import FileTypeList, SpreadSheet, GAMMA_THRESHOLD, proto_path, caffe_path

caffe_model = dnn.readNetFromCaffe(proto_path, caffe_path)


def intersect(corner_vectors, image_sides):
    corn = np.tile(corner_vectors, (4, 1)).reshape((16, 2, 2))

    da = np.fliplr(corn[:, 1] - corn[:, 0])
    da[:, 0] *= -1
    db = np.tile(image_sides[:, 1] - image_sides[:, 0], (4, 1))

    dp = corner_vectors[:, 0] - image_sides[:, 0]
    dp = np.tile([np.append(dp[:2], np.rot90(dp[:2], 2)), np.append(np.rot90(dp[2:], 2), dp[2:])], 2).reshape(16, 2)

    dividend, divisor = np.sum(da * dp, axis=1), np.sum(da * db, axis=1)
    xcv = np.where(divisor == 0, 100 * dividend, dividend / divisor) * db.T
    gcv = xcv.T + np.tile(image_sides[:, 0], (4, 1))
    return gcv.reshape(4, 4, 2)


def gamma(gam=1.0):
    if gam != 1.0:
        return np.power(np.arange(256) / 255, 1.0 / gam) * 255
    else:
        return np.arange(256)


def open_file(input_filename):
    print(input_filename)
    """Given a filename, returns a numpy array"""
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


def determine_safe_zoom(img_h, img_w, x, y, w, h, percent_face):
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
    corner_vectors = np.hstack([np.repeat(center, 4, axis=0).reshape((4, 2)),
                                np.array(list(itertools.product((x, x + w), (y, y + h))))]).reshape((4, 2, 2))
    im = np.zeros((5, 2))
    im[1:2, 1], im[2:3, 0] = img_h, img_w

    intersects = intersect(corner_vectors, np.dstack([im[:-1], im[1:]]))
    B = (intersects >= 0) * (intersects <= im[2])
    V = np.linalg.norm(corner_vectors[:, 1] - corner_vectors[:, 0], axis=1)
    c, v = np.meshgrid(np.linalg.norm(np.linalg.norm(intersects - center, axis=1), axis=1), V)
    u = 100 * v / c
    if np.inf in u and B.all():
        u[:, np.isinf(u[0])] = 1e4 * V.reshape(4, 1)

    result = np.append(percent_face, u)
    return np.amax(result)


def crop_positions(img_h, img_w, x, y, w, h, percent_face, wide, high):
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
    zoom = determine_safe_zoom(img_h, img_w, x, y, w, h, percent_face)
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
    x0, y0, x1, y1 = (output[:, 3:7] * np.array([width, height, width, height]))[np.argmax(confidence_list)]
    return crop_positions(height, width, x0, y0, x1 - x0, y1 - y0, face_perc, wide, high)


def display_crop(img_path, wide, high, conf, face_perc, label, gam):
    bounding_box = box_detect(img_path, wide, high, conf, face_perc)
    label.setScaledContents(False)
    """Save the cropped image with PIL if a face was detected"""
    if bounding_box is not None:
        """Open image and check exif orientation and rotate accordingly"""
        with Image.open(img_path) as img:
            pic = reorient_image(img)
            """crop picture"""
            cropped_pic = np.array(pic.crop((bounding_box[2], bounding_box[0], bounding_box[3], bounding_box[1])))
            cropped_pic = LUT(cropped_pic, gamma(gam * GAMMA_THRESHOLD).astype('uint8'))

            pic_array = cvtColor(np.array(cropped_pic), COLOR_BGR2RGB)
            height, width = pic_array.shape[:2]

            qImg = QImage(pic_array.data, width, height, 3 * width, QImage.Format.Format_BGR888)
            pixmap = QPixmap.fromImage(qImg)
            label.setPixmap(pixmap.scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio))


def reorient_image(im):
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


def crop(image, file_bool, destination, width, height, confidence, face, user_gam, radio, n, lines, radio_choices):
    source, image = os.path.split(image) if file_bool else (lines[n].text(), image)
    path = f'{source}\\{image}'
    bounding_box = box_detect(path, int(width), int(height), float(confidence), float(face))
    """Save the cropped image with PIL if a face was detected"""
    if bounding_box is not None:
        """Open image and check exif orientation and rotate accordingly"""
        pic = reorient_image(Image.open(path))
        """crop picture"""
        cropped_pic = pic.crop((bounding_box[2], bounding_box[0], bounding_box[3], bounding_box[1]))
        """Colour correct as Numpy array"""
        pic_array = cvtColor(np.array(cropped_pic), COLOR_BGR2RGB)
        cropped_image = resize(pic_array, (int(width), int(height)), interpolation=INTER_AREA)
        table = gamma(user_gam * GAMMA_THRESHOLD)
        if not os.path.exists(destination):
            os.makedirs(destination, mode=438, exist_ok=True)
        if radio == radio_choices[0]:
            imwrite(f'{destination}\\{image}', LUT(cropped_image, table))
        elif radio in radio_choices[1:]:
            imwrite(f'{destination}\\{os.path.splitext(image)[0]}{radio}', LUT(cropped_image, table))
    else:
        reject = f'{destination}\\reject'
        if not os.path.exists(reject):
            os.makedirs(reject, mode=438, exist_ok=True)
        to_file = f'{reject}\\{image}'
        shutil.copy(path, to_file)


def m_crop(source_folder, image, new, destination, width, height, confidence, face, user_gam, radio, radio_choices):
    path = f'{source_folder}\\{image}'
    bounding_box = box_detect(path, int(width), int(height), float(confidence), float(face))
    """Save the cropped image with PIL if a face was detected"""
    if bounding_box is not None:
        """Open image and check exif orientation and rotate accordingly"""
        pic = reorient_image(Image.open(path))
        """crop picture"""
        cropped_pic = pic.crop((bounding_box[2], bounding_box[0], bounding_box[3], bounding_box[1]))
        """Colour correct as Numpy array"""
        pic_array = cvtColor(np.array(cropped_pic), COLOR_BGR2RGB)
        cropped_image = resize(pic_array, (int(width), int(height)), interpolation=INTER_AREA)
        table = gamma(user_gam * GAMMA_THRESHOLD)
        if not os.path.exists(destination):
            os.makedirs(destination, mode=438, exist_ok=True)
        if radio == radio_choices[0]:
            imwrite(f'{destination}\\{new}{os.path.splitext(image)[1]}', LUT(cropped_image, table))
        elif radio in radio_choices[1:]:
            imwrite(f'{destination}\\{new}{radio}', LUT(cropped_image, table))
    else:
        reject = f'{destination}\\reject'
        if not os.path.exists(reject):
            os.makedirs(reject, mode=438, exist_ok=True)
        to_file = f'{reject}\\{image}'
        shutil.copy(path, to_file)
