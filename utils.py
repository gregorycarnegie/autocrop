import os
import re
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tifffile as tiff
from PIL import Image
from PyQt6 import QtCore, QtGui, QtWidgets


class ImageWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = None

    def setImage(self, image: QtGui.QPixmap):
        self.image = image
        self.update()

    def paintEvent(self, event):
        if self.image is not None:
            qp = QtGui.QPainter(self)
            qp.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
            scaled_image = self.image.scaled(self.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                                             QtCore.Qt.TransformationMode.SmoothTransformation)
            x_offset = (self.width() - scaled_image.width()) // 2
            y_offset = (self.height() - scaled_image.height()) // 2
            qp.drawPixmap(x_offset, y_offset, scaled_image)


PIL_TYPES = ['.bmp', '.dib', '.jfif', '.jp2', '.jpe', '.jpeg', '.jpg', '.pbm',
             '.pgm', '.png', '.ppm', '.ras', '.sr', '.tif', '.tiff', '.webp']
CV2_TYPES = ['.eps', '.icns', '.ico', '.im', '.msp', '.pcx', '.sgi', '.spi', '.xbm']
RAW_TYPES = ['.dng', '.nef', '.raw']

GAMMA_THRESHOLD = 0.001


def caffe_model():
    return cv2.dnn.readNetFromCaffe("resources\\weights\\deploy.prototxt.txt",
                                    "resources\\models\\res10_300x300_ssd_iter_140000.caffemodel")


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


def correct_exposure(image: cv2.Mat | np.ndarray) -> (cv2.Mat | np.ndarray):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

    # Get the average pixel value
    # average_pixel_value = np.sum(hist * range(256)) / np.sum(hist)

    # Determine whether the image is overexposed or underexposed and Apply brightness and contrast adjustment
    # if average_pixel_value < 127:
    #     # Image is underexposed, increase brightness and contrast
    #     return cv2.convertScaleAbs(image, alpha=1.5, beta=50)
    # elif average_pixel_value > 128:
    #     # Image is overexposed, decrease brightness and contrast
    #     return cv2.convertScaleAbs(image, alpha=0.5, beta=-50)
    # else:
    #     # Image is properly exposed
    #     return image
    if (average_pixel_value := np.sum(hist * range(256)) / np.sum(hist)) < 127:
        # Image is underexposed, increase brightness and contrast
        return cv2.convertScaleAbs(image, alpha=1.5, beta=50)
    elif average_pixel_value > 128:
        # Image is overexposed, decrease brightness and contrast
        return cv2.convertScaleAbs(image, alpha=0.5, beta=-50)
    else:
        # Image is properly exposed
        return image


def open_file(input_filename: Path, exposure: Optional[bool] = False) -> (None | np.ndarray):
    """Given a filename, returns a numpy array or a pandas dataframe"""

    # extension = os.path.splitext(input_filename)[1].lower()
    input_file = Path(input_filename)
    extension = input_file.suffix.lower()
    path = input_file.as_posix()
    if extension in CV2_TYPES:
        """Try with cv2"""
        x = cv2.imread(path)
        assert not isinstance(x, type(None)), 'image not found'
        if exposure:
            x = correct_exposure(x)
        return x
    if extension in PIL_TYPES:
        """Try with PIL"""
        with Image.open(path).convert('RGB') as img_orig:
            x = np.array(img_orig)
            if exposure:
                x = correct_exposure(x)
            return x
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

    # Calculate padding by centering face
    x_pad, y_pad = (width_crop - w) * 0.5, (height_crop - h) * 0.5
    # Calc. positions of crop h1, h2, v1, v2
    return np.array([x - x_pad, y - y_pad, x + w + x_pad, y + h + y_pad]).astype(int)


def box_detect(img_path: cv2.Mat | Path, wide: int, high: int, conf: int, face_perc: int) -> (None | np.ndarray):
    img = img_path if isinstance(img_path, cv2.Mat) else open_file(img_path)
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
    return crop_positions(x0, y0, x1 - x0, y1 - y0, face_perc, wide, high)


def display_crop(img_path: Path, wide: int, high: int, conf: int, face_perc: int, gam: int,
                 image_widget: ImageWidget, file_types: Optional[np.ndarray] = None):
    """if img_path is a Path object and TYPES is None, the function will return None"""
    if img_path.as_posix() in {'', '.', None}:
        return None

    if img_path.is_dir():
        if file_types is None:
            return None
        files = np.fromiter(img_path.glob('*.*'), Path)
        file = np.array([pic for pic in files if pic.suffix in file_types])
        # img_path = dict(enumerate(file))[0]
        img_path = file[0].as_posix()
        # img_path = str(img_path)

    """Save the cropped image with PIL if a face was detected"""
    if (bounding_box := box_detect(img_path, wide, high, conf, face_perc)) is not None:
        # Open image and check exif orientation and rotate accordingly
        if isinstance(img_path, Path):
            photo_path = img_path.as_posix()
        elif isinstance(img_path, str):
            photo_path = img_path

        with Image.open(photo_path) as img:
            pic = reorient_image(img)
            # crop picture
            cropped_pic = np.array(pic.crop(bounding_box))
            cropped_pic = cv2.LUT(cropped_pic, gamma(gam * GAMMA_THRESHOLD).astype('uint8'))

            # Convert image to numpy array
            pic_array = cv2.cvtColor(np.array(cropped_pic), cv2.COLOR_BGR2RGB)

            # Convert numpy array to QImage
            height, width, channel = pic_array.shape
            bytes_per_line = channel * width
            # qImg = QtGui.QImage(pic_array.data, width, height, 3 * width, QtGui.QImage.Format.Format_BGR888)
            qImg = QtGui.QImage(pic_array.data, width, height, bytes_per_line, QtGui.QImage.Format.Format_BGR888)

            # Set image to the image widget
            image_widget.setImage(QtGui.QPixmap.fromImage(qImg))


def reorient_image(im: Image) -> Image:
    try:
        # image_orientation = im.getexif()[274]
        if (image_orientation := im.getexif()[274]) in {2, '2'}:
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


def crop_image(image: Path | np.ndarray, bounding_box, width, height) -> cv2.Mat:
    if isinstance(image, Path):
        pic = reorient_image(Image.open(image.as_posix()))
    elif isinstance(image, np.ndarray):
        pic = Image.fromarray(image)
    else:
        return None
    cropped_pic = pic.crop(bounding_box)
    pic_array = cv2.cvtColor(np.array(cropped_pic), cv2.COLOR_BGR2RGB)
    return cv2.resize(pic_array, (int(width), int(height)), interpolation=cv2.INTER_AREA)


def reject(path: Path, destination: Path, image: Path) -> None:
    # reject_folder = os.path.join(destination, 'reject')
    reject_folder = destination.joinpath(image)
    # os.makedirs(reject_folder, exist_ok=True)
    reject_folder.mkdir(exist_ok=True)
    # shutil.copy(path, os.path.join(reject_folder, image))
    shutil.copy(path, reject_folder.joinpath(image))


def save_detection(path: Path, destination: Path, image: Path, width: int, height: int, confidence: int, face: int,
                   user_gam: int, radio: str, r_choices: np.ndarray, new: Optional[str] = None) -> None:
    """
    This code first checks if bounding_box is not None, and if so, it proceeds to crop the image and create the
    destination directory if it doesn't already exist. It then constructs the file name using a ternary expression
    that appends the file extension to the file name if radio is equal to the first element in radio_choices,
    and appends radio itself if radio is not equal to the first element in radio_choices. The code then constructs
    the file path by joining the destination directory and the file name and saves the cropped image to the file
    using the imwrite() function. If bounding_box is None, the code calls the reject() function to reject the image.
    """
    # Save the cropped image if a face was detected

    if (bounding_box := box_detect(path, width, height, confidence, face)) is not None:
        cropped_image = crop_image(path, bounding_box, width, height)
        # os.makedirs(destination, exist_ok=True)
        destination.mkdir(exist_ok=True)
        # file = f'{new or os.path.splitext(image)[0]}{os.path.splitext(image)[1] if radio == r_choices[0] else radio}'
        file = f'{new or image.stem}{image.suffix if radio == r_choices[0] else radio}'
        # file_path = os.path.join(destination, file)
        file_path = destination.joinpath(file)
        # if Path(file_path).suffix in {'.tif', '.tiff'}:
        if file_path.suffix in {'.tif', '.tiff'}:
            tiff.imwrite(file_path.as_posix(), cv2.LUT(cropped_image, gamma(user_gam * GAMMA_THRESHOLD)))
        else:
            cv2.imwrite(file_path.as_posix(), cv2.LUT(cropped_image, gamma(user_gam * GAMMA_THRESHOLD)))
    else:
        reject(path, destination, image)


def cropframe(frame, video_line_edit: QtWidgets.QLineEdit, destination_line_edit_4: QtWidgets.QLineEdit, wide: int,
              high: int, conf: QtWidgets.QDial, face_perc: QtWidgets.QDial, gamma_dial: QtWidgets.QDial,
              position_label: QtWidgets.QLabel, radio: str, radio_options: np.ndarray) -> None:
    # frame, frame_width, frame_height = self.grab_frame()
    # frame = self.grab_frame(timeline_slider.value(), video_line_edit)

    if (bounding_box := box_detect(frame, wide, high, conf.value(), face_perc.value())) is not None:
        # destination = destination_line_edit_4.text()
        destination = Path(destination_line_edit_4.text())
        # base_name = os.path.splitext(os.path.basename(video_line_edit.text()))[0]
        base_name = Path(video_line_edit.text()).stem

        cropped_image = crop_image(frame, bounding_box, wide, high)
        # os.makedirs(destination, exist_ok=True)
        destination.mkdir(exist_ok=True)
        position = re.sub(':', '_', position_label.text())
        # file = os.path.join(destination,
        #                     f'{base_name} - ({position}){radio_options[2] if radio == radio_options[0] else radio}')
        # file_path = os.path.join(destination, file)
        file_path = destination.joinpath(
            f'{base_name} - ({position}){radio_options[2] if radio == radio_options[0] else radio}')
        if file_path.suffix in {'.tif', '.tiff'}:
            tiff.imwrite(file_path.as_posix(), cv2.LUT(cropped_image, gamma(gamma_dial.value() * GAMMA_THRESHOLD)))
        else:
            cv2.imwrite(file_path.as_posix(), cv2.LUT(cropped_image, gamma(gamma_dial.value() * GAMMA_THRESHOLD)))
    else:
        return None


def frame_extraction(video, output_dir: QtWidgets.QLineEdit, frame_number: int, frame_step: int,
                     width: int, height: int, confidence: QtWidgets.QDial,
                     face: QtWidgets.QDial, user_gam: QtWidgets.QDial, radio: str, radio_options: np.ndarray) -> None:
    # Set the current time in the video.
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Read the current frame from the video.
    ret, frame = video.read()

    # Check if the frame was successfully read.
    if not ret:
        return None

    # Save the current frame as a JPG file.
    destination = Path(output_dir.text())
    # os.makedirs(output_dir.text(), exist_ok=True)
    destination.mkdir(exist_ok=True)
    if (bounding_box := box_detect(frame, width, height, confidence.value(), face.value())) is not None:
        cropped_image = crop_image(frame, bounding_box, width, height)
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        # file = f"frame_{frame_number:06d}{radio_options[2] if radio == radio_options[0] else radio}"
        # file_path = os.path.join(output_dir.text(), file)
        file_path = destination.joinpath(
            f'frame_{frame_number:06d}{radio_options[2] if radio == radio_options[0] else radio}')
        if file_path.suffix in {'.tif', '.tiff'}:
            tiff.imwrite(file_path.as_posix(), cv2.LUT(cropped_image, gamma(user_gam.value() * GAMMA_THRESHOLD)))
        else:
            cv2.imwrite(file_path.as_posix(), cv2.LUT(cropped_image, gamma(user_gam.value() * GAMMA_THRESHOLD)))
    else:
        # file_path = os.path.join(output_dir.text(), f"failed_frame_{frame_number:06d}{radio_options[2] if radio == radio_options[0] else radio}")
        file_path = destination.joinpath(
            f'failed_frame_{frame_number:06d}{radio_options[2] if radio == radio_options[0] else radio}')
        if file_path.suffix in {'.tif', '.tiff'}:
            tiff.imwrite(file_path.as_posix(), frame)
        else:
            cv2.imwrite(file_path.as_posix(), frame)

    # Move to the next frame.
    frame_number += frame_step


# def crop(image: Path, file_bool: bool, destination: Path, width: int, height: int, confidence: int, face: int,
#          user_gam: int, radio: str, line_edit: str, radio_choices: np.ndarray):
#     source, image = os.path.split(image) if file_bool else (line_edit, image)
#     # path = os.path.join(source, image)
#     path = Path(source).joinpath(image)
#     save_detection(path, destination, image, width, height, confidence, face, user_gam, radio, radio_choices, None)

# def crop(image: Path, destination: Path, width: int, height: int, confidence: int, face: int,
#          user_gam: int, radio: str, radio_choices: np.ndarray, line_edit: Optional[str] = None) -> None:
#     # source, image = os.path.split(image) if line_edit is None else (line_edit, image)
#     if line_edit is None:
#         source, image = image.stem, image.suffix
#     else:
#         source, image = line_edit, image
#     # source, image = (image.stem, image.suffix) if line_edit is None else (line_edit, image)
#     # path = os.path.join(source, image)
#     path = Path(source).joinpath(image)
#     save_detection(path, destination, image, width, height, confidence, face, user_gam, radio, radio_choices)


def crop(image: Path, file_bool: bool, destination: Path, width: int, height: int, confidence: int, face: int,
         user_gam: int, radio: str, line_edit: str, radio_choices: np.ndarray):
    source, image_name = (Path(line_edit), image.name) if file_bool else (image.parent, image.name)
    path = source.joinpath(image_name)
    print(path.as_posix())
    # save_detection(path, destination, Path(image_name), width, height, confidence, face, user_gam, radio, radio_choices, None)


# def m_crop(source_folder: Path, image: str, destination: Path, width: int, height: int, confidence: int,
#            face: int, user_gam: int, radio: str, radio_choices: np.ndarray, new: str):
#     # path = os.path.join(source_folder, image)
#     path = source_folder.joinpath(image)
#     save_detection(path, destination, image, width, height, confidence, face, user_gam, radio, radio_choices, new)

def m_crop(image: Path, destination: Path, width: int, height: int, confidence: int,
           face: int, user_gam: int, radio: str, radio_choices: np.ndarray, source_folder: Path, new: str) -> None:
    # path = os.path.join(source_folder, image)
    path = source_folder.joinpath(image)
    save_detection(path, destination, image, width, height, confidence, face, user_gam, radio, radio_choices, new)


# def crap(image: Path, destination: Path, width: int, height: int, confidence: int,
#          face: int, user_gam: int, radio: str, radio_choices: np.ndarray,
#          line_edit: Optional[str] = None, source_folder: Optional[Path] = None, new: Optional[str] = None) -> None:
#     if source_folder is None and new is None:
#         source, image = (image.stem, image.suffix) if line_edit is None else (line_edit, image)
#         source_folder = Path(source)
#         path = source_folder.joinpath(image)
#         save_detection(path, destination, image, width, height, confidence, face, user_gam, radio, radio_choices)
#     elif line_edit is None and isinstance(source_folder, Path):
#         path = source_folder.joinpath(image)
#         save_detection(path, destination, image, width, height, confidence, face, user_gam, radio, radio_choices, new)

def crap(image: Path, destination: Path, width: int, height: int, confidence: int,
         face: int, user_gam: int, radio: str, radio_choices: np.ndarray,
         line_edit: Optional[str] = None, source_folder: Optional[Path] = None, new: Optional[str] = None) -> None:
    if isinstance(source_folder, Path) and isinstance(new, str):
        path = source_folder.joinpath(image)
        save_detection(path, destination, image, width, height, confidence, face, user_gam, radio, radio_choices, new)
    elif isinstance(line_edit, str):
        source, image_name = Path(line_edit), image.name
        path = source.joinpath(image_name)
        save_detection(path, destination, Path(image_name), width, height, confidence, face, user_gam, radio, radio_choices)

"""
# def extract_frames(video_path, start_time, end_time, frame_step, output_dir, width, height, confidence, face, user_gam):
#     \"""
#     Extracts frames from an MP4 video file between two specific times with a specified time step, and saves each frame
#     as a JPG file.

#     Args:
#     - video_path (str): The path to the MP4 video file to extract frames from.
#     - start_time (float): The start time of the frame extraction in seconds.
#     - end_time (float): The end time of the frame extraction in seconds.
#     - time_step (float): The time step between extracted frames in seconds.
#     - output_dir (str): The directory to save the extracted frames in.

#     Returns:
#     - None
#     \"""

#     # Open the video file and get its properties.
#     video = cv2.VideoCapture(video_path)
#     fps = int(video.get(cv2.CAP_PROP_FPS))

#     # Calculate the start and end frame numbers.
#     start_frame = int(start_time * fps)
#     end_frame = int(end_time * fps)

#     # Set the initial frame number and the list of extracted frames.
#     frame_number = start_frame

#     # Iterate over the frames in the video and extract the desired frames.
#     while frame_number <= end_frame:
#         # Set the current time in the video.
#         video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

#         # Read the current frame from the video.
#         ret, frame = video.read()

#         # Check if the frame was successfully read.
#         if not ret:
#             break

#         # Save the current frame as a JPG file.
#         os.makedirs(output_dir, exist_ok=True)
#         if (bounding_box := box_detect(frame, width, height, confidence, face)) is not None:
#             cropped_image = crop_image(frame, bounding_box, width, height)
#             cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
#             file = f"frame_{frame_number:06d}.jpg"
#             file_path = os.path.join(output_dir, file)
#             cv2.imwrite(file_path, cv2.LUT(cropped_image, gamma(user_gam * GAMMA_THRESHOLD)))
#         else:
#             output_path = os.path.join(output_dir, f"failed_frame_{frame_number:06d}.jpg")
#             cv2.imwrite(output_path, frame)

#         # Move to the next frame.
#         frame_number += frame_step

#     # Release the video file.
#     video.release()
"""
