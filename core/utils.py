import cProfile
import collections.abc as c
import pstats
import random
import shutil
from functools import cache, wraps
from pathlib import Path
from typing import Any, Optional, Union

import autocrop_rs as rs
import cv2
import cv2.typing as cvt
import numba
import numpy as np
import numpy.typing as npt
import polars as pl
import rawpy
import tifffile as tiff
from PIL import Image

from file_types import Photo
from .job import Job
from .operation_types import Box, FaceToolPair, ImageArray, SaveFunction
from .resource_path import ResourcePath

# Define constants
GAMMA_THRESHOLD = .001
L_EYE_START, L_EYE_END = 42, 48
R_EYE_START, R_EYE_END = 36, 42

CropFunction = c.Callable[
    [Union[cvt.MatLike, Path], Job, tuple[Any, Any]],
    Optional[Union[cvt.MatLike, c.Iterator[cvt.MatLike]]]
]


def profile_it(func: c.Callable[..., Any]) -> c.Callable[..., Any]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any):
        with cProfile.Profile() as profile:
            func(*args, **kwargs)
        result = pstats.Stats(profile)
        result.sort_stats(pstats.SortKey.CUMULATIVE)
        result.print_stats()

    return wrapper


@numba.njit(cache=True)
def gamma(gamma_value: Union[int, float] = 1.0) -> npt.NDArray[np.float64]:
    """
    The function applies a gamma correction to an array of intensity values ranging from 0 to 255. It returns the
    corrected array.

    Args:
        gamma_value (Union[int, float], optional): The gamma value for the correction. Defaults to 1.0.

    Returns:
        npt.NDArray[np.generic]: The array of intensity values after gamma correction.

    Example:
        ```python
        # Applying gamma correction with gamma value of 2.2
        corrected_array = gamma(2.2)

        # Printing the corrected array
        print(corrected_array)
        ```
    """

    return np.arange(256, dtype=np.float64) if gamma_value <= 1.0 else np.power(np.arange(256) / 255,
                                                                                1.0 / gamma_value) * 255.0


def adjust_gamma(input_image: ImageArray, gam: int) -> cvt.MatLike:
    """
    The function adjusts the gamma of the provided image using a lookup table.

    Args:
        input_image (Union[cvt.MatLike, npt.NDArray[np.uint8]]): The image to adjust the gamma.
        gam (int): The gamma value.

    Returns:
        cvt.MatLike: The image with the gamma adjusted.

    Example:
        ```python
        from autocrop import cvt

        # Creating an image.
        image = cvt.MatLike()

        # Adjusting the gamma of the image.
        adjusted_image = adjust_gamma(image, gam=2)
        ```
    """

    return cv2.LUT(input_image, gamma(gam * GAMMA_THRESHOLD).astype('uint8'))


def convert_color_space(input_image: ImageArray) -> cvt.MatLike:
    """
    The function converts the color space of the provided image from BGR to RGB.

    Args:
        input_image (Union[cvt.MatLike, npt.NDArray[np.uint8]]): The image to convert.

    Returns:
        cvt.MatLike: The image with the color space converted to RGB.

    Example:
        ```python
        from autocrop import cvt

        # Creating an image.
        image = cvt.MatLike()

        # Converting the color space of the image.
        converted_image = convert_color_space(image)
        ```
    """

    return cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)


def numpy_array_crop(input_image: cvt.MatLike, bounding_box: Box) -> npt.NDArray[np.uint8]:
    """
    The function crops the provided image using the specified bounding box and returns the cropped image as a NumPy
    array.

    Args:
        input_image (cvt.MatLike): The image to be cropped.
        bounding_box (Tuple[int, int, int, int]): The bounding box coordinates (left, upper, right, lower) for cropping.

    Returns:
        npt.NDArray[np.uint8]: The cropped image as a NumPy array.

    Example:
        ```python
        from autocrop import cvt

        # Creating an image.
        image = cvt.MatLike()

        # Defining the bounding box.
        bounding_box = (100, 100, 300, 300)

        # Cropping the image.
        cropped_image = numpy_array_crop(image, bounding_box)
        ```
    """

    # Load and crop image using PIL
    picture = Image.fromarray(input_image).crop(bounding_box)
    return np.array(picture)


def correct_exposure(input_image: ImageArray,
                     exposure: bool) -> cvt.MatLike:
    """
    The function corrects the exposure of the provided image using histogram equalization.

    Args:
        input_image (Union[cvt.MatLike, npt.NDArray[np.uint8]]): The image to correct the exposure.
        exposure (bool): Flag indicating whether to perform exposure correction.

    Returns:
        cvt.MatLike: The image with corrected exposure.

    Example:
        ```python
        from autocrop import cvt

        # Creating an image.
        image = cvt.MatLike()

        # Correcting the exposure of the image.
        corrected_image = correct_exposure(image, exposure=True)
        ```
    """

    if not exposure:
        return input_image
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY) if len(input_image.shape) > 2 else input_image
    # Grayscale histogram
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    # Calculate alpha and beta
    alpha, beta = rs.calc_alpha_beta(hist)
    return cv2.convertScaleAbs(src=input_image, alpha=alpha, beta=beta)


def rotate_image(input_image: ImageArray,
                 angle: float,
                 center: tuple[float, float]) -> cvt.MatLike:
    """
    The function rotates the provided image by the specified angle around the given center.

    Args:
        input_image (Union[cvt.MatLike, npt.NDArray[np.uint8]]): The image to rotate.
        angle (float): The angle of rotation in degrees.
        center (Tuple[float, float]): The center point of rotation.

    Returns:
        cvt.MatLike: The rotated image.

    Example:
        ```python
        from autocrop import cvt

        # Creating an image.
        image = cvt.MatLike()

        # Defining the rotation angle and center
        angle = 45.0
        center = (image.shape[1] / 2, image.shape[0] / 2)

        # Rotating the image
        rotated_image = rotate_image(image, angle, center)
        ```
    """

    # Get the rotation matrix.
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    # Apply the rotation to the image.
    return cv2.warpAffine(input_image, rotation_matrix, (input_image.shape[1], input_image.shape[0]))


@numba.njit
def get_dimensions(input_image: ImageArray, output_height: int) -> tuple[int, float]:
    scaling_factor = output_height / input_image.shape[:2][0]
    return int(input_image.shape[:2][1] * scaling_factor), scaling_factor


def format_image(input_image: ImageArray) -> tuple[cvt.MatLike, float]:
    output_height = 256
    output_width, scaling_factor = get_dimensions(input_image, output_height)
    image_array = cv2.resize(input_image, (output_width, output_height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY) if len(image_array.shape) >= 3 else image_array.astype(
        np.int8), scaling_factor


# # TODO: JIT this function
# def get_angle_of_tilt(landmarks_array: npt.NDArray[np.int_], scaling_factor: float) -> Tuple[float, float, float]:
#     # Find the eyes in the image (l_eye - r_eye).
#     print(landmarks_array.shape)
#     eye_diff = np.mean(landmarks_array[L_EYE_START:L_EYE_END], axis=0) - \
#                np.mean(landmarks_array[R_EYE_START:R_EYE_END], axis=0)
#     # Find the center of the face.
#     center_x, center_y = np.mean(landmarks_array[R_EYE_START:L_EYE_END], axis=0) / scaling_factor
#     return np.arctan2(eye_diff[1], eye_diff[0]) * 180 / np.pi, center_x, center_y

@numba.njit
def mean_axis0(arr: npt.NDArray[np.int_]) -> npt.NDArray[np.float64]:
    """Calculate mean along axis 0."""
    n, m = arr.shape
    mean = np.zeros(m)
    for i in range(m):
        sum_ = 0.0
        for j in range(n):
            sum_ += arr[j, i]
        mean[i] = sum_ / n
    return mean


@numba.njit
def get_angle_of_tilt(landmarks_array: npt.NDArray[np.int_], scaling_factor: float) -> tuple[float, float, float]:
    # Find the eyes in the image (l_eye - r_eye).
    # left_eye_mean = mean_axis0(landmarks_array[L_EYE_START:L_EYE_END])
    # right_eye_mean = mean_axis0(landmarks_array[R_EYE_START:R_EYE_END])
    eye_diff = mean_axis0(landmarks_array[L_EYE_START:L_EYE_END]) - mean_axis0(landmarks_array[R_EYE_START:R_EYE_END])

    # Find the center of the face.
    face_center_mean = mean_axis0(landmarks_array[R_EYE_START:L_EYE_END])
    center_x, center_y = face_center_mean / scaling_factor

    angle = np.arctan2(eye_diff[1], eye_diff[0]) * 180 / np.pi

    return angle, center_x, center_y


def align_head(input_image: ImageArray,
               face_detection_tools: FaceToolPair,
               tilt: bool) -> cvt.MatLike:
    """
    The function aligns the head in the provided image using facial landmarks and tilt correction.

    Args:
        input_image (Union[cvt.MatLike, npt.NDArray[np.uint8]]): The image to align the head.
        face_detection_tools ( Tuple[Any, Any]): The face worker object for face detection and landmark prediction.
        tilt (bool): Flag indicating whether to perform tilt correction.

    Returns:
        cvt.MatLike: The aligned image.

    Example:
        ```python
        from autocrop import cvt,  Tuple[Any, Any]

        # Creating an image.
        image = cvt.MatLike()

        # Creating a face worker.
        face_detection_tools =  Tuple[Any, Any]()

        # Aligning the head in the image.
        aligned_image = align_head(image, face_detection_tools, tilt=True)
        ```
    """

    if not tilt:
        return input_image

    image_array, scaling_factor = format_image(input_image)

    face_detector, predictor = face_detection_tools

    faces = face_detector(image_array, 1)
    # If no faces are detected, return the original image.
    if len(faces) == 0:
        return input_image

    # Find the face with the highest confidence score.
    face = max(faces, key=lambda x: x.area())
    # Get the 68 facial landmarks for the face.
    landmarks = predictor(image_array, face)
    landmarks_array = np.array([(p.x, p.y) for p in landmarks.parts()])
    # Find the angle of the tilt and the center of the face
    angle, center_x, center_y = get_angle_of_tilt(landmarks_array, scaling_factor)
    return rotate_image(input_image, angle, (center_x, center_y))


def open_image(input_image: Path,
               face_detection_tools: FaceToolPair, *,
               exposure: bool,
               tilt: bool) -> Optional[cvt.MatLike]:
    """
    The function opens an image file using `cv2` and performs color conversion, exposure correction,
    and head alignment using the provided ` Tuple[Any, Any]` object.

    Args:
        input_image (Path): The path to the image file.
        face_detection_tools ( Tuple[Any, Any]): The  Tuple[Any, Any] object used for aligning the head.
        exposure (bool): Flag indicating whether to correct the exposure.
        tilt (bool): Flag indicating whether to align the head.

    Returns:
        cvt.MatLike: The processed image data.

    Example:
        ```python
        # Opening an image file.
        image_path = Path('/path/to/image.jpg')
        face_detection_tools =  Tuple[Any, Any]()
        processed_image = open_image(image_path, face_detection_tools, exposure=True, tilt=True)
        cv2.imshow('Processed Image', processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        ```
    """

    img = cv2.imread(input_image.as_posix())
    # handle file error
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = correct_exposure(img, exposure)
    return align_head(img, face_detection_tools, tilt)


def open_raw(input_image: Path,
             face_detection_tools: FaceToolPair, *,
             exposure: bool,
             tilt: bool) -> Optional[cvt.MatLike]:
    """
    The function opens a raw image file using `rawpy` and performs post-processing on the raw image data. It corrects
    the exposure and aligns the head using the provided ` Tuple[Any, Any]` object.

    Args:
        input_image (Path): The path to the raw image file.
        face_detection_tools ( Tuple[Any, Any]): The  Tuple[Any, Any] object used for aligning the head.
        exposure (bool): Flag indicating whether to correct the exposure.
        tilt (bool): Flag indicating whether to align the head.

    Returns:
        cvt.MatLike: The processed image data.

    Example:
        ```python
        # Opening a raw image file.
        image_path = Path('/path/to/image.CR2')
        face_detection_tools =  Tuple[Any, Any]()
        processed_image = open_raw(image_path, face_detection_tools, exposure=True, tilt=True)
        cv2.imshow('Processed Image', processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        ```
    """

    with rawpy.imread(input_image.as_posix()) as raw:
        # Post-process the raw image data
        # handle file error
        if raw is None:
            return None
        img = raw.postprocess(use_camera_wb=True)
        img = correct_exposure(img, exposure)
        return align_head(img, face_detection_tools, tilt)


def open_table(input_file: Path) -> pl.DataFrame:
    """
    The function opens a table file and returns its contents as a `pd.DataFrame` object. If the file has a `.csv`
    extension, the function uses `pd.read_csv` to read the file. Otherwise, it uses `pd.read_excel`.

    Args:
        input_file (Path): The path to the table file.

    Returns:
        pd.DataFrame: The contents of the table file as a DataFrame.

    Example:
        ```python
        # Opening a CSV file.
        csv_file = Path('/path/to/table.csv')
        csv_data = open_table(csv_file)
        print(csv_data.head())

        # Opening an Excel file.
        excel_file = Path('/path/to/table.xlsx')
        excel_data = open_table(excel_file)
        print(excel_data.head())
        ```
    """

    return pl.read_csv(input_file) if input_file.suffix.lower() == '.csv' else pl.read_excel(input_file)

def open_pic(input_file: Union[Path, str],
             face_detection_tools: FaceToolPair, *,
             exposure: bool,
             tilt: bool) -> Optional[cvt.MatLike]:
    """
    The function opens an image file based on its extension. If the extension is in the list of supported CV2 types,
    the function calls `open_image` to open the image using `cv2`. If the extension is in the list of supported RAW
    types, the function calls `open_raw` to open the raw image using `rawpy`. If the extension is not supported,
    the function returns None.

    Args:
        input_file (Union[Path, str]): The path or string representing the input image file.
        face_detection_tools ( Tuple[Any, Any]): The  Tuple[Any, Any] object used for aligning the head.
        exposure (bool): Flag indicating whether to correct the exposure.
        tilt (bool): Flag indicating whether to align the head.

    Returns:
        Optional[cvt.MatLike]: The processed image data, or None if the extension is not supported.

    Example:
        ```python
        # Opening an image file.
        image_path = Path('/path/to/image.jpg')
        face_detection_tools =  Tuple[Any, Any]()
        processed_image = open_pic(image_path, face_detection_tools, exposure=True, tilt=True)
        if processed_image is not None:
            cv2.imshow('Processed Image', processed_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Unsupported file extension.")
        ```
    """

    input_file = Path(input_file) if isinstance(input_file, str) else input_file
    match input_file.suffix.lower():
        case extension if extension in Photo.CV2_TYPES:
            return open_image(input_file, face_detection_tools, exposure=exposure, tilt=tilt)
        case extension if extension in Photo.RAW_TYPES:
            return open_raw(input_file, face_detection_tools, exposure=exposure, tilt=tilt)
        case _:
            return


def prepare_detections(input_image: cvt.MatLike) -> npt.NDArray[np.float64]:
    # Create blob from image
    # We standardize the image by scaling it and then subtracting the mean RGB values
    blob = cv2.dnn.blobFromImage(input_image, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)
    # Set the input for the neural network
    proto_txt = ResourcePath('resources\\weights\\deploy.prototxt.txt').meipass_path
    caffe_model = ResourcePath('resources\\models\\res10_300x300_ssd_iter_140000.caffemodel').meipass_path
    caffe = cv2.dnn.readNetFromCaffe(proto_txt, caffe_model)
    caffe.setInput(blob)
    # Forward pass through the network to get detections
    return np.array(caffe.forward())


def rs_crop_positions(box_outputs: npt.NDArray[np.int_], job: Job) -> Box:
    x0, y0, x1, y1 = box_outputs
    return rs.crop_positions(x0, y0, x1 - x0, y1 - y0, job.face_percent, job.width,
                             job.height, job.top, job.bottom, job.left, job.right)


def get_box_coordinates(output: Union[cvt.MatLike, npt.NDArray[np.generic]],
                        job: Job, *,
                        width: int,
                        height: int,
                        x: Optional[npt.NDArray[np.generic]] = None) -> Box:
    box_outputs = output * np.array([width, height, width, height])
    _box = box_outputs.astype(np.int_) if x is None else box_outputs[np.argmax(x)]
    return rs_crop_positions(_box, job)


def get_multibox_coordinates(box_outputs: npt.NDArray[np.int_], job: Job) -> c.Iterator[Box]:
    return map(lambda x: rs_crop_positions(x, job), box_outputs)


def box(input_image: cvt.MatLike,
        job: Job, *,
        width: int,
        height: int, ) -> Optional[Box]:
    # preprocess the image: resize and performs mean subtraction
    detections = prepare_detections(input_image)
    output = np.squeeze(detections)
    # get the confidence
    confidence_list = output[:, 2]
    if np.max(confidence_list) * 100 < job.threshold:
        return
    return get_box_coordinates(output[:, 3:7], job, width=width, height=height, x=confidence_list)


def _draw_box_with_text(input_image: cvt.MatLike,
                        confidence: np.float64, *,
                        x0: int,
                        y0: int,
                        x1: int,
                        y1: int) -> cvt.MatLike:
    colours = (255, 0, 0), (0, 255, 0), (0, 0, 255)
    colour = random.choice(colours)
    font_scale, line_width, text_offset = .45, 2, 10

    text = "{:.2f}%".format(confidence)
    y_text = y0 - text_offset if y0 > 20 else y0 + text_offset
    cv2.rectangle(input_image, (x0, y0), (x1, y1), colour, line_width)
    cv2.putText(input_image, text, (x0, y_text), cv2.FONT_HERSHEY_SIMPLEX, font_scale, colour, line_width)
    return input_image


def get_multi_box_parameters(input_image: cvt.MatLike) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    detections = prepare_detections(convert_color_space(input_image))
    return detections[0, 0, :, 3:7], detections[0, 0, :, 2] * 100.0


def multi_box(input_image: cvt.MatLike, job: Job) -> cvt.MatLike:
    height, width = input_image.shape[:2]
    outputs, confidences = get_multi_box_parameters(input_image)

    input_image = adjust_gamma(input_image, job.gamma)
    input_image = convert_color_space(input_image)

    # Find indexes where confidence surpasses threshold
    valid_indices = np.where(confidences > job.threshold)[0]

    for idx in valid_indices:
        x0, y0, x1, y1 = get_box_coordinates(outputs[idx], job, width=width, height=height)
        input_image = _draw_box_with_text(input_image, np.float64(confidences[idx]), x0=x0, y0=y0, x1=x1, y1=y1)

    return input_image


def multi_box_positions(input_image: cvt.MatLike,
                        job: Job) -> tuple[npt.NDArray[np.float64], c.Iterator[Box]]:
    height, width = input_image.shape[:2]
    outputs, confidences = get_multi_box_parameters(input_image)

    mask = confidences > job.threshold
    box_outputs = outputs[mask] * np.array([width, height, width, height])
    crop_positions = get_multibox_coordinates(box_outputs.astype(np.int_), job)
    return confidences, crop_positions


def box_detect(input_image: cvt.MatLike,
               job: Job) -> Optional[Box]:
    try:
        # get width and height of the image
        height, width = input_image.shape[:2]
    except AttributeError:
        return
    return box(input_image, job, width=width, height=height)


def get_first_file(img_path: Path) -> Optional[Path]:
    """
    The function retrieves the first file with a supported extension from the specified image path.

    Args:
        img_path (Path): The path to the image directory.

    Returns:
        Optional[Path]: The path to the first file with a supported extension, or None if no such file is found.

    Example:
        ```python
        from pathlib import Path

        # Defining the image path.
        img_path = Path("images")

        # Getting the first file.
        first_file = get_first_file(img_path)

        # Printing the first file.
        print(first_file)
        ```
    """

    return next((file for file in img_path.iterdir() if file.suffix.lower() in Photo.file_types), None)


def mask_extensions(file_list: npt.NDArray[np.str_]) -> tuple[npt.NDArray[np.bool_], int]:
    """
    The function masks the file list based on the file extensions and returns a tuple containing the mask array and
    the size of the masked file list.

    Args:
        file_list (npt.NDArray[np.str_]): The file list to be masked.

    Returns:
        Tuple[npt.NDArray[np.bool_], int]: A tuple containing the mask array and the size of the masked file list.

    Example:
        ```python
        import numpy as np

        # Creating the file list.
        file_list = np.array(['file1.jpg', 'file2.png', 'file3.jpg', 'file4.tif'])

        # Masking the file list.
        mask, size = mask_extensions(file_list)

        # Printing the mask and size.
        print(mask)
        print(size)
        ```
    """

    file_suffixes = np.char.lower([Path(file).suffix for file in file_list])
    mask = np.isin(file_suffixes, Photo.file_types)
    return mask, np.count_nonzero(mask)


def split_by_cpus(mask: npt.NDArray[np.bool_],
                  core_count: int,
                  *file_lists: npt.NDArray[np.str_]) -> c.Iterator[list[npt.NDArray[np.str_]]]:
    """
    The function splits the provided file lists based on the given mask and core count, and returns a generator of
    the split lists.

    Args:
        mask (npt.NDArray[np.bool_]): The mask array used for splitting.
        core_count (int): The number of cores to split the file lists.
        *file_lists (npt.NDArray[np.str_]): Variable-length argument of file lists to be split.

    Returns:
        Generator[List[npt.NDArray[np.str_]], None, None]: A generator of split file lists.

    Example:
        ```python
        import numpy as np

        # Creating the mask array.
        mask = np.array([True, False, True, False])

        # Creating the file lists.
        file_list1 = np.array(['file1.jpg', 'file2.jpg', 'file3.jpg', 'file4.jpg'])
        file_list2 = np.array(['fileA.jpg', 'fileB.jpg', 'fileC.jpg', 'fileD.jpg'])

        # Splitting the file lists.
        split_lists = split_by_cpus(mask, 2, file_list1, file_list2)

        # Printing the split lists.
        for split_list in split_lists:
            print(split_list)
        ```
    """
    return map(lambda x: np.array_split(x[mask], core_count), file_lists)


def join_path_suffix(file_str: str, destination: Path) -> tuple[Path, bool]:
    """
    The function joins the given path and suffix and returns the resulting path.
    """
    path = destination.joinpath(file_str)
    return path, path.suffix in Photo.TIFF_TYPES


@cache
def set_filename(radio_options: tuple[str, ...], *,
                 image_path: Path,
                 destination: Path,
                 radio_choice: str,
                 new: Optional[str] = None) -> tuple[Path, bool]:
    """
    The function sets the filename and extension for the image based on the provided parameters.

    Args:
        image_path (Path): The path to the image file.
        destination (Path): The destination folder path.
        radio_choice (str): The selected radio choice.
        radio_options (Tuple[str, ...]): The available radio options.
        new (Optional[str], optional): The new filename. Defaults to None.

    Returns:
        Tuple[Path, bool]: A tuple containing the final path and a flag indicating whether the extension is TIFF.

    Example:
        ```python
        from pathlib import Path
        from typing import Tuple

        # Defining the image path and destination.
        image_path = Path("image.jpg")
        destination = Path("output")

        # Defining the radio choice and options.
        radio_choice = "jpg"
        radio_options = ("jpg", "png", "tiff")

        # Setting the filename and extension.
        final_path, is_tiff = set_filename(image_path, destination, radio_choice, radio_options)

        # Printing the final path and is_tiff flag.
        print(final_path)
        print(is_tiff)
        ```
    """

    if (suffix := image_path.suffix.lower()) in Photo.RAW_TYPES:
        selected_extension = radio_options[2] if radio_choice == radio_options[0] else radio_choice
    else:
        selected_extension = suffix if radio_choice == radio_options[0] else radio_choice
    return join_path_suffix(f'{new or image_path.stem}{selected_extension}', destination)


def reject(*, path: Path,
           destination: Path) -> None:
    """
    The function moves the specified image file to a "rejects" folder within the destination folder.

    Args:
        path (Path): The path to the image file.
        destination (Path): The destination folder path.

    Returns:
        None

    Example:
        ```python
        from pathlib import Path

        # Defining the path and destination.
        path = Path("image.jpg")
        destination = Path("output")

        # Rejecting the image.
        reject(path, destination)
        ```
    """

    reject_folder = destination.joinpath('rejects')
    reject_folder.mkdir(exist_ok=True)
    shutil.copy(path, reject_folder.joinpath(path.name))


def save_image(image: cvt.MatLike,
               file_path: Path,
               user_gam: Union[int, float],
               is_tiff: bool = False) -> None:
    """
    The function saves the provided image to the specified file path.

    Args:
        image (cvt.MatLike): The image to be saved.
        file_path (Path): The path to save the image file.
        user_gam (Union[int, float]): The gamma value for the image.
        is_tiff (bool, optional): Flag indicating whether to save the image as a TIFF file. Defaults to False.

    Returns:
        None

    Example:
        ```python
        from pathlib import Path
        from autocrop import cvt

        # Creating an image.
        image = cvt.MatLike()

        # Defining the file path.
        file_path = Path("image.jpg")

        # Saving the image.
        save_image(image, file_path, user_gam=2.2)
        ```
    """

    if is_tiff:
        image = cv2.convertScaleAbs(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tiff.imwrite(file_path, image)
    else:
        cv2.imwrite(file_path.as_posix(), cv2.LUT(image, gamma(user_gam * GAMMA_THRESHOLD)))


def multi_save_image(cropped_images: c.Iterator[cvt.MatLike],
                     file_path: Path,
                     gamma_value: int,
                     is_tiff: bool) -> None:
    """
    The function saves multiple cropped images to the specified file path with incremental file names.

    Args:
        cropped_images (List[cvt.MatLike]): The list of cropped images to be saved.
        file_path (Path): The path to save the images.
        gamma_value (int): The gamma value for the images.
        is_tiff (bool): Flag indicating whether to save the images as TIFF files.

    Returns:
        None

    Example:
        ```python
        from pathlib import Path
        from typing import List
        from autocrop import cvt

        # Creating a list of cropped images
        cropped_images = [cvt.MatLike(), cvt.MatLike(), cvt.MatLike()]

        # Defining the file path
        file_path = Path("cropped_images")

        # Saving the cropped images
        multi_save_image(cropped_images, file_path, gamma_value=2, is_tiff=False)
        ```
    """

    for i, image in enumerate(cropped_images):
        new_file_path = file_path.with_stem(f'{file_path.stem}_{i}')
        save_image(image, new_file_path, gamma_value, is_tiff)


def get_frame_path(destination: Path,
                   file_enum: str,
                   job: Job) -> tuple[Path, bool]:
    file_str = f'{file_enum}.jpg' if job.radio_choice() == job.radio_options[0] else file_enum + job.radio_choice()
    return join_path_suffix(file_str, destination)


def save_detection(source_image: Path,
                   job: Job,
                   face_detection_tools: FaceToolPair,
                   crop_function: CropFunction,
                   save_function: SaveFunction,
                   new: Optional[str] = None) -> None:
    """
    The function saves the cropped images obtained from the `crop_function` using the `save_function` based on the
    provided job parameters.

    Args: source_image (Path): The path to the source image file. job (Job): The job object containing additional
    parameters. face_detection_tools ( Tuple[Any, Any]): The face worker object used for face detection.
    crop_function (Callable[[Union[cvt.MatLike, Path], Job,  Tuple[Any, Any]], Optional[Union[cvt.MatLike,
    Generator[cvt.MatLike, None, None]]]]): The function used for cropping the image. save_function (Callable[[Any,
    Path, int, bool], None]): The function used for saving the cropped images. image_name (Optional[Path],
    optional): The name of the image file. Defaults to None. new (Optional[str], optional): The name of the new image
    file. Defaults to None.

    Returns:
        None

    Example:
        ```python
        from pathlib import Path
        from autocrop import Job,  Tuple[Any, Any]

        # Creating a job object.
        job = Job()

        # Creating a face worker object.
        face_detection_tools =  Tuple[Any, Any]()

        # Defining the crop function.
        def crop_function(image, job, face_detection_tools):
            # Crop implementation

        # Defining the save function.
        def save_function(cropped_images, file_path, gamma, is_tiff):
            # Save implementation

        # Saving the cropped images.
        source_image = Path("image.jpg")
        save_detection(source_image, job, face_detection_tools, crop_function, save_function)
        ```
    """

    if (destination_path := job.get_destination()) is None:
        # print(f'{source_image} is not a valid image.')
        return

    if (cropped_images := crop_function(source_image, job, face_detection_tools)) is None:
        # print(f'{source_image} is no face')
        reject(path=source_image, destination=destination_path)
        return

    # print(f'{source_image} is a face')
    file_path, is_tiff = set_filename(job.radio_tuple(),
                                      image_path=source_image, destination=destination_path,
                                      radio_choice=job.radio_choice(), new=new)
    # print(f'{source_image} is cropped')
    save_function(cropped_images, file_path, job.gamma, is_tiff)
    # print(f'{source_image} is saved')


def crop_image(input_image: Union[Path, cvt.MatLike],
               job: Job,
               face_detection_tools: FaceToolPair) -> Optional[cvt.MatLike]:
    """
    The function crops an image based on the provided job parameters and returns the cropped image.

    Args:
        input_image (Union[Path, cvt.MatLike]): The image to be cropped, either as a file path or a cvt.MatLike object.
        job (Job): The job object containing additional parameters for cropping.
        face_detection_tools ( Tuple[Any, Any]): The face worker object used for face detection.

    Returns:
        Optional[cvt.MatLike]: The cropped image as a cvt.MatLike object, or None if cropping fails.

    Example:
        ```python
        from pathlib import Path
        from autocrop import Job,  Tuple[Any, Any]

        # Creating a job object.
        job = Job()

        # Creating a face worker object.
        face_detection_tools =  Tuple[Any, Any]()

        # Cropping an image.
        image_path = Path("image.jpg")
        cropped_image = crop_image(image_path, job, face_detection_tools)

        # Displaying the cropped image.
        cvt.imshow(cropped_image)
        ```
    """

    pic_array = open_pic(input_image, face_detection_tools, exposure=job.fix_exposure_job,
                         tilt=job.auto_tilt_job) \
        if isinstance(input_image, Path) else input_image

    if pic_array is None:
        return

    if (bounding_box := box_detect(pic_array, job)) is None:
        return

    cropped_pic = numpy_array_crop(pic_array, bounding_box)
    result = convert_color_space(cropped_pic) if len(cropped_pic.shape) >= 3 else cropped_pic
    return cv2.resize(result, job.size, interpolation=cv2.INTER_AREA)


def process_image(image: cvt.MatLike,
                  job: Job,
                  crop_position: Box) -> cvt.MatLike:
    cropped_image = Image.fromarray(image).crop(crop_position)
    image_array = np.array(cropped_image)
    color_converted = convert_color_space(image_array)
    return cv2.resize(
        src=color_converted, dsize=job.size, interpolation=cv2.INTER_AREA
    )


def multi_crop(source_image: Union[cvt.MatLike, Path],
               job: Job,
               face_detection_tools: FaceToolPair) -> Optional[c.Iterator[cvt.MatLike]]:
    """
    The function takes a source image, a job, and a face worker as input parameters. It returns a generator that
    yields cropped images of faces detected in the source image.

    Args:
        source_image (Union[cvt.MatLike, Path]): The source image from which to extract cropped images.
        job (Job): The job object containing additional parameters for cropping.
        face_detection_tools ( Tuple[Any, Any]): The face worker object used for face detection.

    Returns:
        Optional[Generator[cvt.MatLike, None, None]]: A generator that yields cropped images of faces.

    Examples:
        ```python
        source_image = cvt.MatLike()
        job = Job()
        face_detection_tools =  Tuple[Any, Any]()

        # Generating cropped images.
        cropped_images = multi_crop(source_image, job, face_detection_tools)

        # Printing the cropped images.
        for image in cropped_images:
            print(image)
        ```
    """

    img = open_pic(source_image, face_detection_tools, exposure=job.fix_exposure_job, tilt=job.auto_tilt_job) \
        if isinstance(source_image, Path) else source_image

    if img is None:
        return

    confidences, crop_positions = multi_box_positions(img, job)
    # Check if any faces were detected
    if np.any(confidences > job.threshold):
        # Cropped images
        return map(lambda x: process_image(img, job, x), crop_positions)
    else:
        return


def get_crop_save_functions(job: Job) -> tuple[CropFunction, SaveFunction]:
    return (multi_crop, multi_save_image) if job.multi_face_job else (crop_image, save_image)


def crop(input_image: Union[Path, str],
         job: Job,
         face_detection_tools: FaceToolPair,
         new: Optional[str] = None) -> None:
    """
    Performs cropping and saves the cropped image based on job parameters.
    """
    crop_fn, save_fn = get_crop_save_functions(job)

    if all(x is not None for x in [job.table, job.folder_path, new]):
        source = Path(input_image) if isinstance(input_image, str) else input_image
        save_detection(source, job, face_detection_tools, crop_fn, save_fn, new)
    elif job.folder_path is not None:
        source = job.folder_path / input_image.name
        save_detection(source, job, face_detection_tools, crop_fn, save_fn)
    else:
        save_detection(input_image, job, face_detection_tools, crop_fn, save_fn)


def frame_save(cropped_image_data: cvt.MatLike,
               file_enum_str: str,
               destination: Path,
               job: Job) -> None:
    """
    Saves the cropped image data to a file.

    Args:
        cropped_image_data (cvt.MatLike): The cropped image data.
        file_enum_str (str): The file enumeration string.
        destination (Path): The destination path to save the file.
        job (Job): The job containing the parameters for saving.

    Returns:
        None
    """

    file_path, is_tiff = get_frame_path(destination, file_enum_str, job)
    save_image(cropped_image_data, file_path, job.gamma, is_tiff)


def grab_frame(position_slider: int,
               video_line_edit: str) -> Optional[cvt.MatLike]:
    """
    Grabs a frame from a video at the specified position.

    Args:
        position_slider (int): The position of the frame in milliseconds.
        video_line_edit (str): The path to the video file.

    Returns:
        Optional[cvt.MatLike]: The grabbed frame, or None if the frame could not be grabbed.
    """

    # Set video frame position to timelineSlider value
    cap = cv2.VideoCapture(video_line_edit)
    cap.set(cv2.CAP_PROP_POS_MSEC, position_slider)
    # Read frame from video capture object
    ret, frame = cap.read()
    if not ret:
        return
    cap.release()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
