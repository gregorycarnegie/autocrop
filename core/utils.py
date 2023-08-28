import cProfile
import pstats
import random
import shutil
from functools import cache, wraps
from pathlib import Path
from typing import Any, Callable, Generator, Iterator, List, Optional, Tuple, Union

import autocrop_rs
import cv2
import cv2.typing as cvt
import numpy as np
import numpy.typing as npt
import pandas as pd
import rawpy
import tifffile as tiff
from PIL import Image

from . import window_functions as wf
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
    """
The function takes an image in Pillow format as input and converts it to a NumPy array.

Args:
    image (Image.Image): The image in Pillow format to be converted.

Returns:
    npt.NDArray[np.uint8]: The converted image as a NumPy array.

Example:
    ```python
    from PIL import Image

    # Creating a Pillow image
    image = Image.open("image.jpg")

    # Converting the image to a NumPy array
    numpy_array = pillow_to_numpy(image)

    # Printing the shape of the NumPy array
    print(numpy_array.shape)
    ```
"""

    return np.frombuffer(image.tobytes(), dtype=np.uint8).reshape((image.size[1], image.size[0], len(image.getbands())))


@cache
def gamma(gam: Union[int, float] = 1.0) -> npt.NDArray[np.generic]:
    """
The function applies a gamma correction to an array of intensity values ranging from 0 to 255. It returns the corrected array.

Args:
    gam (Union[int, float], optional): The gamma value for the correction. Defaults to 1.0.

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

    return np.power(np.arange(256) / 255, 1.0 / gam) * 255 if gam != 1.0 else np.arange(256)


def adjust_gamma(image: Union[cvt.MatLike, npt.NDArray[np.uint8]], gam: int) -> cvt.MatLike:
    """
The function adjusts the gamma of the provided image using a lookup table.

Args:
    image (Union[cvt.MatLike, npt.NDArray[np.uint8]]): The image to adjust the gamma.
    gam (int): The gamma value.

Returns:
    cvt.MatLike: The image with the gamma adjusted.

Example:
    ```python
    from autocrop import cvt

    # Creating an image
    image = cvt.MatLike()

    # Adjusting the gamma of the image
    adjusted_image = adjust_gamma(image, gam=2)
    ```
"""

    return cv2.LUT(image, gamma(gam * GAMMA_THRESHOLD).astype('uint8'))


def convert_color_space(image: Union[cvt.MatLike, npt.NDArray[np.uint8]]) -> cvt.MatLike:
    """
The function converts the color space of the provided image from BGR to RGB.

Args:
    image (Union[cvt.MatLike, npt.NDArray[np.uint8]]): The image to convert.

Returns:
    cvt.MatLike: The image with the color space converted to RGB.

Example:
    ```python
    from autocrop import cvt

    # Creating an image
    image = cvt.MatLike()

    # Converting the color space of the image
    converted_image = convert_color_space(image)
    ```
"""

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def numpy_array_crop(image: cvt.MatLike, bounding_box: Tuple[int, int, int, int]) -> npt.NDArray[np.uint8]:
    """
The function crops the provided image using the specified bounding box and returns the cropped image as a NumPy array.

Args:
    image (cvt.MatLike): The image to be cropped.
    bounding_box (Tuple[int, int, int, int]): The bounding box coordinates (left, upper, right, lower) for cropping.

Returns:
    npt.NDArray[np.uint8]: The cropped image as a NumPy array.

Example:
    ```python
    from autocrop import cvt

    # Creating an image
    image = cvt.MatLike()

    # Defining the bounding box
    bounding_box = (100, 100, 300, 300)

    # Cropping the image
    cropped_image = numpy_array_crop(image, bounding_box)
    ```
"""

    # Load and crop image using PIL
    picture = Image.fromarray(image).crop(bounding_box)
    return pillow_to_numpy(picture)


def crop_and_set(image: cvt.MatLike,
                 bounding_box: Tuple[int, int, int, int],
                 gamma_value: int,
                 image_widget: ImageWidget) -> None:
    try:
        cropped_image = numpy_array_crop(image, bounding_box)
        adjusted_image = adjust_gamma(cropped_image, gamma_value)
        final_image = convert_color_space(adjusted_image)
    except (cv2.error, Image.DecompressionBombError):
        return None
    wf.display_image_on_widget(final_image, image_widget)


def correct_exposure(image: Union[cvt.MatLike, npt.NDArray[np.uint8]],
                     exposure: bool) -> cvt.MatLike:
    """
The function corrects the exposure of the provided image using histogram equalization.

Args:
    image (Union[cvt.MatLike, npt.NDArray[np.uint8]]): The image to correct the exposure.
    exposure (bool): Flag indicating whether to perform exposure correction.

Returns:
    cvt.MatLike: The image with corrected exposure.

Example:
    ```python
    from autocrop import cvt

    # Creating an image
    image = cvt.MatLike()

    # Correcting the exposure of the image
    corrected_image = correct_exposure(image, exposure=True)
    ```
"""

    if not exposure: return image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
    # Grayscale histogram
    hist: npt.NDArray[np.generic] = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    # Calculate alpha and beta
    alpha, beta = autocrop_rs.calc_alpha_beta(hist)
    return cv2.convertScaleAbs(src=image, alpha=alpha, beta=beta)


def rotate_image(image: Union[cvt.MatLike, npt.NDArray[np.uint8]],
                 angle: float,
                 center: Tuple[float, float]) -> cvt.MatLike:
    """
The function rotates the provided image by the specified angle around the given center.

Args:
    image (Union[cvt.MatLike, npt.NDArray[np.uint8]]): The image to rotate.
    angle (float): The angle of rotation in degrees.
    center (Tuple[float, float]): The center point of rotation.

Returns:
    cvt.MatLike: The rotated image.

Example:
    ```python
    from autocrop import cvt

    # Creating an image
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
    return cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))


def align_head(image: Union[cvt.MatLike, npt.NDArray[np.uint8]],
               face_worker: FaceWorker,
               tilt: bool) -> cvt.MatLike:
    """
The function aligns the head in the provided image using facial landmarks and tilt correction.

Args:
    image (Union[cvt.MatLike, npt.NDArray[np.uint8]]): The image to align the head.
    face_worker (FaceWorker): The face worker object for face detection and landmark prediction.
    tilt (bool): Flag indicating whether to perform tilt correction.

Returns:
    cvt.MatLike: The aligned image.

Example:
    ```python
    from autocrop import cvt, FaceWorker

    # Creating an image
    image = cvt.MatLike()

    # Creating a face worker
    face_worker = FaceWorker()

    # Aligning the head in the image
    aligned_image = align_head(image, face_worker, tilt=True)
    ```
"""

    if not tilt:
        return image
    height, _ = image.shape[:2]
    scaling_factor = 256 / height
    image_array = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    # Detect the faces in the image.
    if len(image_array.shape) >= 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

    face_detector, predictor = face_worker.worker_tuple()

    faces = face_detector(image_array, 1)
    # If no faces are detected, return the original image.
    if len(faces) == 0:
        return image

    # Find the face with the highest confidence score.
    face = max(faces, key=lambda x: x.area())
    # Get the 68 facial landmarks for the face.
    landmarks = predictor(image_array, face)
    landmarks_array = np.array([(p.x, p.y) for p in landmarks.parts()])
    # Find the angle of the tilt and the center of the face
    angle, center_x, center_y = get_angle_of_tilt(landmarks_array, scaling_factor)
    return rotate_image(image, angle, (center_x, center_y))


def open_image(image: Path,
               face_worker: FaceWorker, *,
               exposure: bool,
               tilt: bool) -> cvt.MatLike:
    """
The function opens an image file using `cv2` and performs color conversion, exposure correction, and head alignment using the provided `FaceWorker` object.

Args:
    image (Path): The path to the image file.
    face_worker (FaceWorker): The FaceWorker object used for aligning the head.
    exposure (bool): Flag indicating whether to correct the exposure.
    tilt (bool): Flag indicating whether to align the head.

Returns:
    cvt.MatLike: The processed image data.

Example:
    ```python
    # Opening an image file
    image_path = Path('/path/to/image.jpg')
    face_worker = FaceWorker()
    processed_image = open_image(image_path, face_worker, exposure=True, tilt=True)
    cv2.imshow('Processed Image', processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ```
"""

    img = cv2.imread(image.as_posix())
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = correct_exposure(img, exposure)
    return align_head(img, face_worker, tilt)


def open_raw(image: Path,
             face_worker: FaceWorker, *,
             exposure: bool,
             tilt: bool) -> cvt.MatLike:
    """
The function opens a raw image file using `rawpy` and performs post-processing on the raw image data. It corrects the exposure and aligns the head using the provided `FaceWorker` object.

Args:
    image (Path): The path to the raw image file.
    face_worker (FaceWorker): The FaceWorker object used for aligning the head.
    exposure (bool): Flag indicating whether to correct the exposure.
    tilt (bool): Flag indicating whether to align the head.

Returns:
    cvt.MatLike: The processed image data.

Example:
    ```python
    # Opening a raw image file
    image_path = Path('/path/to/image.CR2')
    face_worker = FaceWorker()
    processed_image = open_raw(image_path, face_worker, exposure=True, tilt=True)
    cv2.imshow('Processed Image', processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ```
"""

    with rawpy.imread(image.as_posix()) as raw:
        # Post-process the raw image data
        img = raw.postprocess(use_camera_wb=True)
        img = correct_exposure(img, exposure)
        return align_head(img, face_worker, tilt)


def open_table(input_file: Path) -> pd.DataFrame:
    """
The function opens a table file and returns its contents as a `pd.DataFrame` object. If the file has a `.csv` extension, the function uses `pd.read_csv` to read the file. Otherwise, it uses `pd.read_excel`.

Args:
    input_file (Path): The path to the table file.

Returns:
    pd.DataFrame: The contents of the table file as a DataFrame.

Example:
    ```python
    # Opening a CSV file
    csv_file = Path('/path/to/table.csv')
    csv_data = open_table(csv_file)
    print(csv_data.head())

    # Opening an Excel file
    excel_file = Path('/path/to/table.xlsx')
    excel_data = open_table(excel_file)
    print(excel_data.head())
    ```
"""

    return pd.read_csv(input_file) if input_file.suffix.lower() == '.csv' else pd.read_excel(input_file)


def open_pic(input_file: Union[Path, str],
             face_worker: FaceWorker, *,
             exposure: bool,
             tilt: bool) -> Optional[cvt.MatLike]:
    """
The function opens an image file based on its extension. If the extension is in the list of supported CV2 types, the function calls `open_image` to open the image using `cv2`. If the extension is in the list of supported RAW types, the function calls `open_raw` to open the raw image using `rawpy`. If the extension is not supported, the function returns None.

Args:
    input_file (Union[Path, str]): The path or string representing the input image file.
    face_worker (FaceWorker): The FaceWorker object used for aligning the head.
    exposure (bool): Flag indicating whether to correct the exposure.
    tilt (bool): Flag indicating whether to align the head.

Returns:
    Optional[cvt.MatLike]: The processed image data, or None if the extension is not supported.

Example:
    ```python
    # Opening an image file
    image_path = Path('/path/to/image.jpg')
    face_worker = FaceWorker()
    processed_image = open_pic(image_path, face_worker, exposure=True, tilt=True)
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
            return open_image(input_file, face_worker, exposure=exposure, tilt=tilt)
        case extension if extension in Photo.RAW_TYPES:
            return open_raw(input_file, face_worker, exposure=exposure, tilt=tilt)
        case _:
            return None


def get_angle_of_tilt(landmarks_array: npt.NDArray[np.int_], scaling_factor: float) -> Tuple[float, float, float]:
    # Find the eyes in the image (l_eye - r_eye).
    eye_diff = np.mean(landmarks_array[L_EYE_START:L_EYE_END], axis=0) - \
               np.mean(landmarks_array[R_EYE_START:R_EYE_END], axis=0)
    # Find the center of the face.
    center_x, center_y = np.mean(landmarks_array[R_EYE_START:L_EYE_END], axis=0) / scaling_factor
    return np.arctan2(eye_diff[1], eye_diff[0]) * 180 / np.pi, center_x, center_y


def prepare_detections(image: cvt.MatLike,
                       face_worker: FaceWorker) -> npt.NDArray[np.float64]:
    # Create blob from image
    # We standardize the image by scaling it and then subtracting the mean RGB values
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)
    # Set the input for the neural network
    caffe = face_worker.caffe_model()
    caffe.setInput(blob)
    # Forward pass through the network to get detections
    return np.array(caffe.forward())


def get_box_coordinates(output: Union[cvt.MatLike, npt.NDArray[np.generic]],
                        job: Job, *,
                        width: int,
                        height: int,
                        x: Optional[npt.NDArray[np.generic]] = None) -> Tuple[int, int, int, int]:
    box_outputs = output * np.array([width, height, width, height])
    x0, y0, x1, y1 = box_outputs.astype(np.int_) if x is None else box_outputs[np.argmax(x)]
    return autocrop_rs.crop_positions(x0, y0, x1 - x0, y1 - y0, job.face_percent, job.width,
                                      job.height, job.top, job.bottom, job.left, job.right)


def box(img: cvt.MatLike,
        job: Job,
        face_worker: FaceWorker, *,
        width: int,
        height: int,) -> Optional[Tuple[int, int, int, int]]:
    # preprocess the image: resize and performs mean subtraction
    detections = prepare_detections(img, face_worker)
    output = np.squeeze(detections)
    # get the confidence
    confidence_list = output[:, 2]
    if np.max(confidence_list) * 100 < job.threshold:
        return None
    return get_box_coordinates(output[:, 3:7], job, width=width, height=height, x=confidence_list)


def _draw_box_with_text(image: cvt.MatLike, confidence: np.float64, *, x0: int, y0: int, x1: int, y1: int) -> cvt.MatLike:
    COLOURS = (255, 0, 0), (0, 255, 0), (0, 0, 255)
    colour = random.choice(COLOURS)
    FONT_SCALE, LINE_WIDTH, TEXT_OFFSET = .45, 2, 10

    text = "{:.2f}%".format(confidence)
    y_text = y0 - TEXT_OFFSET if y0 > 20 else y0 + TEXT_OFFSET
    cv2.rectangle(image, (x0, y0), (x1, y1), colour, LINE_WIDTH)
    cv2.putText(image, text, (x0, y_text), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, colour, LINE_WIDTH)
    return image


def get_detection_arrays(detections: npt.NDArray[np.float64]) -> Iterator[Tuple[np.float64, npt.NDArray[np.float64]]]:
    """
The function takes an array of detections as input and returns an iterator that yields tuples of confidence values and output arrays.

Args:
    detections (npt.NDArray[np.float64]): The array of detections.

Returns:
    Iterator[Tuple[np.float64, npt.NDArray[np.float64]]]: An iterator that yields tuples of confidence values and output arrays.

Example:
    ```python
    detections = np.array([[1.0, 2.0, 0.8, 0.1, 0.2, 0.3]])
    
    # Generating detection arrays
    detection_arrays = get_detection_arrays(detections)
    
    # Printing the confidence values and output arrays
    for conf, output in detection_arrays:
        print(f"Confidence: {conf}, Output: {output}")
    ```
"""

    x = range(detections.shape[2])
    conf_array: Generator[np.float64, None, None] = (detections[0, 0, i, 2] * 100 for i in x)
    output_array: Generator[npt.NDArray[np.float64], None, None] = (detections[0, 0, i, 3:7] for i in x)
    return zip(conf_array, output_array)


def get_multi_box_parameters(image: cvt.MatLike,
                             face_worker: FaceWorker) -> Tuple[npt.NDArray[np.float64], int, int]:
    height, width = image.shape[:2]
    detections = prepare_detections(convert_color_space(image), face_worker)
    return detections, height, width


def multi_box(image: cvt.MatLike,
              job: Job,
              face_worker: FaceWorker) -> cvt.MatLike:
    detections, height, width = get_multi_box_parameters(image, face_worker)

    image = adjust_gamma(image, job.gamma)
    image = convert_color_space(image)

    for confidence, output in get_detection_arrays(detections):
        if confidence > job.threshold:
            x0, y0, x1, y1 = get_box_coordinates(output, job, width=width, height=height)
            image = _draw_box_with_text(image, confidence, x0=x0, y0=y0, x1=x1, y1=y1)
    return image


def multi_box_positions(image: cvt.MatLike,
                        job: Job,
                        face_worker: FaceWorker) -> Tuple[npt.NDArray[np.float_], npt.NDArray[np.int_]]:
    detections, height, width = get_multi_box_parameters(image, face_worker)

    crop_positions = [get_box_coordinates(output, job, width=width, height=height)
                      for confidence, output in get_detection_arrays(detections)
                      if confidence > job.threshold]
    return np.array(detections), np.array(crop_positions)


def box_detect(img: cvt.MatLike,
               job: Job,
               face_worker: FaceWorker) -> Optional[Tuple[int, int, int, int]]:
    try:
        # get width and height of the image
        height, width = img.shape[:2]
    except AttributeError:
        return None
    return box(img, job, face_worker, width=width, height=height)


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

    # Defining the image path
    img_path = Path("images")

    # Getting the first file
    first_file = get_first_file(img_path)

    # Printing the first file
    print(first_file)
    ```
"""

    files = np.fromiter(img_path.iterdir(), Path)
    file: Generator[Path, None, None] = (pic for pic in files if pic.suffix.lower() in Photo.file_types)
    return next(file, None)


def mask_extensions(file_list: npt.NDArray[np.str_]) -> Tuple[npt.NDArray[np.bool_], int]:
    """
The function masks the file list based on the file extensions and returns a tuple containing the mask array and the size of the masked file list.

Args:
    file_list (npt.NDArray[np.str_]): The file list to be masked.

Returns:
    Tuple[npt.NDArray[np.bool_], int]: A tuple containing the mask array and the size of the masked file list.

Example:
    ```python
    import numpy as np

    # Creating the file list
    file_list = np.array(['file1.jpg', 'file2.png', 'file3.jpg', 'file4.tif'])

    # Masking the file list
    mask, size = mask_extensions(file_list)

    # Printing the mask and size
    print(mask)
    print(size)
    ```
"""

    mask: npt.NDArray[np.bool_] = np.in1d(np.char.lower([Path(file).suffix for file in file_list]), Photo.file_types)
    return mask, file_list[mask].size


def split_by_cpus(mask: npt.NDArray[np.bool_],
                  core_count: int,
                  *file_lists: npt.NDArray[np.str_]) -> Generator[List[npt.NDArray[np.str_]], None, None]:
    """
The function splits the provided file lists based on the given mask and core count, and returns a generator of the split lists.

Args:
    mask (npt.NDArray[np.bool_]): The mask array used for splitting.
    core_count (int): The number of cores to split the file lists.
    *file_lists (npt.NDArray[np.str_]): Variable-length argument of file lists to be split.

Returns:
    Generator[List[npt.NDArray[np.str_]], None, None]: A generator of split file lists.

Example:
    ```python
    import numpy as np

    # Creating the mask array
    mask = np.array([True, False, True, False])

    # Creating the file lists
    file_list1 = np.array(['file1.jpg', 'file2.jpg', 'file3.jpg', 'file4.jpg'])
    file_list2 = np.array(['fileA.jpg', 'fileB.jpg', 'fileC.jpg', 'fileD.jpg'])

    # Splitting the file lists
    split_lists = split_by_cpus(mask, 2, file_list1, file_list2)

    # Printing the split lists
    for split_list in split_lists:
        print(split_list)
    ```
"""

    return (np.array_split(file_list[mask], core_count) for file_list in file_lists)


@cache
def set_filename(image_path: Path,
                 destination: Path,
                 radio_choice: str,
                 radio_options: Tuple[str, ...],
                 new: Optional[str] = None) -> Tuple[Path, bool]:
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

    # Defining the image path and destination
    image_path = Path("image.jpg")
    destination = Path("output")

    # Defining the radio choice and options
    radio_choice = "jpg"
    radio_options = ("jpg", "png", "tiff")

    # Setting the filename and extension
    final_path, is_tiff = set_filename(image_path, destination, radio_choice, radio_options)

    # Printing the final path and is_tiff flag
    print(final_path)
    print(is_tiff)
    ```
"""

    if (suffix := image_path.suffix.lower()) in Photo.RAW_TYPES:
        selected_extension = radio_options[2] if radio_choice == radio_options[0] else radio_choice
    else:
        selected_extension = suffix if radio_choice == radio_options[0] else radio_choice
    final_path = destination.joinpath(f'{new or image_path.stem}{selected_extension}')
    return final_path, final_path.suffix in {'.tif', '.tiff'}


def reject(path: Path,
           destination: Path,
           image: Path) -> None:
    """
The function moves the specified image file to a "rejects" folder within the destination folder.

Args:
    path (Path): The path to the image file.
    destination (Path): The destination folder path.
    image (Path): The name of the image file.

Returns:
    None

Example:
    ```python
    from pathlib import Path

    # Defining the path and destination
    path = Path("image.jpg")
    destination = Path("output")

    # Rejecting the image
    reject(path, destination, image=path.name)
    ```
"""

    reject_folder = destination.joinpath('rejects')
    reject_folder.mkdir(exist_ok=True)
    shutil.copy(path, reject_folder.joinpath(image.name))


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

    # Creating an image
    image = cvt.MatLike()

    # Defining the file path
    file_path = Path("image.jpg")

    # Saving the image
    save_image(image, file_path, user_gam=2.2)
    ```
"""

    if is_tiff:
        image = cv2.convertScaleAbs(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tiff.imwrite(file_path, image)
    else:
        cv2.imwrite(file_path.as_posix(), cv2.LUT(image, gamma(user_gam * GAMMA_THRESHOLD)))


def multi_save_image(cropped_images: List[cvt.MatLike],
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
                   job: Job) -> Tuple[Path, bool]:
    file_str = f'{file_enum}.jpg' if job.radio_choice() == job.radio_options[0] else file_enum + job.radio_choice()
    file_path = destination.joinpath(file_str)
    return file_path, file_path.suffix in {'.tif', '.tiff'}


def save_detection(source_image: Path,
                   job: Job,
                   face_worker: FaceWorker,
                   crop_function: Callable[[Union[cvt.MatLike, Path], Job, FaceWorker], Optional[
                       Union[cvt.MatLike, Generator[cvt.MatLike, None, None]]]],
                   save_function: Callable[[Any, Path, int, bool], None],
                   image_name: Optional[Path] = None,
                   new: Optional[str] = None) -> None:
    """
The function saves the cropped images obtained from the `crop_function` using the `save_function` based on the provided job parameters.

Args:
    source_image (Path): The path to the source image file.
    job (Job): The job object containing additional parameters.
    face_worker (FaceWorker): The face worker object used for face detection.
    crop_function (Callable[[Union[cvt.MatLike, Path], Job, FaceWorker], Optional[Union[cvt.MatLike, Generator[cvt.MatLike, None, None]]]]): The function used for cropping the image.
    save_function (Callable[[Any, Path, int, bool], None]): The function used for saving the cropped images.
    image_name (Optional[Path], optional): The name of the image file. Defaults to None.
    new (Optional[str], optional): The name of the new image file. Defaults to None.

Returns:
    None

Example:
    ```python
    from pathlib import Path
    from autocrop import Job, FaceWorker

    # Creating a job object
    job = Job()

    # Creating a face worker object
    face_worker = FaceWorker()

    # Defining the crop function
    def crop_function(image, job, face_worker):
        # Crop implementation

    # Defining the save function
    def save_function(cropped_images, file_path, gamma, is_tiff):
        # Save implementation

    # Saving the cropped images
    source_image = Path("image.jpg")
    save_detection(source_image, job, face_worker, crop_function, save_function)
    ```
"""

    if (destination_path := job.get_destination()) is None:
        return None

    image_name = source_image if image_name is None else image_name
    if (cropped_images := crop_function(source_image, job, face_worker)) is None:
        reject(source_image, destination_path, image_name)
        return None

    file_path, is_tiff = set_filename(image_name, destination_path, job.radio_choice(), job.radio_tuple(), new)
    save_function(cropped_images, file_path, job.gamma, is_tiff)


def crop_image(image: Union[Path, cvt.MatLike],
               job: Job,
               face_worker: FaceWorker) -> Optional[cvt.MatLike]:
    """
The function crops an image based on the provided job parameters and returns the cropped image.

Args:
    image (Union[Path, cvt.MatLike]): The image to be cropped, either as a file path or a cvt.MatLike object.
    job (Job): The job object containing additional parameters for cropping.
    face_worker (FaceWorker): The face worker object used for face detection.

Returns:
    Optional[cvt.MatLike]: The cropped image as a cvt.MatLike object, or None if cropping fails.

Example:
    ```python
    from pathlib import Path
    from autocrop import Job, FaceWorker

    # Creating a job object
    job = Job()

    # Creating a face worker object
    face_worker = FaceWorker()

    # Cropping an image
    image_path = Path("image.jpg")
    cropped_image = crop_image(image_path, job, face_worker)

    # Displaying the cropped image
    cvt.imshow(cropped_image)
    ```
"""

    pic_array = open_pic(image, face_worker, exposure=job.fix_exposure_job.isChecked(), tilt=job.auto_tilt_job.isChecked()) \
        if isinstance(image, Path) else image
    if pic_array is None: return None
    if (bounding_box := box_detect(pic_array, job, face_worker)) is None: return None
    cropped_pic = numpy_array_crop(pic_array, bounding_box)
    result = convert_color_space(cropped_pic) if len(cropped_pic.shape) >= 3 else cropped_pic
    return cv2.resize(result, job.size, interpolation=cv2.INTER_AREA)


def multi_crop(source_image: Union[cvt.MatLike, Path],
               job: Job,
               face_worker: FaceWorker) -> Optional[Generator[cvt.MatLike, None, None]]:
    """
The function takes a source image, a job, and a face worker as input parameters. It returns a generator that yields cropped images of faces detected in the source image.

Args:
    source_image (Union[cvt.MatLike, Path]): The source image from which to extract cropped images.
    job (Job): The job object containing additional parameters for cropping.
    face_worker (FaceWorker): The face worker object used for face detection.

Returns:
    Optional[Generator[cvt.MatLike, None, None]]: A generator that yields cropped images of faces.

Examples:
    ```python
    source_image = cvt.MatLike()
    job = Job()
    face_worker = FaceWorker()

    # Generating cropped images
    cropped_images = multi_crop(source_image, job, face_worker)

    # Printing the cropped images
    for image in cropped_images:
        print(image)
    ```
"""

    img = open_pic(source_image, face_worker, exposure=job.fix_exposure_job.isChecked(), tilt=job.auto_tilt_job.isChecked()) \
        if isinstance(source_image, Path) else source_image
    if img is None:
        return None

    detections, crop_positions = multi_box_positions(img, job, face_worker)
    # Check if any faces were detected
    x = 100 * detections[0, 0, :, 2] > job.threshold
    if not x.any():
        return None

    # Cropped images
    images = (Image.fromarray(img).crop(crop_position) for crop_position in crop_positions)
    # images as numpy arrays
    image_array: Generator[npt.NDArray[np.uint8], None, None] = (pillow_to_numpy(image) for image in images)
    # numpy arrays with colour space converted
    results: Generator[cvt.MatLike, None, None] = (convert_color_space(array) for array in image_array)
    # return resized results
    return (cv2.resize(src=result, dsize=job.size, interpolation=cv2.INTER_AREA) for result in results)


def crop(image: Path,
         job: Job,
         face_worker: FaceWorker,
         new: Optional[str] = None) -> None:
    """
The function performs cropping of an image based on the provided job parameters and saves the cropped image.

Args:
    image (Path): The path to the image file to be cropped.
    job (Job): The job object containing additional parameters for cropping.
    face_worker (FaceWorker): The face worker object used for face detection.
    new (Optional[str], optional): The name of the new cropped image file. Defaults to None.

Returns:
    None

Example:
    ```python
    from pathlib import Path
    from autocrop import Job, FaceWorker

    # Creating a job object
    job = Job()

    # Creating a face worker object
    face_worker = FaceWorker()

    # Cropping an image
    image_path = Path("image.jpg")
    crop(image_path, job, face_worker, new="cropped_image.jpg")
    ```
"""

    if job.table is not None and job.folder_path is not None and new is not None:
        # Data cropping
        path = job.folder_path.joinpath(image)
        if job.multi_face_job.isChecked():
            save_detection(path, job, face_worker, multi_crop, multi_save_image, image, new)
        else:
            save_detection(path, job, face_worker, crop_image, save_image, image, new)
    elif job.folder_path is not None:
        # Folder cropping
        source, image_name = job.folder_path, image.name
        path = source.joinpath(image_name)
        if job.multi_face_job.isChecked():
            save_detection(path, job, face_worker, multi_crop, multi_save_image, Path(image_name))
        else:
            save_detection(path, job, face_worker, crop_image, save_image, Path(image_name))
    elif job.multi_face_job.isChecked():
        save_detection(image, job, face_worker, multi_crop, multi_save_image)
    else:
        save_detection(image, job, face_worker, crop_image, save_image)


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
    if not ret: return None
    cap.release()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
