import collections.abc as c
from concurrent.futures import ThreadPoolExecutor
import psutil

import dlib

from .operation_types import FaceToolPair
from .resource_path import ResourcePath

THREAD_NUMBER = min(psutil.cpu_count(), psutil.virtual_memory().total // (2 * 1024 ** 3) // 2000, 8)

LANDMARKS = ResourcePath('resources\\models\\shape_predictor_68_face_landmarks.dat').meipass_path


def _create_tool_pair() -> FaceToolPair:
    """
    Create a pair of face detection and shape prediction tools.

    Returns:
        Tuple[dlib.fhog_object_detector, dlib.shape_predictor]:
            A pair of face detection and shape prediction tools.
    """

    # Functions to create each tool
    def create_detector() -> dlib.fhog_object_detector:
        return dlib.get_frontal_face_detector()

    def create_predictor() -> dlib.shape_predictor:
        return dlib.shape_predictor(LANDMARKS)

    # Execute each function in a separate thread
    with ThreadPoolExecutor(max_workers=2) as executor:
        d_future, p_future = executor.submit(create_detector), executor.submit(create_predictor)
        # Now, wait for both tasks to complete and get the results
        detector, predictor = d_future.result(), p_future.result()

    return detector, predictor


def generate_face_detection_tools() -> c.Iterator[FaceToolPair]:
    """
    Generate a list of face detection and shape prediction tools.

    This method creates a list of tuples, where each tuple contains an instance of
    `dlib.fhog_object_detector` for frontal face detection and `dlib.shape_predictor`
    for facial landmark prediction. The number of tuples in the list is determined by
    the class attribute `THREAD_NUMBER`.

    Returns:
        List[Tuple[dlib.fhog_object_detector, dlib.shape_predictor]]:
            A list of face detection and shape prediction tools.
    """

    return (_create_tool_pair() for _ in range(THREAD_NUMBER))
