from typing import ClassVar, Tuple

import cv2
import dlib


class FaceWorker:
    """
A class for performing face-related tasks.

Attributes:
    LANDMARKS (ClassVar[str]): The path to the shape predictor model file.
    PROTOTXT (ClassVar[str]): The path to the Caffe model prototxt file.
    CAFFEMODEL (ClassVar[str]): The path to the Caffe model weights file.

Methods:
    caffe_model(cls) -> cv2.dnn.Net: Returns a pre-trained Caffe model for face detection.
    worker_tuple(cls) -> Tuple[dlib.fhog_object_detector, dlib.shape_predictor]: Returns a tuple containing the frontal face detector and shape predictor.

Example:
    ```python
    worker = FaceWorker()

    # Getting the Caffe model
    model = worker.caffe_model()
    print(model)

    # Getting the worker tuple
    worker_tuple = worker.worker_tuple()
    print(worker_tuple)
    ```
"""

    LANDMARKS: ClassVar[str] = 'resources\\models\\shape_predictor_68_face_landmarks.dat'
    PROTOTXT: ClassVar[str] = 'resources\\weights\\deploy.prototxt.txt'
    CAFFEMODEL: ClassVar[str] = 'resources\\models\\res10_300x300_ssd_iter_140000.caffemodel'

    @classmethod
    def caffe_model(cls) -> cv2.dnn.Net:
        """
Returns a pre-trained Caffe model for face detection.

Returns:
    cv2.dnn.Net: The pre-trained Caffe model for face detection.
"""

        return cv2.dnn.readNetFromCaffe(cls.PROTOTXT, cls.CAFFEMODEL)

    @classmethod
    def worker_tuple(cls) -> Tuple[dlib.fhog_object_detector, dlib.shape_predictor]:
        """
Returns a tuple containing the frontal face detector and shape predictor for face detection and landmark localization.

Returns:
    Tuple[dlib.fhog_object_detector, dlib.shape_predictor]: A tuple containing the frontal face detector and shape predictor.
"""

        return dlib.get_frontal_face_detector(), dlib.shape_predictor(cls.LANDMARKS)
