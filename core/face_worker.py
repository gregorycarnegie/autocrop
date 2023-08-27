from typing import ClassVar, Tuple

import cv2
import dlib


class FaceWorker:
    LANDMARKS: ClassVar[str] = 'resources\\models\\shape_predictor_68_face_landmarks.dat'
    PROTOTXT: ClassVar[str] = 'resources\\weights\\deploy.prototxt.txt'
    CAFFEMODEL: ClassVar[str] = 'resources\\models\\res10_300x300_ssd_iter_140000.caffemodel'

    @classmethod
    def caffe_model(cls) -> cv2.dnn.Net:
        return cv2.dnn.readNetFromCaffe(cls.PROTOTXT, cls.CAFFEMODEL)

    @classmethod
    def worker_tuple(cls) -> Tuple[dlib.fhog_object_detector, dlib.shape_predictor]:
        return dlib.get_frontal_face_detector(), dlib.shape_predictor(cls.LANDMARKS)
