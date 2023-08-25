from typing import ClassVar, Tuple

import cv2
import dlib

class FaceWorker:
    SHAPE_PREDICTOR: ClassVar[str] = 'resources\\models\\shape_predictor_68_face_landmarks.dat'
    PROTOTXT: ClassVar[str] = 'resources\\weights\\deploy.prototxt.txt'
    CAFFEMODEL: ClassVar[str] = 'resources\\models\\res10_300x300_ssd_iter_140000.caffemodel'
    face_detector: ClassVar[dlib.fhog_object_detector] = dlib.get_frontal_face_detector()
    shape_predictor: ClassVar[dlib.shape_predictor] = dlib.shape_predictor(SHAPE_PREDICTOR)
    worker_tuple: ClassVar[Tuple[dlib.fhog_object_detector, dlib.shape_predictor]] = face_detector, shape_predictor

    @classmethod
    def caffe_model(cls) -> cv2.dnn.Net:
        return cv2.dnn.readNetFromCaffe(cls.PROTOTXT, cls.CAFFEMODEL)
