from typing import Tuple
import dlib


class FaceWorker(Tuple):
    FACE_DETECTOR = dlib.get_frontal_face_detector()
    SHAPE_PREDICTOR = dlib.shape_predictor('resources\\models\\shape_predictor_68_face_landmarks.dat')
