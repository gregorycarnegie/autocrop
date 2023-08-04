import cv2
import dlib

SHAPE_PREDICTOR = 'resources\\models\\shape_predictor_68_face_landmarks.dat'
PROTOTXT = 'resources\\weights\\deploy.prototxt.txt'
CAFFEMODEL = 'resources\\models\\res10_300x300_ssd_iter_140000.caffemodel'

class FaceWorker:
    def __init__(self):
        self.face_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(SHAPE_PREDICTOR)
        self.worker_tuple = (self.face_detector, self.shape_predictor)

    @staticmethod
    def caffe_model() -> cv2.dnn.Net:
        return cv2.dnn.readNetFromCaffe(PROTOTXT, CAFFEMODEL)
