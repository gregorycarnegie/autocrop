import cv2
import dlib

class FaceWorker:
    SHAPE_PREDICTOR = 'resources\\models\\shape_predictor_68_face_landmarks.dat'
    PROTOTXT = 'resources\\weights\\deploy.prototxt.txt'
    CAFFEMODEL = 'resources\\models\\res10_300x300_ssd_iter_140000.caffemodel'
    face_detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor(SHAPE_PREDICTOR)
    worker_tuple = (face_detector, shape_predictor)

    @classmethod
    def caffe_model(cls) -> cv2.dnn.Net:
        return cv2.dnn.readNetFromCaffe(cls.PROTOTXT, cls.CAFFEMODEL)
