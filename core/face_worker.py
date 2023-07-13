import dlib


class FaceWorker:
    def __init__(self):
        self.face_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor('resources\\models\\shape_predictor_68_face_landmarks.dat')
        self.worker_tuple = (self.face_detector, self.shape_predictor)
