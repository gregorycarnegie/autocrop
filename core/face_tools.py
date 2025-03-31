import collections.abc as c
import threading
from pathlib import Path

import cv2
import cv2.typing as cvt
import dlib
import numpy as np
import psutil

from .resource_path import ResourcePath

# Calculate thread number based on available resources
THREAD_NUMBER = min(psutil.cpu_count(), psutil.virtual_memory().total // (2 * 1024 ** 3), 8)

# Paths to model files
LANDMARKS = ResourcePath(Path('resources') / 'models' / 'shape_predictor_68_face_landmarks.dat').meipass_path
PROTO_TXT = ResourcePath(Path('resources') / 'weights' / 'deploy.prototxt.txt').meipass_path
CAFFE_MODEL = ResourcePath(Path('resources') / 'models' / 'res10_300x300_ssd_iter_140000.caffemodel').meipass_path

# Thread-local storage for the face detection models
_thread_local = threading.local()

# Eye landmark indices
L_EYE_START, L_EYE_END = 42, 48
R_EYE_START, R_EYE_END = 36, 42


class Rectangle(dlib.rectangle):
    def __init__(self, left: int, top: int, right: int, bottom: int, confidence: float):
        super().__init__(left, top, right, bottom)
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom
        self.confidence = confidence
    
    def area(self):
        return (self.right - self.left) * (self.bottom - self.top)
    

class FaceDetector:
    """
    Thread-safe face detector using OpenCV's DNN module with a pre-trained SSD model.
    Designed to be a drop-in replacement for dlib's face detector.
    """
    
    def __init__(self, confidence_threshold=0.5):
        """Initialize the DNN face detector."""
        self.confidence_threshold = confidence_threshold
        self._net = None
    
    def _get_network(self) -> cv2.dnn.Net:
        """Get or create a thread-local network instance."""
        if not hasattr(_thread_local, 'net'):
            _thread_local.net = cv2.dnn.readNetFromCaffe(PROTO_TXT, CAFFE_MODEL)
        return _thread_local.net
    
    def __call__(self, image: cvt.MatLike) -> list[Rectangle]:
        """
        Detect faces in the image.
        
        Args:
            image: Input image (BGR format from OpenCV or grayscale)
            upsample_num_times: Ignored parameter for dlib compatibility
            
        Returns:
            List of rectangle objects mimicking dlib's detection results
        """
        # Make sure we have a color image
        if len(image.shape) == 2:
            # Convert grayscale to BGR
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Get image dimensions
        (h, w) = image.shape[:2]
        
        # Create a blob from the image
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
        
        # Get network and perform inference
        net = self._get_network()
        net.setInput(blob)
        detections = net.forward()
        
        # Process the detections
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            # Filter by confidence
            if confidence > self.confidence_threshold:
                # Compute the coordinates
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Ensure coordinates are within image boundaries
                startX, startY = max(0, startX), max(0, startY)
                endX, endY = min(w, endX), min(h, endY)
                
                # Create a rectangle object that mimics dlib's rectangle
                faces.append(Rectangle(startX, startY, endX, endY, confidence))
        
        return faces


FaceToolPair = tuple[FaceDetector, dlib.shape_predictor]


def create_tool_pair() -> FaceToolPair:
    """
    Create a pair of face detection and shape prediction tools.
    
    Returns:
        A pair of face detection and shape prediction tools.
    """
    # Use our optimized face detector
    detector = FaceDetector(confidence_threshold=0.5)
    
    # Still use dlib's shape predictor for landmarks
    # Could be replaced with a custom implementation later
    predictor = dlib.shape_predictor(LANDMARKS)
    
    return detector, predictor


def generate_face_detection_tools() -> c.Iterator[FaceToolPair]:
    """
    Generate a list of face detection and shape prediction tools.
    
    This method creates a list of tuples, where each tuple contains an instance of
    our ModernFaceDetector and dlib's shape_predictor for facial landmark detection.
    The number of tuples is determined by THREAD_NUMBER.
    
    Returns:
        Iterator of tool pairs for multi-threading.
    """
    return (create_tool_pair() for _ in range(THREAD_NUMBER))
