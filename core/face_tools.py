from dataclasses import dataclass

import cv2
import psutil

from .resource_path import ResourcePath

# Calculate thread number based on available resources
THREAD_NUMBER = min(psutil.cpu_count(), psutil.virtual_memory().total // (2 * 1024 ** 3), 8)

# Paths to model files
model_directory = ResourcePath('resources\\models')
YUNET_MODEL = (model_directory  / 'face_detection_yunet_2023mar.onnx').as_resource() # MIT license
LBFMODEL = (model_directory / 'lbfmodel.yaml').as_resource() # MIT license

# Eye landmark indices
L_EYE_START, L_EYE_END = 36, 42
R_EYE_START, R_EYE_END = 42, 48


@dataclass(slots=True, frozen=True)
class Rectangle:
    left: int
    top: int
    right: int
    bottom: int
    confidence: float = 1.0

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def area(self) -> int:
        return self.width * self.height

    def __post_init__(self) -> None:
        if self.left >= self.right or self.top >= self.bottom:
            raise ValueError("Invalid rectangle coordinates")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be in [0, 1]")
    

class YuNetFaceDetector:
    """Face detector using YuNet with OpenCV's DNN module."""
    
    def __init__(self):
        # Create YuNet detector
        self._detector = cv2.FaceDetectorYN.create(
            YUNET_MODEL.as_posix(),
            "",
            (320, 320),  # Input size
            0.9,         # Score threshold
            0.3,         # NMS threshold
            5000         # Top K
        )
    
    def __call__(self, image, sensitivity: int) -> list[Rectangle]:
        """
        Detect faces in the image.
        
        Args:
            image: Input image
            sensitivity: Detection threshold (0-100)
            
        Returns:
            List of Rectangle objects mimicking dlib's detection results
        """
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Set input size
        self._detector.setInputSize((width, height))
        
        # Adjust a score threshold based on sensitivity
        score_threshold = max(0.1, sensitivity / 100.0)  # Ensure a minimum threshold
        self._detector.setScoreThreshold(score_threshold)
        
        # Detect faces
        _, faces = self._detector.detect(image)
        
        def create_rectangle(face) -> Rectangle:
            x, y, w, h = map(int, face[:4])
            # Create the rectangle with the top-left (x, y) and bottom-right (x+w, y+h)
            return Rectangle(x, y, x + w, y + h, face[14])

        return list(map(create_rectangle, faces)) if faces is not None else []


type FaceToolPair = tuple[YuNetFaceDetector, cv2.face.Facemark]


def create_tool_pair() -> FaceToolPair:
    """
    Create a pair of face detection and shape prediction tools.
    
    Returns:
        A pair of face detection and shape prediction tools.
    """
    # Use our optimized face detector
    detector = YuNetFaceDetector()
    
    # Use OpenCV's FacemarkLBF instead of dlib's shape predictor
    facemark = cv2.face.createFacemarkLBF()
    facemark.loadModel(LBFMODEL.as_posix())
    
    # Return the detector and facemark model as a pair
    return detector, facemark


def generate_face_detection_tools() -> list[FaceToolPair]:
    """
    Generate a list of face detection and shape prediction tools.
    
    This method creates a list of tuples, where each tuple contains an instance of
    our ModernFaceDetector and dlib's shape_predictor for facial landmark detection.
    The number of tuples is determined by THREAD_NUMBER.
    
    Returns:
        Iterator of tool pairs for multi-threading.
    """
    return [create_tool_pair() for _ in range(THREAD_NUMBER)]
