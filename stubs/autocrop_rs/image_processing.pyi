
import cv2.typing as cvt
import numpy as np
from numpy.typing import NDArray

def reshape_buffer_to_image(
    input_bytes: bytes,
    height: int,
    width: int
) -> NDArray[np.uint8]: ...  # Shape: (height, width, 3)

def correct_exposure(
    image: cvt.MatLike,  # Shape: (H, W, 3)
    exposure: bool,
    video: bool
) -> cvt.MatLike: ...  # Shape: (H, W, 3)

def gamma(gamma_value: float) -> cvt.MatLike: ...  # Shape: (256,)

def calculate_dimensions(
    height: int,
    width: int,
    target_height: int
) -> tuple[int, float]: ...

def get_rotation_matrix(
    left_eye_landmarks: NDArray[np.float64],  # Shape: (N, 2)
    right_eye_landmarks: NDArray[np.float64],  # Shape: (N, 2)
    scale_factor: float
) -> NDArray[np.float64]: ...  # Shape: (2, 3)

def crop_positions(
    face: tuple[float, float, float, float],  # (x, y, width, height)
    face_percent: int,
    dimensions: tuple[int, int],  # (width, height)
    padding: tuple[int, int, int, int]  # (top, bottom, left, right)
) -> tuple[int, int, int, int] | None: ...  # (x1, y1, x2, y2)
