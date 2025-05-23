# autocrop_rs/face_detection.pyi
"""Type stubs for autocrop_rs.face_detection module."""

class Rectangle:
    """
    A rectangle representing a detected face with confidence score.

    Attributes:
        left: Left coordinate of the rectangle
        top: Top coordinate of the rectangle
        right: Right coordinate of the rectangle
        bottom: Bottom coordinate of the rectangle
        confidence: Confidence score between 0.0 and 1.0
    """

    left: int
    top: int
    right: int
    bottom: int
    confidence: float

    def __init__(
        self,
        left: int,
        top: int,
        right: int,
        bottom: int,
        confidence: float
    ) -> None:
        """
        Create a new Rectangle.

        Args:
            left: Left coordinate (must be < right)
            top: Top coordinate (must be < bottom)
            right: Right coordinate (must be > left)
            bottom: Bottom coordinate (must be > top)
            confidence: Confidence score (must be between 0.0 and 1.0)

        Raises:
            ValueError: If coordinates are invalid or confidence is out of range
        """
        ...

    @property
    def width(self) -> int:
        """Width of the rectangle (right - left)."""
        ...

    @property
    def height(self) -> int:
        """Height of the rectangle (bottom - top)."""
        ...

    @property
    def area(self) -> int:
        """Area of the rectangle (width * height)."""
        ...

def determine_scale_factor(
    width: int,
    height: int,
    face_scale_divisor: int
) -> int:
    """
    Determine the optimal scale factor for face detection based on image dimensions.

    This function calculates an appropriate scale factor to resize images for faster
    face detection while maintaining accuracy. Larger images are scaled down more
    aggressively to improve performance.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        face_scale_divisor: Divisor used to calculate scale factor (typically 500)

    Returns:
        Scale factor (minimum 1, based on smaller dimension / face_scale_divisor)

    Example:
        >>> determine_scale_factor(1920, 1080, 500)
        2
        >>> determine_scale_factor(640, 480, 500)
        1
    """
    ...

def scale_face_coordinates(
    face: Rectangle,
    scale_factor: int
) -> tuple[int, int, int, int]:
    """
    Scale face coordinates based on the scale factor used during detection.

    When face detection is performed on a scaled-down image for performance,
    the resulting coordinates need to be scaled back up to match the original
    image dimensions.

    Args:
        face: Rectangle object representing the detected face
        scale_factor: Scale factor that was used during detection

    Returns:
        Tuple of (x, y, width, height) with coordinates scaled to original image size

    Example:
        >>> rect = Rectangle(100, 100, 200, 200, 0.9)
        >>> scale_face_coordinates(rect, 2)
        (200, 200, 200, 200)  # All coordinates doubled
        >>> scale_face_coordinates(rect, 1)
        (100, 100, 100, 100)  # No scaling applied
    """
    ...

def find_best_face(faces: list[Rectangle]) -> Rectangle | None:
    """
    Find the face with the highest confidence score from a list of detections.

    This function selects the most confident face detection from multiple
    candidates, which is useful when only processing a single face is desired.

    Args:
        faces: List of Rectangle objects representing detected faces

    Returns:
        Rectangle with the highest confidence score, or None if the list is empty

    Example:
        >>> face1 = Rectangle(0, 0, 100, 100, 0.8)
        >>> face2 = Rectangle(100, 100, 200, 200, 0.95)
        >>> face3 = Rectangle(200, 200, 300, 300, 0.7)
        >>> best = find_best_face([face1, face2, face3])
        >>> best.confidence
        0.95
        >>> find_best_face([])
        None
    """
    ...
