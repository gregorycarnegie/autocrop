from dataclasses import dataclass

import cv2


@dataclass(frozen=True)
class Config:
    """
    Centralized configuration for processing defaults and magic numbers.
    """
    # Threshold for converting user gamma to lookup scale
    gamma_threshold: float = 0.001

    # Default height for preview/grayscale formatting
    default_preview_height: int = 256

    # Divisor used to determine a scale factor for face detection
    face_scale_divisor: int = 500

    # Default interpolation method for resizing (OpenCV flag)
    interpolation: int = cv2.INTER_AREA

    # Default border colour for padding when cropping outside bounds
    border_colour: tuple[int, int, int] = (0, 0, 0)

    border_type: int = cv2.BORDER_CONSTANT
    # Drawing settings for bounding boxes and confidence text
    bbox_font_scale: float = 0.45
    bbox_line_width: int = 2
    bbox_text_offset: int = 10

    infer_schema_length=1_000
    disable_logging = False
