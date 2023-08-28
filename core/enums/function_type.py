from enum import Enum


class FunctionType(Enum):
    """
Enumeration class for function types.

Function types include:
- PHOTO: Represents a photo function type.
- FOLDER: Represents a folder function type.
- MAPPING: Represents a mapping function type.
- VIDEO: Represents a video function type.
- FRAME: Represents a frame function type.
"""

    PHOTO = 0
    FOLDER = 1
    MAPPING = 2
    VIDEO = 3
    FRAME = 4
