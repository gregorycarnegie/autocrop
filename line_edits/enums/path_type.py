from enum import auto, Enum, unique


@unique
class PathType(Enum):
    """
Enumeration class for path types.

Path types include:
- IMAGE: Represents an image path type.
- TABLE: Represents a table path type.
- VIDEO: Represents a video path type.
- FOLDER: Represents a folder path type.
"""

    IMAGE = auto()
    TABLE = auto()
    VIDEO = auto()
    FOLDER = auto()
