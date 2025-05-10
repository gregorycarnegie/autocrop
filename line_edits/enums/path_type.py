from enum import Flag, auto, unique


@unique
class PathType(Flag):
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
