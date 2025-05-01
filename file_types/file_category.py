from enum import auto, Flag


class FileCategory(Flag):
    """Enumeration of file categories for the application."""
    PHOTO = auto()
    RAW = auto()
    TIFF = auto()
    VIDEO = auto()
    TABLE = auto()
    UNKNOWN = auto()
