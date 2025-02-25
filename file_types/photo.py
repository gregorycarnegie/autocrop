from typing import ClassVar

from .file import FileType


class Photo(FileType):
    """
    Represents a Photo class that inherits from the File class.

    The Photo class provides additional functionality for working with photo files. It defines class variables for save types, CV2 types, and RAW types. The `file_types` attribute is a concatenation of CV2 types and RAW types.

    Attributes:
        SAVE_TYPES (ClassVar[Set[str]]): A set of file extensions for save types.
        CV2_TYPES (ClassVar[Set[str]]): A set of file extensions for CV2 types.
        RAW_TYPES (ClassVar[Set[str]]): A set of file extensions for RAW types.
        file_types (ClassVar[Set[str]]): A set of file types supported by the class.
        default_directory (str): The default directory for file operations.

    Methods:
        file_filter(): Returns an array of file filters based on the file types.
        type_string(): Returns a string representing the supported file types.
        
    Example:
        ```python
        # Creating an instance of the Photo class
        photo = Photo()

        # Getting the file types
        types = photo.file_types
        print(types)
        ```
    """

    SAVE_TYPES: ClassVar[set[str]] = {'.bmp', '.jpg', '.png', '.tiff', '.webp'}
    CV2_TYPES: ClassVar[set[str]] = {'.bmp', '.dib', '.jfif', '.jpeg', '.jpg',
                                     '.jpe', '.jp2', '.png', '.webp', '.pbm',
                                     '.pgm', '.ppm', '.pxm', '.pnm', '.pfm',
                                     '.sr', '.ras', '.tiff', '.tif', '.exr',
                                     '.hdr', '.pic'}
    RAW_TYPES: ClassVar[set[str]] = {'.dng', '.arw', '.cr2', '.crw', '.erf',
                                     '.kdc', '.nef', '.nrw', '.orf', '.pef',
                                     '.raf', '.raw', '.sr2', '.srw', '.x3f'}
    TIFF_TYPES: ClassVar[set[str]] = {'.tiff', '.tif'}
    file_types: ClassVar[set[str]] = CV2_TYPES | RAW_TYPES  # Union of sets
