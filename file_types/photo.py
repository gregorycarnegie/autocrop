import collections.abc as c
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from .file import File


class Photo(File):
    """
    Represents a Photo class that inherits from the File class.

    The Photo class provides additional functionality for working with photo files. It defines class variables for save types, CV2 types, and RAW types. The `file_types` attribute is a concatenation of CV2 types and RAW types.

    Attributes:
        SAVE_TYPES (ClassVar[Set[str]]): A set of file extensions for save types.
        CV2_TYPES (ClassVar[npt.NDArray[np.str_]]): An array of file extensions for CV2 types.
        RAW_TYPES (ClassVar[npt.NDArray[np.str_]]): An array of file extensions for RAW types.
        file_types (npt.NDArray[np.str_]): An array of file types supported by the class.
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

    SAVE_TYPES: ClassVar[c.Set[str]] = {'.bmp', '.jpg', '.png', '.tiff', '.webp'}
    CV2_TYPES: ClassVar[npt.NDArray[np.str_]] = np.array(['.bmp', '.dib', '.jpeg', '.jpg', '.jpe', '.jp2', '.png',
                                                          '.webp', '.pbm', '.pgm', '.ppm', '.pxm', '.pnm', '.pfm',
                                                          '.sr', '.ras', '.tiff', '.tif', '.exr', '.hdr', '.pic'])
    RAW_TYPES: ClassVar[npt.NDArray[np.str_]] = np.array(['.dng', '.arw', '.cr2', '.crw', '.erf',
                                                          '.kdc', '.nef', '.nrw', '.orf', '.pef',
                                                          '.raf', '.raw', '.sr2', '.srw', '.x3f'])
    TIFF_TYPES: ClassVar[c.Set[str]] = {'.tiff', '.tif'}
    file_types = np.concatenate((CV2_TYPES, RAW_TYPES))
