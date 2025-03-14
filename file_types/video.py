from typing import ClassVar
from pathlib import Path

from .file import FileType


class Video(FileType):
    """
    Represents a Video class that inherits from the File class.

    The Video class provides functionality for working with video files. It defines the `default_directory` attribute as the default directory for video operations, and the `file_types` attribute as an array of supported video file types.

    Attributes:
        default_directory (str): The default directory for video operations.
        file_types (set[str]): An set of video file types supported by the class.

    Methods:
        file_filter(): Returns an array of file filters based on the file types.
        type_string(): Returns a string representing the supported file types.
        
    Example:
        ```python
        # Creating an instance of the Video class
        video = Video()

        # Getting the default directory.
        directory = video.default_directory
        print(directory)

        # Getting the file types.
        types = video.file_types
        print(types)
        ```
    """

    file_types: ClassVar[set[str]] = {'.avi', '.m4v', '.mkv', '.mov', '.mp4'}
    default_directory: ClassVar[str] = Path.home().joinpath('Videos').as_posix()
