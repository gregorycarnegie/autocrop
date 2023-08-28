from typing import Optional

from pathlib import Path

import numpy as np
import numpy.typing as npt


class File:
    """
    Represents a File class.

    The File class provides methods for working with file types, file filters, and type strings.

    Attributes:
        file_types (npt.NDArray[np.str_]): An array of file types.
        default_directory (str): The default directory for file operations.

    Methods:
        file_filter(): Returns an array of file filters based on the file types.
        type_string(): Returns a string representing the supported file types.

    Example:
        ```python
        # Creating an instance of the File class
        file = File()

        # Getting file filters
        filters = file.file_filter()
        print(filters)

        # Getting the type string
        types = file.type_string()
        print(types)
        ```
    """

    file_types: npt.NDArray[np.str_] = np.array([''])
    default_directory = f'{Path.home()}\\Pictures'

    @classmethod
    def file_filter(cls) -> npt.NDArray[np.str_]:
        """
        The method returns an array of file filters based on the file types defined in the class. Each file filter is in the format `*<file_type>`.

        Returns:
            npt.NDArray[np.str_]: An array of file filters.

        Example:
            ```python
            # Getting file filters
            filters = File.file_filter()
            print(filters)
            ```
        """

        return np.array([f'*{file}' for file in cls.file_types])

    @classmethod
    def type_string(cls) -> Optional[str]:
        """
        The method returns a string representing the file types supported by the class. If no file types are defined, the method returns None. Otherwise, it returns a string in the format 'All Files (*);;File Type 1 (*File Type 1);;File Type 2 (*File Type 2);;...'.

        Returns:
            Optional[str]: A string representing the file types supported by the class, or None if no file types are defined.

        Example:
            ```python
            # Getting the type string
            types = File.type_string()
            print(types)
            ```
        """

        if not cls.file_types[0]:
            return None
        return 'All Files (*);;' + ';;'.join(f'{_} Files (*{_})' for _ in np.sort(cls.file_types))
