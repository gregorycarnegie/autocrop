from typing import ClassVar

from .file import FileType


class Table(FileType):
    """
    Represents a Table class that inherits from the File class.

    The Table class provides functionality for working with table files. It defines the `file_types` attribute as an array of supported file types.

    Attributes:
        file_types (Set[str]): A set of file types supported by the class.

    Methods:
        file_filter(): Returns an array of file filters based on the file types.
        type_string(): Returns a string representing the supported file types.
        
    Example:
        ```python
        # Creating an instance of the Table class
        table = Table()

        # Getting the file types.
        types = table.file_types
        print(types)
        ```
    """

    file_types: ClassVar[set[str]] = {'.csv', '.xlsx', '.xlsm', '.xltx', '.xltm'}
