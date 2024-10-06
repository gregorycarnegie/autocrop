import os
import shutil
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
import numpy.typing as npt
import polars as pl
from PyQt6.QtWidgets import QComboBox

from file_types import Photo

StringArrayTuple = tuple[npt.NDArray[np.str_], npt.NDArray[np.str_]]
RadioButtonTuple = tuple[bool, bool, bool, bool, bool, bool]

class Job(NamedTuple):
    """
    The `Job` class represents a job with various properties and methods for image processing.

    Attributes:
        width (int): The width of the job.
        height (int): The height of the job.
        fix_exposure_job (bool): The checkbox for fixing exposure.
        multi_face_job (bool): The checkbox for multi-face detection.
        auto_tilt_job (bool): The checkbox for auto tilt correction.
        sensitivity (int): The sensitivity value.
        face_percent (int): The face percentage value.
        gamma (int): The gamma value.
        top (int): The top value.
        bottom (int): The bottom value.
        left (int): The left value.
        right (int): The right value.
        radio_buttons (Tuple[bool, ...]): The tuple of radio buttons.
        radio_options (npt.NDArray[np.str_]): The array of radio button options.
        destination (Optional[Path]): The optional destination path.
        photo_path (Optional[Path]): The optional photo path.
        folder_path (Optional[Path]): The optional folder path.
        video_path (Optional[Path]): The optional video path.
        start_position (Optional[float]): The optional start position.
        stop_position (Optional[float]): The optional stop position.
        table (Optional[pd.DataFrame]): The optional DataFrame.
        column1 (Optional[QComboBox]): The optional QComboBox for column 1.
        column2 (Optional[QComboBox]): The optional QComboBox for column 2.

    Methods:
        file_list() -> Optional[Tuple[npt.NDArray[Any], int]]:
            Generates a list of valid image files in the specified folder path.

        radio_tuple() -> Tuple[str, ...]:
            Returns a tuple of radio button options.

        radio_choice() -> str:
            Gets the selected image format from the radio buttons.

        size() -> Tuple[int, int]:
            Returns the size of the job as a tuple.

        threshold() -> int:
            Returns the threshold value.

        get_destination() -> Optional[Path]:
            Gets the destination path specified by the user.

        file_list_to_numpy() -> Optional[Tuple[npt.NDArray[np.str_], npt.NDArray[np.str_]]]:
            Converts the specified DataFrame columns to numpy arrays.

    Example:
        ```python
        # Creating a job instance
        job = Job(width=800, height=600, fix_exposure_job=QCheckBox(), multi_face_job=QCheckBox(),
                auto_tilt_job=QCheckBox(), sensitivity=80, face_percent=50, gamma=2, top=0, bottom=0,
                left=0, right=0, radio_buttons=(QRadioButton(),), radio_options=np.array(['No', '.bmp']),
                destination=Path('output'), photo_path=Path('photos'), folder_path=Path('images'),
                video_path=Path('videos'), start_position=0.0, stop_position=1.0, table=pd.DataFrame(),
                column1=QComboBox(), column2=QComboBox())

        # Generating a list of valid image files
        file_list = job.file_list()
        print(file_list)

        # Getting the selected image format
        image_format = job.radio_choice()
        print(image_format)

        # Getting the size of the job
        size = job.size
        print(size)

        # Getting the threshold value
        threshold = job.threshold
        print(threshold)

        # Getting the destination path
        destination = job.get_destination()
        print(destination)

        # Converting DataFrame columns to numpy arrays
        numpy_arrays = job.file_list_to_numpy()
        print(numpy_arrays)
        ```
    """

    width: int
    height: int
    fix_exposure_job: bool
    multi_face_job: bool
    auto_tilt_job: bool
    sensitivity: int
    face_percent: int
    gamma: int
    top: int
    bottom: int
    left: int
    right: int
    radio_buttons: RadioButtonTuple
    radio_options: npt.NDArray[np.str_] = np.array(['No', '.bmp', '.jpg', '.png', '.tiff', '.webp'])
    destination: Optional[Path] = None
    photo_path: Optional[Path] = None
    folder_path: Optional[Path] = None
    video_path: Optional[Path] = None
    start_position: Optional[float] = None
    stop_position: Optional[float] = None
    # table: Optional[pd.DataFrame] = None
    table: Optional[pl.DataFrame] = None
    column1: Optional[QComboBox] = None
    column2: Optional[QComboBox] = None

    def file_list(self) -> Optional[tuple[list[Path], int]]:
        """
        The method retrieves a list of files from the specified folder path. It filters the files based on their suffix and returns the filtered list along with its length.

        Returns:
            Optional[Tuple[npt.NDArray[Any], int]]: A tuple containing the filtered file list and its length, or None if no folder path is specified.

        Example:
            ```python
            # Creating a job instance
            job = Job(folder_path=Path('/path/to/folder'))

            # Getting the file list
            files = job.file_list()
            print(files)
            ```
        """

        if self.folder_path is not None:
            if result := [
                i
                for i in self.folder_path.iterdir()
                if i.suffix.lower() in Photo.file_types
            ]:
                return result, len(result)
            return
        return

    def radio_tuple(self) -> tuple[np.str_, ...]:
        """
        The method returns a tuple of radio button options.

        Returns:
            Tuple[str, ...]: A tuple of radio button options.

        Example:
            ```python
            # Creating a job instance
            job = Job(radio_options=np.array(['No', '.bmp']))

            # Getting the radio button options.
            options = job.radio_tuple()
            print(options)
            ```
"""

        return tuple(self.radio_options)

    def radio_choice(self) -> str:
        """
        The method gets the selected image format from the radio buttons.

        Returns:
            str: The selected image format.

        Example:
            ```python
            # Creating a job instance
            job = Job(radio_buttons=(QRadioButton(),), radio_options=np.array(['No', '.bmp']))

            # Getting the selected image format
            image_format = job.radio_choice()
            print(image_format)
            ```
        """

        bool_iter = np.fromiter(self.radio_buttons, np.bool_)
        return str(self.radio_options[bool_iter][0])

    @property
    def size(self) -> tuple[int, int]:
        """
        The property returns the size of the job as a tuple of width and height.

        Returns:
            Tuple[int, int]: The size of the job as a tuple.

        Example:
            ```python
            # Creating a job instance
            job = Job(width=800, height=600)

            # Getting the size of the job
            size = job.size
            print(size)
            ```
        """

        return self.width, self.height

    @property
    def threshold(self) -> int:
        """
        The property calculates the threshold value based on the sensitivity value.

        Returns:
            int: The calculated threshold value.

        Example:
            ```python
            # Creating a job instance
            job = Job(sensitivity=80)

            # Getting the threshold value
            threshold = job.threshold
            print(threshold)
            ```
        """

        return 100 - self.sensitivity

    def get_destination(self) -> Optional[Path]:
        """
        The method gets the destination path specified by the user. If the path does not exist, it is created.

        Returns:
            Optional[Path]: The specified destination path or None if no path was specified.

        Example:
            ```python
            # Creating a job instance
            job = Job()

            # Getting the destination path
            destination = job.get_destination()
            print(destination)
            ```
        """

        if self.destination is None:
            return None
        self.destination.mkdir(exist_ok=True)
        return self.destination

    def _table_to_numpy(self, table: pl.DataFrame, *,
                        column_1: str,
                        column_2: str) -> StringArrayTuple:
        """
        The function converts a specified table to a NumPy array of strings. It requires a `pd.DataFrame` object and the names of two columns in the table. The function returns a tuple containing two NumPy arrays of strings representing the values in the specified columns.

        Args:
            table (pd.DataFrame): The table to convert to a NumPy array.
            column_1 (str): The name of the first column.
            column_2 (str): The name of the second column.

        Returns:
            Tuple[npt.NDArray[np.str_], npt.NDArray[np.str_]]: A tuple containing two NumPy arrays of strings representing the values in the specified columns.

        Example:
            ```python
            # Creating a DataFrame
            df = pd.DataFrame({'Column1': ['A', 'B', 'C'], 'Column2': ['X', 'Y', 'Z']})

            # Converting the DataFrame to NumPy arrays
            result = table_to_numpy(df, 'Column1', 'Column2')
            print(result)
            ```
        """

        # Convert columns to numpy arrays
        old_file_list, new_file_list = map(lambda x: table[x].to_numpy().astype(np.str_), [column_1, column_2])

        # Get a set of all files in the folder for membership checks
        existing_files = set(self.folder_path.iterdir())

        # Vectorized Check for file existence
        mask = np.fromiter((self.folder_path / old_file in existing_files for old_file in old_file_list), np.bool_)

        # Filter using the mask and return
        return old_file_list[mask], new_file_list[mask]

    def file_list_to_numpy(self) -> Optional[StringArrayTuple]:
        """
        The method converts a table to a NumPy array of strings. It requires a `pd.DataFrame` object, a `QComboBox` object for column 1, and a `QComboBox` object for column 2. If any of these requirements are not met, the method returns None.

        Returns:
            Optional[Tuple[npt.NDArray[np.str_], npt.NDArray[np.str_]]]: A tuple containing two NumPy arrays of strings representing the values in column 1 and column 2 of the table, or None if the requirements are not met.

        Example:
            ```python
            # Creating a job instance
            job = Job(table=pd.DataFrame({'Column1': ['A', 'B', 'C'], 'Column2': ['X', 'Y', 'Z']}), column1=QComboBox(), column2=QComboBox())

            # Converting the table to a NumPy array
            result = job.file_list_to_numpy()
            print(result)
            ```
        """

        if any(_ is None for _ in (self.table, self.column1, self.column2)):
            return
        return self._table_to_numpy(self.table, column_1=self.column1.currentText(),
                                    column_2=self.column2.currentText())
    
    @property
    def destination_accessible(self) -> bool:
        return os.access(self.destination, os.W_OK) if self.destination else False
    
    @property
    def available_space(self) -> int:
        return shutil.disk_usage(self.destination).free
    
    @property
    def byte_size(self) -> int:
        return self.width * self.height * 3
