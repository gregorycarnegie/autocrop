from pathlib import Path
from typing import Any, NamedTuple, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from PyQt6.QtWidgets import QCheckBox, QComboBox, QRadioButton

from file_types import Photo


class Job(NamedTuple):
    """
    The `Job` class represents a job with various properties and methods for image processing.

    Attributes:
        width (int): The width of the job.
        height (int): The height of the job.
        fix_exposure_job (QCheckBox): The checkbox for fixing exposure.
        multi_face_job (QCheckBox): The checkbox for multi-face detection.
        auto_tilt_job (QCheckBox): The checkbox for auto tilt correction.
        sensitivity (int): The sensitivity value.
        face_percent (int): The face percentage value.
        gamma (int): The gamma value.
        top (int): The top value.
        bottom (int): The bottom value.
        left (int): The left value.
        right (int): The right value.
        radio_buttons (Tuple[QRadioButton, ...]): The tuple of radio buttons.
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
    fix_exposure_job: QCheckBox
    multi_face_job: QCheckBox
    auto_tilt_job: QCheckBox
    sensitivity: int
    face_percent: int
    gamma: int
    top: int
    bottom: int
    left: int
    right: int
    radio_buttons: Tuple[QRadioButton, ...]
    radio_options: npt.NDArray[np.str_] = np.array(['No', '.bmp', '.jpg', '.png', '.tiff', '.webp'])
    destination: Optional[Path] = None
    photo_path: Optional[Path] = None
    folder_path: Optional[Path] = None
    video_path: Optional[Path] = None
    start_position: Optional[float] = None
    stop_position: Optional[float] = None
    table: Optional[pd.DataFrame] = None
    column1: Optional[QComboBox] = None
    column2: Optional[QComboBox] = None

    def file_list(self) -> Optional[Tuple[npt.NDArray[Any], int]]:
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
            x = np.fromiter(self.folder_path.iterdir(), Path)
            y: npt.NDArray[np.bool_] = np.array([pic.suffix.lower() in Photo.file_types for pic in x])
            result = x[y]
            return result, len(result)
        return None

    def radio_tuple(self) -> Tuple[str, ...]:
        """
        The method returns a tuple of radio button options.

        Returns:
            Tuple[str, ...]: A tuple of radio button options.

        Example:
            ```python
            # Creating a job instance
            job = Job(radio_options=np.array(['No', '.bmp']))

            # Getting the radio button options
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

        x: npt.NDArray[np.bool_] = np.array([r.isChecked() for r in self.radio_buttons])
        return self.radio_options[x][0]

    @property
    def size(self) -> Tuple[int, int]:
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

    @staticmethod
    def _table_to_numpy(table: pd.DataFrame,
                        column_1: str,
                        column_2: str) -> Tuple[npt.NDArray[np.str_], npt.NDArray[np.str_]]:
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

        return table[column_1].to_numpy().astype(str), table[column_2].to_numpy().astype(str)

    def file_list_to_numpy(self) -> Optional[Tuple[npt.NDArray[np.str_], npt.NDArray[np.str_]]]:
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

        if (not isinstance(self.table, pd.DataFrame)
                or not isinstance(self.column1, QComboBox)
                or not isinstance(self.column2, QComboBox)):
            return None
        return self._table_to_numpy(self.table, self.column1.currentText(), self.column2.currentText())
