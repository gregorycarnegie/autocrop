import os
import shutil
from collections.abc import Iterator
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
        width (int): The width of the job (in pixels).
        height (int): The height of the job (in pixels).
        fix_exposure_job (bool): Whether to apply exposure fixes.
        multi_face_job (bool): Whether to detect multiple faces.
        auto_tilt_job (bool): Whether to correct for tilt automatically.
        sensitivity (int): The sensitivity value (used to compute threshold).
        face_percent (int): The face percentage value (for cropping logic).
        gamma (int): The gamma value for image adjustments.
        top (int): The top offset for cropping.
        bottom (int): The bottom offset for cropping.
        left (int): The left offset for cropping.
        right (int): The right offset for cropping.
        radio_buttons (RadioButtonTuple): A tuple of booleans representing radio button states.
        radio_options (npt.NDArray[np.str_]): The array of format options (e.g., ['No', '.bmp']).
        destination (Optional[Path]): The optional destination path.
        photo_path (Optional[Path]): The optional photo path.
        folder_path (Optional[Path]): The optional folder path (for batch processing).
        video_path (Optional[Path]): The optional video path.
        start_position (Optional[float]): The optional video start position (in ms).
        stop_position (Optional[float]): The optional video stop position (in ms).
        table (Optional[pl.DataFrame]): An optional Polars DataFrame for metadata.
        column1 (Optional[QComboBox]): An optional QComboBox for selecting the first column name.
        column2 (Optional[QComboBox]): An optional QComboBox for selecting the second column name.

    Methods:
        file_list() -> Optional[tuple[list[Path], int]]:
            Returns a list of image files in `folder_path` with supported extensions and its length.

        radio_tuple() -> tuple[np.str_, ...]:
            Returns a tuple of the available radio button format options.

        radio_choice() -> str:
            Returns the selected image format based on the radio button states.

        size() -> tuple[int, int]:
            Returns (width, height).

        threshold() -> int:
            Returns a threshold value computed from `sensitivity` (e.g., 100 - sensitivity).

        get_destination() -> Optional[Path]:
            Creates and returns the `destination` path if specified.

        file_list_to_numpy() -> Optional[StringArrayTuple]:
            Converts two columns of the `table` into string NumPy arrays, filtered by existing files.

        destination_accessible -> bool:
            Checks if the destination is accessible for writing.

        available_space -> int:
            Returns the free disk space (in bytes) at the destination.

        byte_size -> int:
            Returns an approximate size in bytes (width * height * 3).
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
    table: Optional[pl.DataFrame] = None
    column1: Optional[QComboBox] = None
    column2: Optional[QComboBox] = None

    def __repr__(self):
        return f"""
Dimensions: {self.width} x {self.height}
Fix Exposure: {self.fix_exposure_job}
Multi Face: {self.multi_face_job}
Auto Tilt: {self.auto_tilt_job}
Sensitivity: {self.sensitivity}
Face Percent: {self.face_percent}%
Gamma: {self.gamma}
Top padding: {self.top}
Bottom padding: {self.bottom}
Left padding: {self.left}
Right padding: {self.right}
Radio Buttons: {self.radio_buttons}
Radio Options: {self.radio_options}
Destination: {self.destination}
Photo Path: {self.photo_path}
Folder Path: {self.folder_path}
Video Path: {self.video_path}
Start Position: {self.start_position}
Stop Position: {self.stop_position}
Table: {self.table}
Column 1: {self.column1}
Column 2: {self.column2}
"""

    def path_iter(self) -> Optional[tuple[Iterator[Path], int]]:
        """
        Retrieves a list of files from `folder_path` whose suffix is in `Photo.file_types`.
        Returns a tuple of (iterator_of_files, count). If `folder_path` is None, returns None.
        """
        if self.folder_path is None:
            return None
        
        # Create a list of valid files once instead of iterating twice
        valid_files = [i for i in self.folder_path.iterdir() if i.suffix.lower() in Photo.file_types]
        return iter(valid_files), len(valid_files)

    def radio_tuple(self) -> tuple[np.str_, ...]:
        """
        Returns the radio button format options as a tuple of strings.

        Example:
            ```python
            job = Job(radio_options=np.array(['No', '.bmp']))
            options = job.radio_tuple()
            print(options)  # ('No', '.bmp')
            ```
        """

        return tuple(self.radio_options)

    def radio_choice(self) -> str:
        """
        Returns the selected image format from the radio buttons.

        Example:
            ```python
            # Suppose radio_buttons=(False, True, False, ...) and
            # radio_options=np.array(['No', '.bmp', '.jpg'])
            job = Job(radio_buttons=(False, True, False),
                      radio_options=np.array(['No', '.bmp', '.jpg']))
            selected_format = job.radio_choice()
            print(selected_format)  # ".bmp"
            ```
        """

        bool_array = np.fromiter(self.radio_buttons, np.bool_)
        return str(self.radio_options[bool_array][0])

    @property
    def size(self) -> tuple[int, int]:
        """
        Returns (width, height).

        Example:
            ```python
            job = Job(width=800, height=600, ...)
            print(job.size)  # (800, 600)
            ```
        """

        return self.width, self.height

    @property
    def threshold(self) -> int:
        """
        Computes threshold as 100 - sensitivity.

        Example:
            ```python
            job = Job(sensitivity=80, ...)
            print(job.threshold)  # 20
            ```
        """

        return 100 - self.sensitivity

    def get_destination(self) -> Optional[Path]:
        """
        Creates and returns the `destination` directory if specified. Returns None if not set.

        Example:
            ```python
            job = Job(destination=Path('output'))
            dest_path = job.get_destination()
            print(dest_path)  # Path('output')
            ```
        """

        if self.destination is None:
            return None
        self.destination.mkdir(exist_ok=True)
        return self.destination

    def _table_to_numpy(self, table: pl.DataFrame,
                        *,
                        column_1: str,
                        column_2: str) -> StringArrayTuple:
        """
        Converts two columns from a Polars DataFrame into NumPy arrays of strings, filtering out
        rows whose corresponding file does not exist in `folder_path`.

        Args:
            table (pl.DataFrame): The source Polars DataFrame.
            column_1 (str): The name of the first column.
            column_2 (str): The name of the second column.

        Returns:
            StringArrayTuple: A tuple of two 1D string arrays (old_file_list, new_file_list).
        """

        # Convert columns to numpy arrays
        old_file_array = table[column_1].to_numpy().astype(np.str_)
        new_file_array = table[column_2].to_numpy().astype(np.str_)

        # Get a set of all files in the folder for membership checks
        existing_files = set(self.folder_path.iterdir())

        # Vectorized Check for file existence
        mask = np.fromiter((self.folder_path / old_file in existing_files for old_file in old_file_array), np.bool_)

        # Filter using the mask and return
        return old_file_array[mask], new_file_array[mask]

    def file_list_to_numpy(self) -> Optional[StringArrayTuple]:
        """
        Converts two columns of the `table` (specified by `column1` and `column2`) into
        NumPy string arrays, filtering by actual file existence in `folder_path`.

        Returns:
            A tuple (old_file_list, new_file_list), or None if requirements are not met.

        Example:
            ```python
            job = Job(
                table=pl.DataFrame({
                    'Column1': ['file1.jpg', 'file2.bmp'],
                    'Column2': ['result1.jpg', 'result2.bmp']
                }),
                column1=some_QComboBox_with_value_Column1,
                column2=some_QComboBox_with_value_Column2,
                folder_path=Path('/path/to/folder')
            )
            arrays = job.file_list_to_numpy()
            if arrays:
                old_files, new_files = arrays
            ```
        """

        if any(_ is None for _ in (self.table, self.column1, self.column2)):
            return None
        return self._table_to_numpy(self.table, column_1=self.column1.currentText(),
                                    column_2=self.column2.currentText())
    
    @property
    def destination_accessible(self) -> bool:
        """
        Checks if the `destination` path is accessible (writable). Returns False if `destination` is None.
        """

        return os.access(self.destination, os.W_OK) if self.destination else False
    
    @property
    def available_space(self) -> int:
        """
        Returns the free disk space at `destination`. If `destination` is None, raises an error.
        """

        return shutil.disk_usage(self.destination).free
    
    @property
    def byte_size(self) -> int:
        """
        Approximate size in bytes for an image of shape (height, width, 3).
        """

        return self.width * self.height * 3
