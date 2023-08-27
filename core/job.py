from pathlib import Path
from typing import Any, NamedTuple, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from PyQt6.QtWidgets import QCheckBox, QComboBox, QRadioButton

from file_types import Photo


class Job(NamedTuple):
    """
    Represents a Job in an image processing application. 

    Each field represents a PyQt6 widget or a parameter required by the job. 
    Contains several helper methods to extract information from the widgets and 
    perform common operations.

    Attributes:
        width int: The width of the image specified by the user.
        height int: The height of the image specified by the user.
        fix_exposure_job (QCheckBox): Checkbox to indicate if exposure correction is needed.
        multi_face_job (QCheckBox): Checkbox to indicate if multiple faces are to be considered.
        auto_tilt_job (QCheckBox): Checkbox to indicate if auto-tilting of the image is required.
        sensitivity int: Value of the sensitivity of the application.
        face_percent int: Value of the face percentage of the application.
        gamma int: Value of the gamma of the application.
        top int: Value of the top margin.
        bottom int: Value of the bottom margin.
        left int: Value of the left margin.
        right int: Value of the right margin.
        radio_buttons (Tuple[QRadioButton, ...]): Tuple of radio buttons for image format selection.
        radio_options (np.ndarray): Array of image format options corresponding to the radio buttons.
        destination (Optional[Path]): The destination path where output will be saved.
        photo_path (Optional[Path]): The path to the photo to be processed.
        folder_path (Optional[Path]): The path to the folder containing multiple photos to be processed.
        video_path (Optional[Path]): The path to the video to be processed.
        start_position (Optional[float]): The start position for video processing.
        stop_position (Optional[float]): The stop position for video processing.
        table (Optional[pd.DataFrame]): A DataFrame for any tabular data needed for the job.
        column1 (Optional[QComboBox]): Dropdown for selecting a column from the DataFrame.
        column2 (Optional[QComboBox]): Dropdown for selecting another column from the DataFrame.
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
        Generates a list of valid image files in the folder specified by 'folder_path'.

        Returns:
            numpy.ndarray: An array of pathlib.Path objects representing the valid image files.
        """
        if self.folder_path is not None:
            x = np.fromiter(self.folder_path.iterdir(), Path)
            y: npt.NDArray[np.bool_] = np.array([pic.suffix.lower() in Photo.file_types for pic in x])
            result = x[y]
            return result, len(result)
        return None

    def radio_tuple(self) -> Tuple[str, ...]:
        return tuple(self.radio_options)

    def radio_choice(self) -> str:
        """
        Gets the image format selected by the user via the radio buttons.

        Returns:
            str: The selected image format.
        """
        x: npt.NDArray[np.bool_] = np.array([r.isChecked() for r in self.radio_buttons])
        return self.radio_options[x][0]
    
    @property
    def size(self) -> Tuple[int, int]:
        return self.width, self.height
    
    @property
    def threshold(self) -> int:
        return 100 - self.sensitivity

    def get_destination(self) -> Optional[Path]:
        """
        Gets the destination path specified by the user.

        If the path does not exist, it is created.

        Returns:
            pathlib.Path: The specified destination path
            None: if no path was specified.
        """
        if self.destination is None:
            return None
        self.destination.mkdir(exist_ok=True)
        return self.destination
    
    def file_list_to_numpy(self) -> Optional[Tuple[npt.NDArray[np.str_], npt.NDArray[np.str_]]]:
        """
        Converts the specified DataFrame columns to numpy arrays.

        Returns:
            tuple(numpy.ndarray, numpy.ndarray): Two numpy arrays representing the specified DataFrame columns.
            None: If 'table', 'column1', or 'column2' is not in the expected format.
        """
        def table_to_numpy(table: pd.DataFrame,
                           column_1: str,
                           column_2: str) -> Tuple[npt.NDArray[np.str_], npt.NDArray[np.str_]]:
            return table[column_1].to_numpy().astype(str), table[column_2].to_numpy().astype(str)
        
        if (not isinstance(self.table, pd.DataFrame)
            or not isinstance(self.column1, QComboBox)
            or not isinstance(self.column2, QComboBox)):
            return None
        return table_to_numpy(self.table, self.column1.currentText(), self.column2.currentText())
