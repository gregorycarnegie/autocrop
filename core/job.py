from pathlib import Path
from typing import Any, NamedTuple, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from PyQt6.QtWidgets import QCheckBox, QComboBox, QDial, QRadioButton

from file_types import Photo
from line_edits import NumberLineEdit, PathLineEdit


class Job(NamedTuple):
    """
    Represents a Job in an image processing application. 

    Each field represents a PyQt6 widget or a parameter required by the job. 
    Contains several helper methods to extract information from the widgets and 
    perform common operations.

    Attributes:
        width (QLineEdit): The width of the image specified by the user.
        height (QLineEdit): The height of the image specified by the user.
        fix_exposure_job (QCheckBox): Checkbox to indicate if exposure correction is needed.
        multi_face_job (QCheckBox): Checkbox to indicate if multiple faces are to be considered.
        auto_tilt_job (QCheckBox): Checkbox to indicate if auto-tilting of the image is required.
        sensitivity (QDial): Dial to adjust the sensitivity of the application.
        face_percent (QDial): Dial to adjust the face percentage of the application.
        gamma (QDial): Dial to adjust the gamma of the application.
        top (QDial): Dial to adjust the top margin.
        bottom (QDial): Dial to adjust the bottom margin.
        left (QDial): Dial to adjust the left margin.
        right (QDial): Dial to adjust the right margin.
        radio_buttons (Tuple[QRadioButton, ...]): Tuple of radio buttons for image format selection.
        radio_options (np.ndarray): Array of image format options corresponding to the radio buttons.
        destination (Optional[QLineEdit]): The destination path where output will be saved.
        photo_path (Optional[QLineEdit]): The path to the photo to be processed.
        folder_path (Optional[QLineEdit]): The path to the folder containing multiple photos to be processed.
        video_path (Optional[QLineEdit]): The path to the video to be processed.
        start_position (Optional[float]): The start position for video processing.
        stop_position (Optional[float]): The stop position for video processing.
        table (Optional[pd.DataFrame]): A DataFrame for any tabular data needed for the job.
        column1 (Optional[QComboBox]): Dropdown for selecting a column from the DataFrame.
        column2 (Optional[QComboBox]): Dropdown for selecting another column from the DataFrame.
    """
    width: NumberLineEdit
    height: NumberLineEdit
    fix_exposure_job: QCheckBox
    multi_face_job: QCheckBox
    auto_tilt_job: QCheckBox
    sensitivity: QDial
    face_percent: QDial
    gamma: QDial
    top: QDial
    bottom: QDial
    left: QDial
    right: QDial
    radio_buttons: Tuple[QRadioButton, ...]
    radio_options: npt.NDArray[np.str_] = np.array(['No', '.bmp', '.jpg', '.png', '.tiff', '.webp'])
    destination: Optional[PathLineEdit] = None
    photo_path: Optional[PathLineEdit] = None
    folder_path: Optional[PathLineEdit] = None
    video_path: Optional[PathLineEdit] = None
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
            x = np.fromiter(Path(self.folder_path.text()).iterdir(), Path)
            y = np.array([pic.suffix.lower() in Photo().file_types for pic in x])
            result = x[y]
            return result, len(result)

    def radio_tuple(self) -> Tuple[str, ...]:
        return tuple(self.radio_options)

    def radio_choice(self) -> str:
        """
        Gets the image format selected by the user via the radio buttons.

        Returns:
            str: The selected image format.
        """
        x = np.array([r.isChecked() for r in self.radio_buttons])
        return self.radio_options[x][0]
    
    def width_value(self) -> int:
        """
        Gets the image width specified by the user.

        Returns:
            int: The specified image width.
        """
        return int(self.width.text())
    
    def height_value(self) -> int:
        """
        Gets the image height specified by the user.

        Returns:
            int: The specified image height.
        """
        return int(self.height.text())
    
    def size(self) -> Tuple[int, int]:
        return int(self.width.text()), int(self.height.text())

    def get_destination(self) -> Optional[Path]:
        """
        Gets the destination path specified by the user.

        If the path does not exist, it is created.

        Returns:
            pathlib.Path: The specified destination path
            None: if no path was specified.
        """
        if self.destination is None: return None
        x = Path(self.destination.text())
        x.mkdir(exist_ok=True)
        return x
    
    def file_list_to_numpy(self) -> Optional[Tuple[npt.NDArray[np.str_], npt.NDArray[np.str_]]]:
        """
        Converts the specified DataFrame columns to numpy arrays.

        Returns:
            tuple(numpy.ndarray, numpy.ndarray): Two numpy arrays representing the specified DataFrame columns.
            None: If 'table', 'column1', or 'column2' is not in the expected format.
        """
        def _table_to_numpy(table: pd.DataFrame,
                            column_1: str,
                            column_2: str) -> Tuple[npt.NDArray[np.str_], npt.NDArray[np.str_]]:
            return table[column_1].to_numpy().astype(str), table[column_2].to_numpy().astype(str) # type: ignore
        
        if (not isinstance(self.table, pd.DataFrame)
            or not isinstance(self.column1, QComboBox)
            or not isinstance(self.column2, QComboBox)):
            return None
        return _table_to_numpy(self.table, self.column1.currentText(), self.column2.currentText())
