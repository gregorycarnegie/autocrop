from pathlib import Path
from typing import NamedTuple
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from PyQt6 import QtWidgets

# from files import IMAGE_TYPES
from .file_types import IMAGE_TYPES


class Job(NamedTuple):
    width: QtWidgets.QLineEdit
    height: QtWidgets.QLineEdit
    fix_exposure_job: QtWidgets.QCheckBox
    multiface_job: QtWidgets.QCheckBox
    autotilt_job: QtWidgets.QCheckBox
    sensitivity: QtWidgets.QDial
    facepercent: QtWidgets.QDial
    gamma: QtWidgets.QDial
    top: QtWidgets.QDial
    bottom: QtWidgets.QDial
    left: QtWidgets.QDial
    right: QtWidgets.QDial
    radio_buttons: Tuple[QtWidgets.QRadioButton, ...]
    radio_options: np.ndarray = np.array(['No', '.bmp', '.jpg', '.png', '.tiff', '.webp'])
    destination: Optional[QtWidgets.QLineEdit] = None
    photo_path: Optional[QtWidgets.QLineEdit] = None
    folder_path: Optional[QtWidgets.QLineEdit] = None
    video_path: Optional[QtWidgets.QLineEdit] = None
    start_position: Optional[float] = None
    stop_position: Optional[float] = None
    table: Optional[pd.DataFrame] = None
    column1: Optional[QtWidgets.QComboBox] = None
    column2: Optional[QtWidgets.QComboBox] = None

    def file_list(self):
        x = np.fromiter(Path(self.folder_path.text()).iterdir(), Path)
        y = np.array([pic.suffix.lower() in IMAGE_TYPES for pic in x])
        return x[y]
    
    def radio_choice(self) -> str:
        x = np.array([r.isChecked() for r in self.radio_buttons])
        return self.radio_options[x][0]
    
    def width_value(self) -> int:
        return int(self.width.text())
    
    def height_value(self) -> int:
        return int(self.height.text())
    
    def get_destination(self) -> Optional[Path]:
        if self.destination is None:
            return None
        x = Path(self.destination.text())
        x.mkdir(exist_ok=True)
        return x
    
    def file_list_to_numpy(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        if (
            not isinstance(self.table, pd.DataFrame)
            or not isinstance(self.column1, QtWidgets.QComboBox)
            or not isinstance(self.column2, QtWidgets.QComboBox)
        ):
            return None
        x = self.table[self.column1.currentText()].to_numpy().astype(str)
        y = self.table[self.column2.currentText()].to_numpy().astype(str)
        return x, y