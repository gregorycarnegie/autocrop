import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
from PyQt6.QtWidgets import QComboBox

from file_types import file_manager, FileCategory


@dataclass(slots=True, frozen=True, repr=True)
class Job:
    """
    The Job class represents a job with various properties and methods for image processing.

    Attributes:
        width: The width of the job (in pixels).
        height: The height of the job (in pixels).
        fix_exposure_job: Whether to apply exposure fixes.
        multi_face_job: Whether to detect multiple faces.
        auto_tilt_job: Whether to correct for tilt automatically.
        sensitivity: The sensitivity value (used to compute the threshold).
        face_percent: The face percentage value (for cropping logic).
        gamma: The gamma value for image adjustments.
        top: The top offset for cropping.
        bottom: The bottom offset for cropping.
        left: The left offset for cropping.
        right: The right offset for cropping.
        radio_buttons: A tuple of booleans representing radio button states.
        radio_options: The array of format options (e.g., ['No', '.bmp']).
        destination: The optional destination path.
        photo_path: The optional photo path.
        folder_path: The optional folder path (for batch processing).
        video_path: The optional video path.
        start_position: The optional video start position (in ms).
        stop_position: The optional video stop position (in ms).
        table: An optional Polars DataFrame for metadata.
        column1: An optional QComboBox for selecting the first column name.
        column2: An optional QComboBox for selecting the second column name.
    """
    # Required fields
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
    radio_buttons: tuple[bool, bool, bool, bool, bool, bool]

    # Optional fields with default values
    radio_options: np.ndarray = field(default_factory=lambda: np.array(['No', '.bmp', '.jpg', '.png', '.tiff', '.webp']))
    destination: Optional[Path] = None
    photo_path: Optional[Path] = None
    folder_path: Optional[Path] = None
    video_path: Optional[Path] = None
    start_position: Optional[float] = None
    stop_position: Optional[float] = None
    table: Optional[pl.DataFrame] = None
    column1: Optional[QComboBox] = None
    column2: Optional[QComboBox] = None

    def iter_images(self) -> Optional[list[Path]]:
        """
        Retrieves a list of files from `folder_path` whose suffix is in supported file types.

        Returns:
            A list of files, or None if `folder_path` is None.
        """
        if self.folder_path is None:
            return None

        return list(
            filter(
                lambda f: f.is_file() and (
                        file_manager.is_valid_type(f, FileCategory.PHOTO) or
                        file_manager.is_valid_type(f, FileCategory.RAW) or
                        file_manager.is_valid_type(f, FileCategory.TIFF)
                ),
                self.folder_path.iterdir()
            )
        )

    def radio_tuple(self) -> tuple[str, ...]:
        """
        Returns the radio button format options as a tuple of strings.
        """
        return tuple(self.radio_options)

    def radio_choice(self) -> str:
        """
        Returns the selected image format from the radio buttons.
        """
        bool_array = np.fromiter(self.radio_buttons, np.bool_)
        return str(self.radio_options[bool_array][0])

    @property
    def size(self) -> tuple[int, int]:
        """
        Returns (width, height) with padding applied.
        """
        width = self.width * (100 + self.left + self.right) / 100
        height = self.height * (100 + self.top + self.bottom) / 100
        return int(width), int(height)

    @property
    def threshold(self) -> int:
        """
        Computes the threshold as 100 - sensitivity.
        """
        return 100 - self.sensitivity

    def get_destination(self) -> Optional[Path]:
        """
        Creates and returns the `destination` directory if specified.
        Returns None if not set.
        """
        if self.destination is None:
            return None
        self.destination.mkdir(exist_ok=True)
        return self.destination

    def file_list_to_numpy(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """
        Converts two columns of the `table` (specified by `column1` and `column2`) into
        NumPy string arrays, filtering by actual file existence in `folder_path`.
        """
        if any(_ is None for _ in (self.table, self.column1, self.column2, self.folder_path)):
            return None
        if self.table.is_empty():
            return None
        old_arr = self.table[self.column1.currentText()].to_numpy().astype(np.str_)
        new_arr = self.table[self.column2.currentText()].to_numpy().astype(np.str_)
        existing = {p.name for p in self.folder_path.iterdir()}
        mask = np.isin(old_arr, list(existing))
        return old_arr[mask], new_arr[mask]

    @property
    def destination_accessible(self) -> bool:
        """
        Checks if the `destination` path is accessible (writable).
        Returns False if `destination` is None.
        """
        import os
        return os.access(self.destination, os.W_OK) if self.destination else False

    @property
    def free_space(self) -> int:
        """
        Returns the free disk space at `destination`.
        """
        return shutil.disk_usage(self.destination).free

    @property
    def approx_byte_size(self) -> int:
        """
        Approximate size in bytes for an image of shape (height, width, 3).
        """
        return self.size * 3
