import math
import os
import shutil
from contextlib import suppress
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from PyQt6.QtWidgets import QComboBox

from file_types import FileCategory, file_manager


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
        radio_options: The array of format options (e.g. ['No', '.bmp']).
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
    radio_options: np.ndarray = field(
        default_factory=partial(np.array, ['No', '.bmp', '.jpg', '.png', '.tiff', '.webp'])
    )
    destination: Path | None = None
    photo_path: Path | None = None
    folder_path: Path | None = None
    video_path: Path | None = None
    start_position: float | None = None
    stop_position: float | None = None
    table: pl.DataFrame | None = None
    column1: QComboBox | None = None
    column2: QComboBox | None = None

    @staticmethod
    def _validate_path(path: Path | None) -> Path | None:
        """
        Validate a path to ensure it's safe and accessible.
        Returns None if the path is invalid or inaccessible.
        """
        if path is None:
            return None

        with suppress(Exception):
            # Resolve to get an absolute normalized path
            resolved_path = path.resolve()

            # Check that the path exists
            if not resolved_path.exists():
                return None

            # Check that the path is accessible
            return resolved_path if os.access(resolved_path, os.R_OK) else None
        return None

    @property
    def safe_photo_path(self) -> Path | None:
        """Returns a validated photo path or None if invalid."""
        return self._validate_path(self.photo_path)

    @property
    def safe_folder_path(self) -> Path | None:
        """Returns a validated folder path or None if invalid."""
        return self._validate_path(self.folder_path)

    @property
    def safe_destination(self) -> Path | None:
        """Returns a validated destination path or None if invalid."""
        return self._validate_path(self.destination)

    @property
    def safe_video_path(self) -> Path | None:
        """Returns a validated video path or None if invalid."""
        return self._validate_path(self.video_path)

    def iter_images(self) -> list[Path] | None:
        """
        Retrieves a list of files from `folder_path` whose suffix is in supported file types.
        Includes path validation for security.

        Returns:
            A list of validated file paths, or None if `folder_path` is None or invalid.
        """
        # Validate the folder path first
        safe_folder = self.safe_folder_path
        if safe_folder is None:
            return None

        # Only include files that:
        # 1. Are actually files (not directories)
        # 2. Have supported file types
        # 3. Are accessible
        valid_files = []
        for f in safe_folder.iterdir():
            try:
                if (f.is_file() and
                    (file_manager.is_valid_type(f, FileCategory.PHOTO) or
                     file_manager.is_valid_type(f, FileCategory.RAW) or
                     file_manager.is_valid_type(f, FileCategory.TIFF)) and
                     os.access(f, os.R_OK)):
                    valid_files.append(f)
            except OSError:
                # Skip files that cause errors
                continue

        return valid_files

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

    def get_destination(self) -> Path | None:
        """
        Creates and returns the `destination` directory if specified.
        Returns None if not set or invalid.
        """
        # Validate the destination path
        safe_dest = self.safe_destination
        if safe_dest is None:
            return None

        # Try to create the directory
        with suppress(Exception):
            safe_dest.mkdir(exist_ok=True)
            return safe_dest
        return None

    def file_list_to_numpy(self) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Converts two columns of the `table` (specified by `column1` and `column2`) into
        NumPy string arrays, filtering by actual file existence in `folder_path`.
        Includes path validation for security.
        """
        # Check for None values or empty table
        if any(_ is None for _ in (self.table, self.column1, self.column2)):
            return None

        # Validate the folder path
        safe_folder = self.safe_folder_path
        if safe_folder is None:
            return None

        if self.table is not None:
            if self.table.is_empty():
                return None

            with suppress(Exception):
                # Extract column data safely
                col1_name = self.column1.currentText() if self.column1 is not None else ''
                col2_name = self.column2.currentText() if self.column2 is not None else ''

                # Validate column names
                if not col1_name or not col2_name:
                    return None

                if col1_name not in self.table.columns or col2_name not in self.table.columns:
                    return None

                # Convert to NumPy arrays
                old_arr = self.table[col1_name].to_numpy().astype(np.str_)
                new_arr = self.table[col2_name].to_numpy().astype(np.str_)

                # Build a set of existing filenames with proper validation
                existing = set()
                for p in safe_folder.iterdir():
                    try:
                        if p.is_file() and os.access(p, os.R_OK):
                            existing.add(p.name)
                    except OSError:
                        continue

                # Create a mask for existing files
                mask = np.isin(old_arr, list(existing))

                # Apply the mask
                return old_arr[mask], new_arr[mask]
        return None

    @property
    def destination_accessible(self) -> bool:
        """
        Checks if the `destination` path is accessible (writable).
        Returns False if `destination` is None or inaccessible.
        """
        # Validate the destination path
        safe_dest = self.safe_destination
        if safe_dest is None:
            return False

        # Check if the directory can be written to
        try:
            return os.access(safe_dest, os.W_OK)
        except OSError:
            return False

    @property
    def free_space(self) -> int:
        """
        Returns the free disk space at `destination`.
        Returns 0 if the destination is invalid or inaccessible.
        """
        # Validate the destination path
        safe_dest = self.safe_destination
        if safe_dest is None:
            return 0

        # Check free space with error handling
        try:
            return shutil.disk_usage(safe_dest).free
        except OSError:
            return 0

    @property
    def approx_byte_size(self) -> int:
        """
        Approximate size in bytes for an image of shape (height, width, 3).
        """
        return math.prod(self.size) * 3

    # Add to core/job.py
    def serialize(self) -> dict[str, Any]:
        """Convert job to a serializable dictionary"""
        return {
            'width': self.width,
            'height': self.height,
            'fix_exposure_job': self.fix_exposure_job,
            'multi_face_job': self.multi_face_job,
            'auto_tilt_job': self.auto_tilt_job,
            'sensitivity': self.sensitivity,
            'face_percent': self.face_percent,
            'gamma': self.gamma,
            'top': self.top,
            'bottom': self.bottom,
            'left': self.left,
            'right': self.right,
            'radio_buttons': tuple(self.radio_buttons),
            'radio_options': self.radio_options.tolist(),
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> 'Job':
        """Create job from serialized dictionary"""

        # Convert radio_options to numpy array if present
        radio_options = data.get('radio_options')
        if radio_options is not None:
            radio_options = np.array(radio_options)

        return cls(
            width=data['width'],
            height=data['height'],
            fix_exposure_job=data['fix_exposure_job'],
            multi_face_job=data['multi_face_job'],
            auto_tilt_job=data['auto_tilt_job'],
            sensitivity=data['sensitivity'],
            face_percent=data['face_percent'],
            gamma=data['gamma'],
            top=data['top'],
            bottom=data['bottom'],
            left=data['left'],
            right=data['right'],
            radio_buttons=tuple(data['radio_buttons']),
            radio_options=(
                radio_options
                if radio_options is not None
                else field(default_factory=partial(np.array, cls.radio_options))
            ),
        )
