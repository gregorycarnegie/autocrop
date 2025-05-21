# Add to a new file: core/crop_instruction.py
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass
class CropInstruction:
    """Serializable instructions for a crop operation"""
    file_path: str
    output_path: str
    bounding_box: tuple[int, int, int, int]  # x0, y0, x1, y1
    job_params: dict[str, Any]  # Serialized job parameters
    multi_face: bool
    face_index: int | None
    rotation_matrix: npt.NDArray[np.float64] | None = None
