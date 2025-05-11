import numpy as np
from numpy.typing import NDArray

def validate_files(
    file_paths: list[str],
    categories: list[int]
) -> NDArray[np.bool_]: ...

def verify_file_type(
    file_path: str,
    category: int
) -> bool: ...
