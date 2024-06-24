from pathlib import Path
from typing import Any, Callable, Tuple, Union

import cv2.typing as cvt
import numpy as np
import numpy.typing as npt

Box = Tuple[int, int, int, int]
ImageArray = Union[cvt.MatLike, npt.NDArray[np.uint8]]
SaveFunction = Callable[[Any, Path, int, bool], None]
FaceToolPair = Tuple[Any, Any]
