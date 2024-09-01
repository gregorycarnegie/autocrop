import collections.abc as c
from pathlib import Path
from typing import Any, Union

import cv2.typing as cvt
import dlib
import numpy as np
import numpy.typing as npt

Box = tuple[int, int, int, int]
ImageArray = Union[cvt.MatLike, npt.NDArray[np.uint8]]
SaveFunction = c.Callable[[Any, Path, int, bool], None]
FaceToolPair = tuple[dlib.fhog_object_detector, dlib.shape_predictor]
