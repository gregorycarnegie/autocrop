import collections.abc as c
from pathlib import Path
from typing import Any, Union, Optional
import cv2.typing as cvt
from .job import Job

Box = tuple[int, int, int, int]  # (x0, y0, x1, y1)
type SaveFunction = c.Callable[[Any, Path, int, bool], None]
type CropFunction = c.Callable[
    [Union[cvt.MatLike, Path], Job, tuple[Any, Any]],
    Optional[Union[cvt.MatLike, c.Iterator[cvt.MatLike]]]
]