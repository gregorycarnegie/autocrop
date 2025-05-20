import collections.abc as c
from pathlib import Path
from typing import Any

import cv2.typing as cvt

from .job import Job

Box = tuple[int, int, int, int]  # (x0, y0, x1, y1)
type CropFunction = c.Callable[
    [cvt.MatLike | Path, Job, tuple[Any, Any]],
    cvt.MatLike | c.Iterator[cvt.MatLike] | None
]
Pipeline = list[c.Callable[[cvt.MatLike], cvt.MatLike]]
