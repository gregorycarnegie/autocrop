import collections.abc as c
from pathlib import Path
from typing import Any

import cv2

from .job import Job

Box = tuple[int, int, int, int]  # (x0, y0, x1, y1)
type CropFunction = c.Callable[
    [cv2.Mat | Path, Job, tuple[Any, Any]],
    cv2.Mat | c.Iterator[cv2.Mat] | None
]
