import collections.abc as c
from pathlib import Path
from typing import Any

Box = tuple[int, int, int, int]  # (x0, y0, x1, y1)
SaveFunction = c.Callable[[Any, Path, int, bool], None]
