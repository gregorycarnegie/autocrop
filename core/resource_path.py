from __future__ import annotations

import sys
from functools import cached_property
from pathlib import Path
from typing import Self

_BasePath = type(Path())  # PosixPath | WindowsPath at runtime


class ResourcePath(_BasePath):
    """Path‑like helper that resolves to the correct on‑disk location
    whether the code is frozen by PyInstaller or running unpacked.

    Adapted from: https://stackoverflow.com/questions/31836104/pyinstaller-and-onefile-how-to-include-an-image-in-the-exe-file"""

    @cached_property
    def _base_dir(self) -> Path:
        return Path(getattr(sys, "_MEIPASS2", Path.cwd()))

    def as_resource(self) -> Self:     #  ← returns its own concrete subclass
        return type(self)(self._base_dir / self)
