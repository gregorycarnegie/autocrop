
import sys
from pathlib import Path


class ResourcePath:
    def __init__(self, relative_path: str) -> None:
        self._relative_path = relative_path

    @property
    def meipass_path(self) -> str:
        base_path = Path(getattr(sys, '_MEIPASS2', Path().resolve()))
        return (base_path / self._relative_path).as_posix()