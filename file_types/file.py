from pathlib import Path

import numpy as np
import numpy.typing as npt

class File:
    def __init__(self) -> None:
        self.default_directory = f'{Path.home()}\\Pictures'
        self.file_types: npt.NDArray[np.str] = None

    @property
    def file_filter(self) -> npt.NDArray[np.str_]:
        return np.array([f'*{file}' for file in self.file_types])

    @property
    def type_string(self) -> str:
        return 'All Files (*);;' + ';;'.join(f'{_} Files (*{_})' for _ in np.sort(self.file_types))
