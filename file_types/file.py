from typing import Optional

from pathlib import Path

import numpy as np
import numpy.typing as npt


class File:
    file_types: npt.NDArray[np.str_] = np.array([''])
    default_directory = f'{Path.home()}\\Pictures'

    @classmethod
    def file_filter(cls) -> npt.NDArray[np.str_]:
        """Only sublasses of this class should implement this method"""
        return np.array([f'*{file}' for file in cls.file_types])

    @classmethod
    def type_string(cls) -> Optional[str]:
        """Only sublasses of this class should implement this method"""
        if not cls.file_types[0]:
            return None
        return 'All Files (*);;' + ';;'.join(f'{_} Files (*{_})' for _ in np.sort(cls.file_types))
