from pathlib import Path

import numpy as np
import numpy.typing as npt

CV2_TYPES = np.array(['.bmp', '.dib', '.jpeg', '.jpg', '.jpe', '.jp2', '.png',
                      '.webp', '.pbm', '.pgm', '.ppm', '.pxm', '.pnm', '.pfm',
                      '.sr', '.ras', '.tiff', '.tif', '.exr', '.hdr', '.pic'])
RAW_TYPES = np.array(['.dng', '.arw', '.cr2', '.crw', '.erf',
                      '.kdc', '.nef', '.nrw', '.orf', '.pef',
                      '.raf', '.raw', '.sr2', '.srw', '.x3f'])
IMAGE_TYPES = np.concatenate((CV2_TYPES, RAW_TYPES))

class Photo:
    def __init__(self) -> None:
        self.default_directory = f'{Path.home()}\\Pictures'

    @staticmethod
    def file_filter() -> npt.NDArray[np.str_]:
        return np.array([f'*{file}' for file in IMAGE_TYPES])

    @staticmethod
    def type_string() -> str:
        return 'All Files (*);;' + ';;'.join(f'{_} Files (*{_})' for _ in np.sort(IMAGE_TYPES))
