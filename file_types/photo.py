from pathlib import Path

import numpy as np
import numpy.typing as npt

from .file import File

# CV2_TYPES = np.array(['.bmp', '.dib', '.jpeg', '.jpg', '.jpe', '.jp2', '.png',
#                       '.webp', '.pbm', '.pgm', '.ppm', '.pxm', '.pnm', '.pfm',
#                       '.sr', '.ras', '.tiff', '.tif', '.exr', '.hdr', '.pic'])
# RAW_TYPES = np.array(['.dng', '.arw', '.cr2', '.crw', '.erf',
#                       '.kdc', '.nef', '.nrw', '.orf', '.pef',
#                       '.raf', '.raw', '.sr2', '.srw', '.x3f'])
# IMAGE_TYPES = np.concatenate((CV2_TYPES, RAW_TYPES))

class Photo(File):
    def __init__(self) -> None:
        super().__init__()
        self.CV2_TYPES = np.array(['.bmp', '.dib', '.jpeg', '.jpg', '.jpe', '.jp2', '.png',
                                   '.webp', '.pbm', '.pgm', '.ppm', '.pxm', '.pnm', '.pfm',
                                   '.sr', '.ras', '.tiff', '.tif', '.exr', '.hdr', '.pic'])
        self.RAW_TYPES = np.array(['.dng', '.arw', '.cr2', '.crw', '.erf',
                                   '.kdc', '.nef', '.nrw', '.orf', '.pef',
                                   '.raf', '.raw', '.sr2', '.srw', '.x3f'])
        self.file_types = np.concatenate((self.CV2_TYPES, self.RAW_TYPES))
        # self.default_directory = f'{Path.home()}\\Pictures'

    # @property
    # def file_filter(self) -> npt.NDArray[np.str_]:
    #     return np.array([f'*{file}' for file in self.file_types])
    #
    # @property
    # def type_string(self) -> str:
    #     return 'All Files (*);;' + ';;'.join(f'{_} Files (*{_})' for _ in np.sort(self.file_types))
