import numpy as np

from .file import File

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
