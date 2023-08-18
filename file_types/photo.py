import numpy as np

from .file import File


class Photo(File):
    SAVE_TYPES = {'.bmp', '.jpg', '.png', '.tiff', '.webp'}
    CV2_TYPES = np.array(['.bmp', '.dib', '.jpeg', '.jpg', '.jpe', '.jp2', '.png',
                          '.webp', '.pbm', '.pgm', '.ppm', '.pxm', '.pnm', '.pfm',
                          '.sr', '.ras', '.tiff', '.tif', '.exr', '.hdr', '.pic'])
    RAW_TYPES = np.array(['.dng', '.arw', '.cr2', '.crw', '.erf',
                          '.kdc', '.nef', '.nrw', '.orf', '.pef',
                          '.raf', '.raw', '.sr2', '.srw', '.x3f'])
    file_types = np.concatenate((CV2_TYPES, RAW_TYPES))
