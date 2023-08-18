import numpy as np
from pathlib import Path

from .file import File


class Video(File):
    default_directory = f'{Path.home()}\\Videos'
    file_types = np.array(['.avi', '.m4v', '.mkv', '.mov', '.mp4'])
