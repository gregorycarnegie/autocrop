import numpy as np
from pathlib import Path

from .file import File

class Video(File):
    def __init__(self):
        super().__init__()
        self.default_directory = f'{Path.home()}\\Videos'
        self.file_types = np.array(['.avi', '.m4v', '.mkv', '.mov', '.mp4'])
