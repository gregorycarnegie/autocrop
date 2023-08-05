import numpy as np

from .file import File

class Table(File):
    def __init__(self):
        super().__init__()
        self.file_types = np.array(['.csv', '.xlsx', '.xlsm', '.xltx', '.xltm'])
