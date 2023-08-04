import numpy as np

from .file import File

# PANDAS_TYPES = np.array(['.csv', '.xlsx', '.xlsm', '.xltx', '.xltm'])

class Table(File):
    def __init__(self):
        super().__init__()
        self.file_types = np.array(['.csv', '.xlsx', '.xlsm', '.xltx', '.xltm'])
