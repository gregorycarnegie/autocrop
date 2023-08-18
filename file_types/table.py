import numpy as np

from .file import File


class Table(File):
    file_types = np.array(['.csv', '.xlsx', '.xlsm', '.xltx', '.xltm'])
