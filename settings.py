import numpy as np
import json
from pathlib import Path


class FileTypeList:
    def __init__(self):
        self.list1, self.list2 = self.get_json()[0], self.get_json()[1]
        self.file_filter = np.array([f'*{file}' for file in self.all_types()])
        self.file_inputs = np.append(self.all_types(), np.char.upper(self.all_types()))

    def type_string(self):
        return f"All Files (*){''.join(f';;{_} Files (*{_})' for _ in np.sort(self.all_types()))}"

    def all_types(self):
        return np.append(self.list1, self.list2)

    @staticmethod
    def get_json():
        with open(f'{Path.cwd()}\\settings.json', 'r') as f:
            data = json.load(f)
            return [data['PIL_TYPES'], data['CV2_TYPES']]


class SpreadSheet(FileTypeList):
    def __init__(self):
        super().__init__()
        self.list1, self.list2 = self.get_json()[0], self.get_json()[1:]

    @staticmethod
    def get_json():
        with open(f'{Path.cwd()}\\settings.json', 'r') as f:
            data = json.load(f)
            return data['PANDAS']


def get_json():
    with open(f'{Path.cwd()}\\settings.json', 'r') as f:
        data = json.load(f)
        return [data['GAMMA'], data['GAMMA_THRESHOLD'], f"{Path.home()}{data['default_directory']}",
                f"{Path.cwd()}{data['proto_path']}", f"{Path.cwd()}{data['caffe_path']}"]


GAMMA, GAMMA_THRESHOLD, default_dir, proto_path, caffe_path = get_json()
