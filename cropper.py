from multiprocessing import cpu_count
from threading import Thread

from PyQt6.QtCore import pyqtSignal, QObject

from settings import FileTypeList
from utils import crop, np, os, m_crop, pd


class Cropper(QObject):
    started, finished = pyqtSignal(), pyqtSignal()
    folder_progress, mapping_progress = pyqtSignal(int), pyqtSignal(int)
    
    def __init__(self, parent = None):
        super(Cropper, self).__init__(parent)
        self.bar_value = 0

    def cropdir(self, files: int, file_list: np.ndarray, destination: str, line_3: int, line_4: int,
                slider_4: int, slider_3: int, slider_2: int, radio_choice: str, n: int, lines: dict,
                radio_choices: np.ndarray):
        for image in file_list:
            crop(image, False, destination, line_3, line_4, slider_4, slider_3, slider_2, radio_choice, n, lines,
                 radio_choices)
            self.bar_value += 1
            x = int(100 * self.bar_value / files)
            self.folder_progress.emit(x)

        if self.bar_value == files:
            self.finished.emit()

    def crop_dir(self, file_list: np.ndarray, destination: str, line_3: int, line_4: int, slider_4: int, slider_3: int,
                 slider_2: int, radio_choice: str, n: int, lines: dict, radio_choices: np.ndarray):
        self.started.emit()
        split_array = np.array_split(file_list, cpu_count())
        threads = []
        file_amount = len(file_list)
        self.bar_value = 0
        self.folder_progress.emit(self.bar_value)
        for array in split_array:
            t = Thread(target=self.cropdir, args=(file_amount, array, destination, line_3, line_4, slider_4, slider_3,
                                                  slider_2, radio_choice, n, lines, radio_choices))
            threads.append(t)
            t.start()

    def map_crop(self, files: int, source_folder, old, new, destination, width, height, confidence, face, user_gam,
                 radio, radio_choices):
        for i, image in enumerate(old):
            m_crop(source_folder, image, new[i], destination, width, height, confidence,
                   face, user_gam, radio, radio_choices)
            self.bar_value += 1
            x = int(100 * self.bar_value / files)
            self.mapping_progress.emit(x)
        
        if self.bar_value == files:
            self.finished.emit()

    def mapping_crop(self, source_folder: str, data: pd.DataFrame, name_column: str, mapping: str,
                     destination: str, width: int, height: int, confidence: int, face: int, user_gam: int, radio: str,
                     radio_choices: np.ndarray):
        self.started.emit()
        file_list = np.array(data[name_column]).astype(str)
        extensions = np.char.lower([os.path.splitext(i)[1] for i in file_list])
        types = FileTypeList().all_types()

        r, s = np.meshgrid(extensions, types)
        g = r == s
        h = [g[:, i].any() for i in range(len(file_list))]

        old, new = np.array_split(file_list[h], cpu_count()), np.array_split(np.array(data[mapping])[h], cpu_count())
        threads = []
        file_amount = len(file_list[h])
        self.bar_value = 0
        self.folder_progress.emit(self.bar_value)
        for i, _ in enumerate(old):
            t = Thread(target=self.map_crop, args=(file_amount, source_folder, _, new[i], destination, width, height,
                                                   confidence, face, user_gam, radio, radio_choices))
            threads.append(t)
            t.start()
