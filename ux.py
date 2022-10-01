import shutil

from PyQt6.QtCore import QDir, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QIcon, QIntValidator, QFileSystemModel
from PyQt6.QtWidgets import QDialog, QFileDialog, QMainWindow, QMessageBox
from PyQt6 import uic
from cv2 import imwrite, resize, INTER_AREA
from utils import *


class Cropper(QObject):
    finished, progress = pyqtSignal(), pyqtSignal(int)

    def crop_dir(self, file_list, destination, line_3, line_4, slider_4, slider_3, slider_2, radio_choice, n, lines,
                 radio_choices, progress_bar):
        for v, image in enumerate(file_list, start=1):
            self.crop(image, False, destination, line_3.text(), line_4.text(), slider_4.value(),
                      slider_3.value(), slider_2.value(), radio_choice, n, lines, radio_choices)
            self.progress.emit(v)
            progress_bar.setValue(int(100 * v / len(file_list)))
        self.finished.emit()

    @staticmethod
    def crop(image, file_bool, destination, width, height, confidence, face, user_gam, radio, n, lines, radio_choices):
        source, image = os.path.split(image) if file_bool else (lines[n].text(), image)
        path = f'{source}\\{image}'
        bounding_box = box_detect(path, int(width), int(height), float(confidence), float(face))
        """Save the cropped image with PIL if a face was detected"""
        if bounding_box is not None:
            """Open image and check exif orientation and rotate accordingly"""
            pic = reorient_image(Image.open(path))
            """crop picture"""
            cropped_pic = pic.crop((bounding_box[2], bounding_box[0], bounding_box[3], bounding_box[1]))
            """Colour correct as Numpy array"""
            pic_array = cvtColor(np.array(cropped_pic), COLOR_BGR2RGB)
            cropped_image = resize(pic_array, (int(width), int(height)), interpolation=INTER_AREA)
            table = gamma(user_gam * GAMMA_THRESHOLD)
            cropped_image = LUT(cropped_image, table)
            if not os.path.exists(destination):
                os.makedirs(destination, mode=438, exist_ok=True)
            if radio == radio_choices[0]:
                imwrite(f'{destination}\\{image}', cropped_image)
            elif radio in radio_choices[1:]:
                name, extension = os.path.splitext(image)[0], radio
                imwrite(f'{destination}\\{name}{extension}', cropped_image)
        else:
            reject = f'{destination}\\reject'
            if not os.path.exists(reject):
                os.makedirs(reject, mode=438, exist_ok=True)
            to_file = f'{reject}\\{image}'
            shutil.copy(path, to_file)


class AboutDialog(QDialog):
    def __init__(self):
        super(AboutDialog, self).__init__()
        uic.loadUi('resources\\forms\\about_form.ui', self)
        self.pixmap = QPixmap('resources\\logos\\logo.png')
        self.label.setPixmap(self.pixmap)


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.worker = None
        self.thread = None
        uic.loadUi('resources\\forms\\form.ui', self)

        self.setWindowIcon(QIcon("resources\\logos\\logo.ico"))

        self.lines = {1: self.lineEdit_1, 2: self.lineEdit_2, 3: self.lineEdit_3, 4: self.lineEdit_4,
                      5: self.lineEdit_5}

        self.fileModel = QFileSystemModel(self)
        self.load_svgs()

        self.CropPushButton_1.setEnabled(False)
        self.CropPushButton_2.setEnabled(False)
        self.CancelPushButton.setEnabled(False)

        self.validator = QIntValidator(100, 10000)
        self.lineEdit_3.setValidator(self.validator)
        self.lineEdit_4.setValidator(self.validator)

        self.lineEdit_2.setText(default_dir)
        self.fileModel.setRootPath(default_dir)
        self.fileModel.setFilter(QDir.Filter.NoDotAndDotDot | QDir.Filter.Files)
        self.fileModel.setNameFilters(FILE_FILTER)

        self.treeView.setModel(self.fileModel)
        self.treeView.setRootIndex(self.fileModel.index(default_dir))
        self.treeView.selectionModel().selectionChanged.connect(
            lambda: display_crop(
                self.fileModel.filePath(self.treeView.currentIndex()),
                int(self.lineEdit_3.text()), int(self.lineEdit_4.text()), float(self.horizontalSlider_4.value()),
                float(self.horizontalSlider_3.value()), self.label_2, self.horizontalSlider_2.value()
            )
        )

        self.FilePushButton.clicked.connect(lambda: self.open_image())
        self.FolderPushButton.clicked.connect(lambda: self.open_folder(2))
        self.DestinationPushButton_1.clicked.connect(
            lambda: self.lineEdit_5.setText(QFileDialog.getExistingDirectory(self, 'Select Directory', default_dir)))
        self.AboutPushButton.clicked.connect(lambda: self.show_popup())

        self.CropPushButton_1.clicked.connect(
            lambda: Cropper().crop(
                self.lineEdit_1.text(), True,
                self.lineEdit_5.text(),
                self.lineEdit_3.text(),
                self.lineEdit_4.text(),
                self.horizontalSlider_4.value(),
                self.horizontalSlider_3.value(),
                self.horizontalSlider_2.value(),
                self.radio_choices[np.where(self.radio)[0][0]],
                1, self.lines, self.radio_choices
            )
        )

        self.CropPushButton_2.clicked.connect(
            lambda: self.folder_process(self.lineEdit_2.text(), self.lineEdit_5.text()))

        self.radio_choices = np.array(['No', '.bmp', '.jpg', '.png', '.webp'])
        self.radio_buttons = [self.radioButton_1, self.radioButton_2, self.radioButton_3,
                              self.radioButton_4, self.radioButton_5]
        self.radio = np.array([r.isChecked() for r in self.radio_buttons])

        self.radioButton_1.clicked.connect(lambda: self.radio_logic(0))
        self.radioButton_2.clicked.connect(lambda: self.radio_logic(1))
        self.radioButton_3.clicked.connect(lambda: self.radio_logic(2))
        self.radioButton_4.clicked.connect(lambda: self.radio_logic(3))
        self.radioButton_5.clicked.connect(lambda: self.radio_logic(4))

        for line_edit in [self.lineEdit_1, self.lineEdit_2, self.lineEdit_5]:
            line_edit. textChanged.connect(lambda: self.change_pushbutton())

        for slider in [self.horizontalSlider_2, self.horizontalSlider_3, self.horizontalSlider_4]:
            slider.valueChanged[int].connect(
                lambda: display_crop(
                    self.lineEdit_1.text(), int(self.lineEdit_3.text()),
                    int(self.lineEdit_4.text()), float(self.horizontalSlider_4.value()),
                    float(self.horizontalSlider_3.value()), self.label, self.horizontalSlider_2.value()
                )
            )

        self.reload_pushButton_1.clicked.connect(
            lambda: display_crop(
                self.lineEdit_1.text(), int(self.lineEdit_3.text()),
                int(self.lineEdit_4.text()), float(self.horizontalSlider_4.value()),
                float(self.horizontalSlider_3.value()), self.label, self.horizontalSlider_2.value()
            )
        )

        self.reload_pushButton_2.clicked.connect(
            lambda: display_crop(
                self.fileModel.filePath(self.treeView.currentIndex()),
                int(self.lineEdit_3.text()), int(self.lineEdit_4.text()), float(self.horizontalSlider_4.value()),
                float(self.horizontalSlider_3.value()), self.label_2, self.horizontalSlider_2.value()
            ) if self.fileModel.filePath(self.treeView.currentIndex()) not in {None, ''} else display_crop(
                f'{self.lineEdit_2.text()}\\{np.array([image for image in os.listdir(self.lineEdit_2.text()) if os.path.splitext(image)[1] in COMBINED_FILETYPES])[0]}',
                int(self.lineEdit_3.text()), int(self.lineEdit_4.text()), float(self.horizontalSlider_4.value()),
                float(self.horizontalSlider_3.value()), self.label_2, self.horizontalSlider_2.value()
            )
        )

    def load_svgs(self):
        x = {0: (self.reload_pushButton_1, 'reload.svg'), 1: (self.reload_pushButton_2, 'reload.svg'),
             2: (self.CropPushButton_1, 'crop.svg'), 3: (self.CropPushButton_2, 'crop.svg'),
             4: (self.DestinationPushButton_1, 'folder.svg'), 5: (self.FilePushButton, 'picture.svg'),
             6: (self.FolderPushButton, 'folder.svg'), 7: (self.CancelPushButton, 'cancel.svg'),
             8: (self.radioButton_1, 'no.svg'), 9: (self.radioButton_2, 'bmp.svg'),
             10: (self.radioButton_3, 'jpg.svg'), 11: (self.radioButton_4, 'png.svg'),
             12: (self.radioButton_5, 'webp.svg')}
        for i in range(len(x)):
            x[i][0].setIcon(QIcon(f'resources\\icons\\{x[i][1]}'))
        x = None

    def show_message_box(self):
        msg = QMessageBox()
        msg.setWindowTitle('Open Destination Folder')
        msg.setText('Open destination folder?')
        msg.setIcon(QMessageBox.Icon.Question)
        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        msg.buttonClicked.connect(self.message_button)
        x = msg.exec()

    def message_button(self, i):
        if i.text() == '&Yes':
            os.startfile(self.lineEdit_5.text())

    @staticmethod
    def show_popup():
        msg = AboutDialog()
        msg.exec()

    def change_pushbutton(self):
        if self.lineEdit_1.text() not in {'', None} and self.lineEdit_5.text() not in {'', None}:
            self.CropPushButton_1.setEnabled(True)
        else:
            self.CropPushButton_1.setEnabled(False)
        if self.lineEdit_2.text() not in {'', None} and self.lineEdit_5.text() not in {'', None}:
            self.CropPushButton_2.setEnabled(True)
        else:
            self.CropPushButton_2.setEnabled(False)

    def radio_logic(self, n):
        self.radio = [not _ if _ else _ for _ in self.radio]
        self.radio[n] = True

    def open_image(self):
        fname = QFileDialog.getOpenFileName(self, 'Open File', default_dir, file_type_list)
        self.lineEdit_1.setText(fname[0])
        display_crop(fname[0], int(self.lineEdit_3.text()),
                     int(self.lineEdit_4.text()), float(self.horizontalSlider_4.value()),
                     float(self.horizontalSlider_3.value()), self.label, self.horizontalSlider_2.value())

    def open_folder(self, n):
        fname = QFileDialog.getExistingDirectory(self, 'Select Directory', default_dir)
        self.fileModel.setRootPath(fname)
        self.treeView.setModel(self.fileModel)
        self.treeView.setRootIndex(self.fileModel.index(fname))
        self.lines[n].setText(fname)
        if n == 2:
            try:
                file = np.array([pic for pic in os.listdir(fname) if os.path.splitext(pic)[1] in COMBINED_FILETYPES])[0]
                display_crop(f'{fname}\\{file}', int(self.lineEdit_3.text()),
                             int(self.lineEdit_4.text()), float(self.horizontalSlider_4.value()),
                             float(self.horizontalSlider_3.value()), self.label_2, self.horizontalSlider_2.value())
            except (IndexError, FileNotFoundError):
                return

    def disable_ui(self, boolean: bool):
        x = [self.CancelPushButton, self.CropPushButton_2, self.reload_pushButton_2,
             self.FolderPushButton, self.DestinationPushButton_1,
             self.lineEdit_1, self.lineEdit_2, self.lineEdit_3, self.lineEdit_4, self.lineEdit_5,
             self.radioButton_1, self.radioButton_2, self.radioButton_3, self.radioButton_4, self.radioButton_5,
             self.horizontalSlider_2, self.horizontalSlider_3, self.horizontalSlider_4]
        if boolean:
            for i in range(len(x)):
                if i == 0:
                    x[0].setEnabled(True)
                else:
                    x[i].setEnabled(False)
        else:
            for i in range(len(x)):
                if i == 0:
                    x[0].setEnabled(False)
                else:
                    x[i].setEnabled(True)

    """ #################_CROPPING_FUNCTIONS_################# """

    def folder_process(self, source: str, destination: str):
        self.thread, self.worker = QThread(), Cropper()
        self.worker.moveToThread(self.thread)
        file_list = np.array([pic for pic in os.listdir(source) if os.path.splitext(pic)[1] in COMBINED_FILETYPES])
        self.thread.started.connect(lambda: self.disable_ui(True))
        self.thread.started.connect(
            lambda: self.worker.crop_dir(file_list, destination, self.lineEdit_3, self.lineEdit_4,
                                         self.horizontalSlider_4, self.horizontalSlider_3, self.horizontalSlider_2,
                                         self.radio_choices[np.where(self.radio)[0][0]], 2, self.lines,
                                         self.radio_choices, self.progressBar))
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()
        self.thread.finished.connect(lambda: self.disable_ui(False))
        self.thread.finished.connect(lambda: self.show_message_box())
