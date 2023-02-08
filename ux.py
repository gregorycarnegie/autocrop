from multiprocessing import Process

from PyQt6 import uic
from PyQt6.QtCore import QDir, QAbstractTableModel, Qt
from PyQt6.QtGui import QIcon, QIntValidator, QFileSystemModel, QPixmap
from PyQt6.QtWidgets import QDialog, QFileDialog, QMainWindow, QMessageBox

from cropper import Cropper
from settings import FileTypeList, SpreadSheet, default_dir
from utils import crop, display_crop, np, os, open_file


class PandasModel(QAbstractTableModel):
    def __init__(self, data):
        QAbstractTableModel.__init__(self)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if index.isValid() and role == Qt.ItemDataRole.DisplayRole:
            return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return self._data.columns[col]
        return None


class AboutDialog(QDialog):
    def __init__(self):
        super(AboutDialog, self).__init__()
        uic.loadUi('resources\\forms\\about_form.ui', self)
        self.setWindowIcon(QIcon("resources\\logos\\logo.ico"))
        self.pixmap = QPixmap('resources\\logos\\logo.png')
        self.label.setPixmap(self.pixmap)


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.data_frame = None

        self.msg = AboutDialog()

        self.ALL_PICTYPES = FileTypeList().all_types()
        self.PIC_FILTER = FileTypeList().file_filter
        self.pic_type_list = FileTypeList().type_string
        uic.loadUi('resources\\forms\\form.ui', self)

        self.setWindowIcon(QIcon("resources\\logos\\logo.ico"))

        self.lines = {1: self.lineEdit_1, 2: self.lineEdit_2, 3: self.lineEdit_3, 4: self.lineEdit_4,
                      5: self.lineEdit_5, 6: self.lineEdit_6, 7: self.lineEdit_7}

        self.fileModel = QFileSystemModel(self)
        self.load_svgs()

        self.process = Process()

        self.cropper = Cropper()
        self.cropper.started.connect(lambda: self.disable_ui(True))
        self.cropper.finished.connect(lambda: self.disable_ui(False))
        self.cropper.finished.connect(lambda: self.show_message_box())
        self.cropper.folder_progress.connect(self.update_progress_1)
        self.cropper.mapping_progress.connect(self.update_progress_2)

        self.CropPushButton_1.setEnabled(False)
        self.CropPushButton_2.setEnabled(False)
        self.CropPushButton_3.setEnabled(False)
        self.CancelPushButton_1.setEnabled(False)
        self.CancelPushButton_2.setEnabled(False)
        self.lineEdit_7.setEnabled(False)
        self.TablePushButton.setEnabled(False)

        self.CancelPushButton_1.clicked.connect(lambda: self.terminate())
        self.CancelPushButton_2.clicked.connect(lambda: self.terminate())

        self.CropPushButton_2.clicked.connect(lambda: self.disable_ui(True))
        self.CropPushButton_3.clicked.connect(lambda: self.disable_ui(True))

        self.validator = QIntValidator(100, 10000)
        self.lineEdit_3.setValidator(self.validator)
        self.lineEdit_4.setValidator(self.validator)

        self.lineEdit_2.setText(default_dir)
        self.fileModel.setRootPath(default_dir)
        self.fileModel.setFilter(QDir.Filter.NoDotAndDotDot | QDir.Filter.Files)
        self.fileModel.setNameFilters(self.PIC_FILTER)

        self.treeView.setModel(self.fileModel)
        self.treeView.setRootIndex(self.fileModel.index(default_dir))
        self.treeView.selectionModel().selectionChanged.connect(
            lambda: display_crop(self.fileModel.filePath(self.treeView.currentIndex()), int(self.lineEdit_3.text()),
                                 int(self.lineEdit_4.text()), self.hSlider_4.value(), self.hSlider_3.value(),
                                 self.crop_label_2, self.hSlider_2.value()))

        self.FilePushButton.clicked.connect(lambda: self.open_item(True))
        self.TablePushButton.clicked.connect(lambda: self.open_item(False))
        self.FolderPushButton_1.clicked.connect(lambda: self.open_folder(2))
        self.FolderPushButton_2.clicked.connect(lambda: self.open_folder(6))
        self.DestinationPushButton.clicked.connect(
            lambda: self.lineEdit_5.setText(QFileDialog.getExistingDirectory(self, 'Select Directory', default_dir)))

        self.actionAbout_Face_Cropper.triggered.connect(lambda: self.msg.exec())
        self.actionGolden_Ratio.triggered.connect(lambda: self.load_preset(0.5 * (1 + 5 ** 0.5)))
        self.action2_3_Ratio.triggered.connect(lambda: self.load_preset(1.5))
        self.action3_4_Ratio.triggered.connect(lambda: self.load_preset(4 / 3))
        self.action4_5_Ratio.triggered.connect(lambda: self.load_preset(1.25))
        self.actionSquare.triggered.connect(lambda: self.load_preset(1))

        self.actionCrop_File.triggered.connect(lambda: self.tabWidget.setCurrentIndex(0))
        self.actionCrop_Folder.triggered.connect(lambda: self.tabWidget.setCurrentIndex(1))
        self.actionUse_Mapping.triggered.connect(lambda: self.tabWidget.setCurrentIndex(2))

        self.CropPushButton_1.clicked.connect(
            lambda: crop(self.lineEdit_1.text(), True, self.lineEdit_5.text(), int(self.lineEdit_3.text()),
                         int(self.lineEdit_4.text()), self.hSlider_4.value(), self.hSlider_3.value(),
                         self.hSlider_2.value(), self.radio_choices[np.where(self.radio)[0][0]], 1, self.lines,
                         self.radio_choices))

        self.CropPushButton_2.clicked.connect(
            lambda: self.folder_process(self.lineEdit_2.text(), self.lineEdit_5.text()))

        self.CropPushButton_3.clicked.connect(
            lambda: self.mapping_process(self.lineEdit_6.text(), self.lineEdit_5.text()))

        self.radio_choices = np.array(['No', '.bmp', '.jpg', '.png', '.webp'])
        self.radio_buttons = [self.radioButton_1, self.radioButton_2, self.radioButton_3,
                              self.radioButton_4, self.radioButton_5]
        self.radio = np.array([r.isChecked() for r in self.radio_buttons])

        self.radioButton_1.clicked.connect(lambda: self.radio_logic(0))
        self.radioButton_2.clicked.connect(lambda: self.radio_logic(1))
        self.radioButton_3.clicked.connect(lambda: self.radio_logic(2))
        self.radioButton_4.clicked.connect(lambda: self.radio_logic(3))
        self.radioButton_5.clicked.connect(lambda: self.radio_logic(4))

        for line_edit in [self.lineEdit_1, self.lineEdit_2, self.lineEdit_5, self.lineEdit_6, self.lineEdit_7]:
            line_edit.textChanged.connect(lambda: self.change_pushbutton())

        for slider in [self.hSlider_2, self.hSlider_3, self.hSlider_4]:
            slider.valueChanged[int].connect(lambda: self.slider_update())

        self.reload_pushButton_1.clicked.connect(lambda: self.reload_1())
        self.reload_pushButton_2.clicked.connect(lambda: self.reload_2())
        self.reload_pushButton_3.clicked.connect(lambda: self.reload_3())

    def terminate(self):
        self.cropper.end_task = True
        self.disable_ui(False)
    
    def update_progress_1(self, value):
        self.progressBar_1.setValue(value)

    def update_progress_2(self, value):
        self.progressBar_2.setValue(value)

    def message_button(self, i):
        if i.text() == '&Yes':
            os.startfile(self.lineEdit_5.text())

    def show_message_box(self):
        msg = QMessageBox()
        self.configure_message_box(msg, 'Open Destination Folder', 'Open destination folder?')
        msg.buttonClicked.connect(self.message_button)
        x = msg.exec()

    @staticmethod
    def configure_message_box(msg: QMessageBox, arg1: str, arg2: str):
        msg.setWindowTitle(arg1)
        msg.setText(arg2)
        msg.setIcon(QMessageBox.Icon.Question)
        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

    def load_preset(self, phi):
        if phi == 1:
            if int(self.lineEdit_3.text()) > int(self.lineEdit_4.text()):
                self.lineEdit_4.setText(self.lineEdit_3.text())
            elif int(self.lineEdit_3.text()) < int(self.lineEdit_4.text()):
                self.lineEdit_3.setText(self.lineEdit_4.text())
        elif int(self.lineEdit_3.text()) >= int(self.lineEdit_4.text()):
            self.lineEdit_4.setText(str(int(float(self.lineEdit_3.text()) * phi)))
        elif int(self.lineEdit_3.text()) < int(self.lineEdit_4.text()):
            self.lineEdit_3.setText(str(int(float(self.lineEdit_4.text()) / phi)))
        self.slider_update()

    def slider_update(self):
        self.reload_1()
        self.reload_2()
        self.reload_3()

    def reload_1(self):
        display_crop(self.lineEdit_1.text(), int(self.lineEdit_3.text()), int(self.lineEdit_4.text()),
                     self.hSlider_4.value(), self.hSlider_3.value(), self.crop_label_1,
                     self.hSlider_2.value())

    def reload_2(self):
        if self.fileModel.filePath(self.treeView.currentIndex()) not in {None, ''}:
            display_crop(
                self.fileModel.filePath(self.treeView.currentIndex()), int(self.lineEdit_3.text()),
                int(self.lineEdit_4.text()), self.hSlider_4.value(), self.hSlider_3.value(),
                self.crop_label_2, self.hSlider_2.value())
        else:
            display_crop(
                os.path.join(self.lineEdit_2.text(), np.array([image for image in os.listdir(self.lineEdit_2.text()) if
                                                               os.path.splitext(image)[1] in self.ALL_PICTYPES])[0]),
                int(self.lineEdit_3.text()), int(self.lineEdit_4.text()), self.hSlider_4.value(),
                self.hSlider_3.value(), self.crop_label_2, self.hSlider_2.value())

    def reload_3(self):
        if self.lineEdit_6.text() not in {None, ''}:
            display_crop(
                os.path.join(self.lineEdit_6.text(), np.array([image for image in os.listdir(self.lineEdit_6.text()) if
                                                               os.path.splitext(image)[1] in self.ALL_PICTYPES])[0]),
                int(self.lineEdit_3.text()), int(self.lineEdit_4.text()), self.hSlider_4.value(),
                self.hSlider_3.value(), self.crop_label_3, self.hSlider_2.value())

    def load_svgs(self):
        x = {0: (self.reload_pushButton_1, 'reload.svg'), 1: (self.reload_pushButton_2, 'reload.svg'),
             2: (self.reload_pushButton_3, 'reload.svg'), 3: (self.CropPushButton_1, 'crop.svg'),
             4: (self.CropPushButton_2, 'crop.svg'), 5: (self.CropPushButton_3, 'crop.svg'),
             6: (self.DestinationPushButton, 'folder.svg'), 7: (self.FilePushButton, 'picture.svg'),
             8: (self.TablePushButton, 'excel.svg'), 9: (self.FolderPushButton_1, 'folder.svg'),
             10: (self.FolderPushButton_2, 'folder.svg'), 11: (self.CancelPushButton_1, 'cancel.svg'),
             12: (self.CancelPushButton_2, 'cancel.svg'), 13: (self.radioButton_1, 'no.svg'),
             14: (self.radioButton_2, 'bmp.svg'), 15: (self.radioButton_3, 'jpg.svg'),
             16: (self.radioButton_4, 'png.svg'), 17: (self.radioButton_5, 'webp.svg'),
             18: (self.actionGolden_Ratio, 'webp.svg'), 19: (self.actionSquare, 'webp.svg'),
             20: (self.action2_3_Ratio, 'webp.svg'), 21: (self.action3_4_Ratio, 'webp.svg'),
             22: (self.action4_5_Ratio, 'webp.svg'), 23: (self.actionCrop_File, 'picture.svg'),
             24: (self.actionCrop_Folder, 'folder.svg'), 25: (self.actionUse_Mapping, 'excel.svg')}
        for i in range(len(x)):
            x[i][0].setIcon(QIcon(f'resources\\icons\\{x[i][1]}'))

    def change_pushbutton(self):
        E = {'', None}
        if self.lineEdit_1.text() not in E and self.lineEdit_5.text() not in E:
            self.CropPushButton_1.setEnabled(True)
        else:
            self.CropPushButton_1.setEnabled(False)
        if self.lineEdit_2.text() not in E and self.lineEdit_5.text() not in E:
            self.CropPushButton_2.setEnabled(True)
        else:
            self.CropPushButton_2.setEnabled(False)
        if self.lineEdit_5.text() not in E and self.lineEdit_6.text() not in E and self.lineEdit_7.text() not in E:
            self.CropPushButton_3.setEnabled(True)
        else:
            self.CropPushButton_3.setEnabled(False)

    def radio_logic(self, n: int):
        self.radio = [not _ if _ else _ for _ in self.radio]
        self.radio[n] = True

    def open_item(self, boolean: bool):
        if boolean:
            f_name = QFileDialog.getOpenFileName(self, 'Open File', default_dir, FileTypeList().type_string())
            self.lineEdit_1.setText(f_name[0])
            display_crop(f_name[0], int(self.lineEdit_3.text()), int(self.lineEdit_4.text()),
                         self.hSlider_4.value(), self.hSlider_3.value(), self.crop_label_1,
                         self.hSlider_2.value())
        else:
            f_name = QFileDialog.getOpenFileName(self, 'Open File', default_dir, SpreadSheet().type_string())
            if os.path.splitext(f_name[0])[1] in SpreadSheet().get_json():
                self.lineEdit_7.setText(f_name[0])
                self.data_frame = open_file(f_name[0])
                self.tableView.setModel(PandasModel(self.data_frame))
                for _ in self.data_frame.columns:
                    self.comboBox_1.addItem(_)
                    self.comboBox_2.addItem(_)
                if self.lineEdit_6.text() not in {'', None} and os.path.exists(self.lineEdit_6.text()):
                    try:
                        display_crop(
                            os.path.join(self.lineEdit_6.text(), self.data_frame[self.data_frame.columns[0]][0]),
                            int(self.lineEdit_3.text()), int(self.lineEdit_4.text()),
                            self.hSlider_4.value(), self.hSlider_3.value(),
                            self.crop_label_3, self.hSlider_2.value())
                    except (IndexError, FileNotFoundError):
                        return

    def open_folder(self, n: int):
        f_name = QFileDialog.getExistingDirectory(self, 'Select Directory', default_dir)
        self.lines[n].setText(f_name)
        if n == 2:
            self.fileModel.setRootPath(f_name)
            self.treeView.setModel(self.fileModel)
            self.treeView.setRootIndex(self.fileModel.index(f_name))
            try:
                file = np.array([pic for pic in os.listdir(f_name) if os.path.splitext(pic)[1] in self.ALL_PICTYPES])[0]
                display_crop(f'{f_name}\\{file}', int(self.lineEdit_3.text()), int(self.lineEdit_4.text()),
                             self.hSlider_4.value(), self.hSlider_3.value(),
                             self.crop_label_2, self.hSlider_2.value())
            except (IndexError, FileNotFoundError):
                return
        elif n == 6:
            self.lineEdit_7.setEnabled(True)
            self.TablePushButton.setEnabled(True)

    def disable_ui(self, boolean: bool):
        x = [self.CancelPushButton_1, self.CancelPushButton_2, self.CropPushButton_2, self.CropPushButton_3,
             self.reload_pushButton_2, self.FolderPushButton_1, self.FolderPushButton_2, self.TablePushButton,
             self.DestinationPushButton, self.comboBox_1, self.comboBox_2, self.lineEdit_1, self.lineEdit_2,
             self.lineEdit_3, self.lineEdit_4, self.lineEdit_5, self.lineEdit_6, self.lineEdit_7,
             self.radioButton_1, self.radioButton_2, self.radioButton_3, self.radioButton_4, self.radioButton_5,
             self.hSlider_2, self.hSlider_3, self.hSlider_4]
        for i in range(len(x)):
            if boolean and i in {0, 1} or not boolean and i not in {0, 1}:
                x[i].setEnabled(True)
            else:
                x[i].setEnabled(False)

    def folder_process(self, source: str, destination: str):
        self.cropper.end_task = False
        self.cropper.message_box = True
        file_list = np.array([pic for pic in os.listdir(source) if os.path.splitext(pic)[1] in self.ALL_PICTYPES])
        self.process = Process(target=self.cropper.crop_dir, daemon=True,
                               args=(file_list, destination, int(self.lineEdit_3.text()), int(self.lineEdit_4.text()),
                                     self.hSlider_4.value(), self.hSlider_3.value(), self.hSlider_2.value(),
                                     self.radio_choices[np.where(self.radio)[0][0]], 2,
                                     self.lines, self.radio_choices))
        self.process.run()

    def mapping_process(self, source: str, destination: str):
        self.cropper.end_task = False
        self.cropper.message_box = True
        self.process = Process(target=self.cropper.mapping_crop, daemon=True,
                               args=(source, self.data_frame, self.comboBox_1.currentText(),
                                     self.comboBox_2.currentText(), destination, int(self.lineEdit_3.text()),
                                     int(self.lineEdit_4.text()), self.hSlider_4.value(), self.hSlider_3.value(),
                                     self.hSlider_2.value(), self.radio_choices[np.where(self.radio)[0][0]],
                                     self.radio_choices))
        self.process.run()
