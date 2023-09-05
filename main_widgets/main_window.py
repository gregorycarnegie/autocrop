from pathlib import Path
from typing import Union, Tuple, Optional

from PyQt6 import QtCore, QtGui, QtWidgets

from core import Cropper, CustomDialWidget, ExtWidget, FunctionType, Preset
from core import utils as ut
from core import window_functions as wf
from file_types import Photo, Table, Video
from line_edits import NumberLineEdit, PathLineEdit, LineEditState
from .crop_folder_widget import CropFolderWidget
from .crop_map_widget import CropMapWidget
from .crop_photo_widget import CropPhotoWidget
from .crop_vid_widget import CropVideoWidget
from .custom_crop_widget import CustomCropWidget


class UiMainWindow(QtWidgets.QMainWindow):
    """
    Represents the main window of the application.

    The main window contains various widgets and handles drag and drop events, as well as button and combo box interactions.

    Args:
        self: The instance of the class.

    Example:
        ```python
        main_window = MainWindow()
        main_window.show()
        ```
    """

    def __init__(self) -> None:
        super(UiMainWindow, self).__init__()
        self.setAcceptDrops(True)
        self.crop_worker = Cropper()
        self.setObjectName('MainWindow')
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap('resources\\logos\\logo.ico'), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(parent=self)
        self.centralwidget.setObjectName('centralwidget')
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName('horizontalLayout_2')
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName('verticalLayout_2')
        self.function_tabWidget = QtWidgets.QTabWidget(parent=self.centralwidget)
        self.function_tabWidget.setMovable(True)
        self.function_tabWidget.setObjectName('function_tabWidget')
        self.verticalLayout_2.addWidget(self.function_tabWidget)
        self.settings_tabWidget = QtWidgets.QTabWidget(parent=self.centralwidget)
        self.settings_tabWidget.setTabShape(QtWidgets.QTabWidget.TabShape.Rounded)
        self.settings_tabWidget.setTabsClosable(False)
        self.settings_tabWidget.setMovable(True)
        self.settings_tabWidget.setTabBarAutoHide(False)
        self.settings_tabWidget.setObjectName('')
        self.settingsTab = QtWidgets.QWidget()
        self.settingsTab.setObjectName('settingsTab')
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.settingsTab)
        self.horizontalLayout_3.setObjectName('horizontalLayout_3')
        self.gamma_dialArea = CustomDialWidget(
            _label='gamma', _max=2_000, single_step=5, page_step=100, _value=1_000, parent=self.settingsTab)
        self.face_dialArea = CustomDialWidget(_label='face', _min=0, _value=62, parent=self.settingsTab)
        self.sensitivity_dialArea = CustomDialWidget(
            _label='sensitivity', _min=0, _value=50, parent=self.settingsTab)
        wf.add_widgets(self.horizontalLayout_3, self.gamma_dialArea, self.face_dialArea, self.sensitivity_dialArea)
        spacerItem = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem)
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setObjectName('verticalLayout_6')
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName('horizontalLayout_9')
        self.verticalLayout_9 = QtWidgets.QVBoxLayout()
        self.verticalLayout_9.setObjectName('verticalLayout_9')
        self.label_4 = QtWidgets.QLabel(parent=self.settingsTab)
        self.label_4.setObjectName('label_4')
        self.verticalLayout_9.addWidget(self.label_4)
        self.widthLineEdit = NumberLineEdit(name='widthLineEdit', parent=self.settingsTab)
        self.verticalLayout_9.addWidget(self.widthLineEdit)
        self.horizontalLayout_9.addLayout(self.verticalLayout_9)
        self.verticalLayout_10 = QtWidgets.QVBoxLayout()
        self.verticalLayout_10.setObjectName('verticalLayout_10')
        self.label_5 = QtWidgets.QLabel(parent=self.settingsTab)
        self.label_5.setObjectName('label_5')
        self.verticalLayout_10.addWidget(self.label_5)
        self.heightLineEdit = NumberLineEdit(name='heightLineEdit', parent=self.settingsTab)
        self.verticalLayout_10.addWidget(self.heightLineEdit)
        self.horizontalLayout_9.addLayout(self.verticalLayout_10)
        self.verticalLayout_6.addLayout(self.horizontalLayout_9)
        self.label_7 = QtWidgets.QLabel(parent=self.settingsTab)
        self.label_7.setObjectName('label_7')
        self.verticalLayout_6.addWidget(self.label_7, 0, QtCore.Qt.AlignmentFlag.AlignHCenter)
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName('horizontalLayout_10')
        self.top_dialArea = CustomDialWidget(_label='top', _min=0, parent=self.settingsTab)
        self.bottom_dialArea = CustomDialWidget(_label='bottom', _min=0, parent=self.settingsTab)
        self.left_dialArea = CustomDialWidget(_label='left', _min=0, parent=self.settingsTab)
        self.right_dialArea = CustomDialWidget(_label='right', _min=0, parent=self.settingsTab)
        wf.add_widgets(self.horizontalLayout_10, self.top_dialArea, self.bottom_dialArea, self.left_dialArea,
                       self.right_dialArea)
        self.verticalLayout_6.addLayout(self.horizontalLayout_10)
        self.horizontalLayout_3.addLayout(self.verticalLayout_6)
        self.horizontalLayout_3.setStretch(0, 1)
        self.horizontalLayout_3.setStretch(1, 1)
        self.horizontalLayout_3.setStretch(2, 1)
        self.horizontalLayout_3.setStretch(4, 2)
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap('resources\\icons\\settings.svg'), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.settings_tabWidget.addTab(self.settingsTab, icon5, '')
        self.formatTab = QtWidgets.QWidget()
        self.formatTab.setObjectName('formatTab')
        self.horizontalLayout_27 = QtWidgets.QHBoxLayout(self.formatTab)
        self.horizontalLayout_27.setObjectName('horizontalLayout_27')
        self.extWidget = ExtWidget(parent=self.formatTab)
        self.extWidget.setObjectName('extWidget')
        self.horizontalLayout_27.addWidget(self.extWidget)

        # Tab Widgets
        self.photoTab, icon1 = self.setup_custom_crop_widget(self.function_tabWidget, FunctionType.PHOTO)
        self.folder_Tab, icon2 = self.setup_custom_crop_widget(self.function_tabWidget, FunctionType.FOLDER)
        self.mappingTab, icon3 = self.setup_custom_crop_widget(self.function_tabWidget, FunctionType.MAPPING)
        self.videoTab, icon4 = self.setup_custom_crop_widget(self.function_tabWidget, FunctionType.VIDEO)

        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap('resources\\icons\\memory_card.svg'),
                        QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.settings_tabWidget.addTab(self.formatTab, icon6, '')
        self.verticalLayout_2.addWidget(self.settings_tabWidget)
        self.verticalLayout_2.setStretch(0, 13)
        self.verticalLayout_2.setStretch(1, 3)
        self.horizontalLayout_2.addLayout(self.verticalLayout_2)
        self.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(parent=self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1_348, 22))
        self.menubar.setObjectName('menubar')
        self.menuFile = QtWidgets.QMenu(parent=self.menubar)
        self.menuFile.setObjectName('menuFile')
        self.menuTools = QtWidgets.QMenu(parent=self.menubar)
        self.menuTools.setObjectName('menuTools')
        self.menuInfo = QtWidgets.QMenu(parent=self.menubar)
        self.menuInfo.setObjectName('menuInfo')
        self.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(parent=self)
        self.statusbar.setObjectName('statusbar')
        self.setStatusBar(self.statusbar)
        self.actionAbout_Face_Cropper = QtGui.QAction(parent=self)
        self.actionAbout_Face_Cropper.setObjectName('actionAbout_Face_Cropper')
        self.actionUse_Mapping = QtGui.QAction(parent=self)
        self.actionUse_Mapping.setIcon(icon3)
        self.actionUse_Mapping.setObjectName('actionUse_Mapping')
        self.actionCrop_File = QtGui.QAction(parent=self)
        self.actionCrop_File.setIcon(icon1)
        self.actionCrop_File.setObjectName('actionCrop_File')
        self.actionCrop_Folder = QtGui.QAction(parent=self)
        self.actionCrop_Folder.setIcon(icon2)
        self.actionCrop_Folder.setObjectName('actionCrop_Folder')
        self.actionSquare = QtGui.QAction(parent=self)
        self.actionSquare.setObjectName('actionSquare')
        self.actionGolden_Ratio = QtGui.QAction(parent=self)
        self.actionGolden_Ratio.setObjectName('actionGolden_Ratio')
        self.action2_3_Ratio = QtGui.QAction(parent=self)
        self.action2_3_Ratio.setObjectName('action2_3_Ratio')
        self.action3_4_Ratio = QtGui.QAction(parent=self)
        self.action3_4_Ratio.setObjectName('action3_4_Ratio')
        self.action4_5_Ratio = QtGui.QAction(parent=self)
        self.action4_5_Ratio.setObjectName('action4_5_Ratio')
        self.actionCrop_Video = QtGui.QAction(parent=self)
        self.actionCrop_Video.setIcon(icon4)
        self.actionCrop_Video.setObjectName('actionCrop_Video')
        self.menuFile.addAction(self.actionSquare)
        self.menuFile.addAction(self.actionGolden_Ratio)
        self.menuFile.addAction(self.action2_3_Ratio)
        self.menuFile.addAction(self.action3_4_Ratio)
        self.menuFile.addAction(self.action4_5_Ratio)
        self.menuTools.addAction(self.actionUse_Mapping)
        self.menuTools.addAction(self.actionCrop_File)
        self.menuTools.addAction(self.actionCrop_Folder)
        self.menuTools.addAction(self.actionCrop_Video)
        self.menuInfo.addAction(self.actionAbout_Face_Cropper)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuTools.menuAction())
        self.menubar.addAction(self.menuInfo.menuAction())

        # CONNECTIONS
        self.connect_combo_boxes(self.mappingTab)

        self.actionAbout_Face_Cropper.triggered.connect(lambda: wf.load_about_form())
        self.actionGolden_Ratio.triggered.connect(lambda: self.load_preset(Preset.GOLDEN_RATIO))
        self.action2_3_Ratio.triggered.connect(lambda: self.load_preset(Preset.TWO_THIRDS))
        self.action3_4_Ratio.triggered.connect(lambda: self.load_preset(Preset.THREE_QUARTERS))
        self.action4_5_Ratio.triggered.connect(lambda: self.load_preset(Preset.FOUR_FIFTHS))
        self.actionSquare.triggered.connect(lambda: self.load_preset(Preset.SQUARE))
        self.actionCrop_File.triggered.connect(
            lambda: self.function_tabWidget.setCurrentIndex(FunctionType.PHOTO.value))
        self.actionCrop_Folder.triggered.connect(
            lambda: self.function_tabWidget.setCurrentIndex(FunctionType.FOLDER.value))
        self.actionUse_Mapping.triggered.connect(
            lambda: self.function_tabWidget.setCurrentIndex(FunctionType.MAPPING.value))
        self.actionCrop_Video.triggered.connect(
            lambda: self.function_tabWidget.setCurrentIndex(FunctionType.VIDEO.value))

        self.function_tabWidget.currentChanged.connect(lambda: self.check_tab_selection())
        self.function_tabWidget.currentChanged.connect(lambda: self.videoTab.player.pause())

        self.retranslateUi()
        self.function_tabWidget.setCurrentIndex(FunctionType.PHOTO.value)
        self.settings_tabWidget.setCurrentIndex(0)
        self.actionCrop_File.triggered.connect(self.function_tabWidget.setFocus)  # type: ignore
        self.actionCrop_Folder.triggered.connect(self.function_tabWidget.setFocus)  # type: ignore
        self.actionCrop_Video.triggered.connect(self.function_tabWidget.setFocus)  # type: ignore
        self.actionUse_Mapping.triggered.connect(self.function_tabWidget.setFocus)  # type: ignore
        QtCore.QMetaObject.connectSlotsByName(self)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate('MainWindow', 'MainWindow'))
        self.function_tabWidget.setTabText(
            self.function_tabWidget.indexOf(self.photoTab), _translate('MainWindow', 'Photo Crop'))
        self.function_tabWidget.setTabText(
            self.function_tabWidget.indexOf(self.folder_Tab), _translate('MainWindow', 'Folder Crop'))
        self.function_tabWidget.setTabText(
            self.function_tabWidget.indexOf(self.mappingTab), _translate('MainWindow', 'Mapping Crop'))
        self.function_tabWidget.setTabText(
            self.function_tabWidget.indexOf(self.videoTab), _translate('MainWindow', 'Video Crop'))
        self.label_4.setText(_translate('MainWindow', 'Width (px)'))
        self.widthLineEdit.setPlaceholderText(_translate('MainWindow', 'Try typing a number e.g. 400'))
        self.label_5.setText(_translate('MainWindow', 'Height (px)'))
        self.heightLineEdit.setPlaceholderText(_translate('MainWindow', 'Try typing a number e.g. 400'))
        self.label_7.setText(_translate('MainWindow', 'Padding'))
        self.settings_tabWidget.setTabText(
            self.settings_tabWidget.indexOf(self.settingsTab), _translate('MainWindow', 'Settings'))
        self.settings_tabWidget.setTabText(
            self.settings_tabWidget.indexOf(self.formatTab), _translate('MainWindow', 'Format Conversion'))
        self.menuFile.setTitle(_translate('MainWindow', 'Presets'))
        self.menuTools.setTitle(_translate('MainWindow', 'Tools'))
        self.menuInfo.setTitle(_translate('MainWindow', 'Info'))
        self.actionAbout_Face_Cropper.setText(_translate('MainWindow', 'About Face Cropper'))
        self.actionUse_Mapping.setText(_translate('MainWindow', 'Use Mapping'))
        self.actionCrop_File.setText(_translate('MainWindow', 'Crop File'))
        self.actionCrop_Folder.setText(_translate('MainWindow', 'Crop Folder'))
        self.actionSquare.setText(_translate('MainWindow', 'Square'))
        self.actionGolden_Ratio.setText(_translate('MainWindow', 'Golden Ratio'))
        self.action2_3_Ratio.setText(_translate('MainWindow', '2:3 Ratio'))
        self.action3_4_Ratio.setText(_translate('MainWindow', '3:4 Ratio'))
        self.action4_5_Ratio.setText(_translate('MainWindow', '4:5 Ratio'))
        self.actionCrop_Video.setText(_translate('MainWindow', 'Crop Video'))

    def adjust_ui(self, app: QtWidgets.QApplication):        
        if (screen := app.primaryScreen()) is None:
            return
        size = screen.size()
        width, height = size.width(), size.height()
        base_font_size = 12

        # Adjust based on screen resolution
        if width >= 3_840:
            self.resize(width >> 1, height >> 1)
        else:
            self.resize(3 * width >> 2, 3 * height >> 2)

        if height > 1_080:
            scale_factor = height / 1_080
            base_font_size = int(base_font_size * scale_factor)

        font = app.font()
        font.setPointSize(base_font_size)
        app.setFont(font)
    
    def dragEnterEvent(self, a0: Optional[QtGui.QDragEnterEvent]) -> None:
        """
        Handles the drag enter event by checking the event type and calling the check_mime_data method.

        Args:
            self: The instance of the class.
            a0 (Optional[QtGui.QDragEnterEvent]): The drag enter event.

        Returns:
            None
        """

        try:
            assert isinstance(a0, QtGui.QDragEnterEvent)
        except AssertionError:
            return
        wf.check_mime_data(a0)

    def dragMoveEvent(self, a0: Optional[QtGui.QDragMoveEvent]) -> None:
        """
        Handles the drag move event by checking the event type and calling the check_mime_data method.

        Args:
            self: The instance of the class.
            a0 (Optional[QtGui.QDragMoveEvent]): The drag move event.

        Returns:
            None
        """

        try:
            assert isinstance(a0, QtGui.QDragMoveEvent)
        except AssertionError:
            return
        wf.check_mime_data(a0)

    def dropEvent(self, a0: Optional[QtGui.QDropEvent]) -> None:
        """
        Handles the drop event by checking the dropped item, setting the appropriate drop action, and calling the corresponding handler method.

        Args:
            self: The instance of the class.
            a0 (Optional[QtGui.QDropEvent]): The drop event.

        Returns:
            None
        """

        try:
            assert isinstance(a0, QtGui.QDropEvent)
        except AssertionError:
            return

        if (x := a0.mimeData()) is None:
            return

        if not x.hasUrls():
            a0.ignore()
            return

        a0.setDropAction(QtCore.Qt.DropAction.CopyAction)
        file_path = Path(x.urls()[0].toLocalFile())
        if file_path.is_dir():
            self.handle_path_main(file_path)
        elif file_path.is_file():
            self.handle_file(file_path)
        a0.accept()

    def handle_path(self, file_path: Path,
                    tab_index: FunctionType,
                    line_edit: PathLineEdit) -> None:
        """
        Handles a file path by setting the function tab widget to the specified tab index, updating the line edit with the file path, and handling the selection state of the tabs.

        Args:
            self: The instance of the class.
            file_path (Path): The path to the file.
            tab_index (FunctionType): The index of the tab to set.
            line_edit (PathLineEdit): The line edit to update.

        Returns:
            None
        """

        self.function_tabWidget.setCurrentIndex(tab_index.value)
        line_edit.setText(file_path.as_posix())

        try:
            assert isinstance(self.mappingTab, CropMapWidget)
            assert isinstance(self.folder_Tab, CropFolderWidget)
            assert isinstance(self.photoTab, CropPhotoWidget)
        except AssertionError:
            return

        if self.photoTab.selection_state == self.photoTab.SELECTED:
            self.handle_function_tab_state(self.photoTab, self.folder_Tab, self.mappingTab, self.videoTab)
            self.photoTab.display_crop()
        elif self.folder_Tab.selection_state == self.folder_Tab.SELECTED:
            self.handle_function_tab_state(self.folder_Tab, self.photoTab, self.mappingTab, self.videoTab)
            self.folder_Tab.load_data()
        elif self.mappingTab.selection_state == self.mappingTab.SELECTED:
            self.handle_function_tab_state(self.mappingTab, self.photoTab, self.folder_Tab, self.videoTab)
            self.mappingTab.display_crop()

    def handle_path_main(self, file_path: Path) -> None:
        """
        Handles a file path by checking the file extensions in the directory, validating the mapping and folder tabs, and calling the handle_path method with the appropriate arguments.

        Args:
            self: The instance of the class.
            file_path (Path): The path to the file.

        Returns:
            None
        """

        extensions = {y.suffix.lower() for y in file_path.iterdir()}
        mask = {ext in extensions for ext in Table.file_types}
        try:
            assert isinstance(self.mappingTab, CropMapWidget)
            assert isinstance(self.folder_Tab, CropFolderWidget)
        except AssertionError:
            return
        if any(mask):
            self.handle_path(file_path, FunctionType.MAPPING, self.mappingTab.folderLineEdit)
        else:
            self.handle_path(file_path, FunctionType.FOLDER, self.folder_Tab.folderLineEdit)

    def handle_file(self, file_path: Path) -> None:
        """
        Handles a file based on its file extension by calling the appropriate handler method.

        Args:
            self: The instance of the class.
            file_path (Path): The path to the file.

        Returns:
            None
        """

        match file_path.suffix.lower():
            case suffix if suffix in Photo.file_types:
                self.handle_image_file(file_path)
            case suffix if suffix in Video.file_types:
                self.handle_video_file(file_path)
            case suffix if suffix in Table.file_types:
                self.handle_pandas_file(file_path)
            case _:
                pass

    def handle_image_file(self, file_path: Path) -> None:
        """
        Handles an image file by validating the file path and calling the handle_path method with the appropriate arguments.

        Args:
            self: The instance of the class.
            file_path (Path): The path to the image file.

        Returns:
            None
        """

        try:
            assert isinstance(self.photoTab, CropPhotoWidget)
        except AssertionError:
            return
        self.handle_path(file_path, FunctionType.PHOTO, self.photoTab.photoLineEdit)

    def handle_video_file(self, file_path: Path) -> None:
        """
        Handles a video file by setting the function tab widget to the video tab, validating the file path, and configuring the video player.

        Args:
            self: The instance of the class.
            file_path (Path): The path to the video file.

        Returns:
            None
        """

        self.handle_function_tab_state(self.videoTab, self.folder_Tab, self.photoTab, self.mappingTab)
        self.function_tabWidget.setCurrentIndex(FunctionType.VIDEO.value)
        try:
            assert isinstance(self.videoTab, CropVideoWidget)
        except AssertionError:
            return
        self.videoTab.videoLineEdit.setText(file_path.as_posix())
        self.videoTab.playButton.setEnabled(True)
        self.videoTab.playButton.setIcon(QtGui.QIcon('resources\\icons\\multimedia_play.svg'))
        self.videoTab.player.setSource(QtCore.QUrl.fromLocalFile(self.videoTab.videoLineEdit.text()))

    def handle_pandas_file(self, file_path: Path) -> None:
        """
        Handles a pandas file by setting the function tab widget to the mapping tab, validating the file path, and opening the table.

        Args:
            self: The instance of the class.
            file_path (Path): The path to the pandas file.

        Returns:
            None
        """

        self.function_tabWidget.setCurrentIndex(FunctionType.MAPPING.value)
        try:
            assert isinstance(self.mappingTab, CropMapWidget)
        except AssertionError:
            return
        self.mappingTab.tableLineEdit.setText(file_path.as_posix())
        data = ut.open_table(file_path)
        self.mappingTab.validate_pandas_file(data)

    def check_tab_selection(self) -> None:
        """
        Checks the current selection of the function tab widget and handles the tab states accordingly.

        Args:
            self: The instance of the class.

        Returns:
            None
        """

        match self.function_tabWidget.currentIndex():
            case FunctionType.PHOTO.value:
                self.handle_function_tab_state(self.photoTab, self.folder_Tab, self.mappingTab, self.videoTab)
            case FunctionType.FOLDER.value:
                self.handle_function_tab_state(self.folder_Tab, self.mappingTab, self.videoTab, self.photoTab)
            case FunctionType.MAPPING.value:
                self.handle_function_tab_state(self.mappingTab, self.videoTab, self.photoTab, self.folder_Tab)
            case FunctionType.VIDEO.value:
                self.handle_function_tab_state(self.videoTab, self.photoTab, self.folder_Tab, self.mappingTab)
            case _:
                pass

    @staticmethod
    def handle_function_tab_state(selected_tab: CustomCropWidget, *other_tabs: CustomCropWidget):
        """
        Sets the selection state of the selected tab to SELECTED and the selection state of other tabs to NOT_SELECTED.

        Args:
            selected_tab (CustomCropWidget): The selected tab.
            *other_tabs (CustomCropWidget): The other tabs.

        Returns:
            None
        """

        selected_tab.selection_state = selected_tab.SELECTED
        for tab in other_tabs:
            tab.selection_state = tab.NOT_SELECTED

    def load_preset(self, phi: Preset) -> None:
        """
        Loads a preset value into the width and height line edits.

        Args:
            self: The instance of the class.
            phi (Preset): The preset value to load.

        Returns:
            None
        """

        if any(line.state is LineEditState.INVALID_INPUT for line in (self.widthLineEdit, self.heightLineEdit)):
            self.widthLineEdit.setText('1000')
            self.heightLineEdit.setText('1000')

        match phi:
            case Preset.SQUARE:
                if self.widthLineEdit.value() > self.heightLineEdit.value():
                    self.heightLineEdit.setText(self.widthLineEdit.text())
                elif self.widthLineEdit.value() < self.heightLineEdit.value():
                    self.widthLineEdit.setText(self.heightLineEdit.text())
            case Preset.GOLDEN_RATIO | Preset.TWO_THIRDS | Preset.THREE_QUARTERS | Preset.FOUR_FIFTHS:
                if self.widthLineEdit.value() >= self.heightLineEdit.value():
                    self.heightLineEdit.setText(str(int(self.widthLineEdit.value() * phi.value)))
                else:
                    self.widthLineEdit.setText(str(int(self.heightLineEdit.value() / phi.value)))

    @staticmethod
    def all_filled(*line_edits: Union[PathLineEdit, NumberLineEdit, QtWidgets.QComboBox]) -> bool:
        x = all(edit.state == LineEditState.VALID_INPUT
                for edit in line_edits if isinstance(edit, (PathLineEdit, NumberLineEdit)))
        y = all(edit.currentText() for edit in line_edits if isinstance(edit, QtWidgets.QComboBox))
        return x and y

    def disable_buttons(self) -> None:
        """
        Disables buttons based on the filled state of line edits and combo boxes.

        Args:
            self: The instance of the class.

        Returns:
            None
        """

        common_line_edits = (self.widthLineEdit, self.heightLineEdit)
        try:
            assert isinstance(self.mappingTab, CropMapWidget)
            assert isinstance(self.folder_Tab, CropFolderWidget)
            assert isinstance(self.photoTab, CropPhotoWidget)
            assert isinstance(self.videoTab, CropVideoWidget)
        except AssertionError:
            return
        # Photo logic
        wf.update_widget_state(
            self.all_filled(self.photoTab.photoLineEdit, self.photoTab.destinationLineEdit, *common_line_edits),
            self.photoTab.cropButton)
        # Folder logic
        wf.update_widget_state(
            self.all_filled(self.folder_Tab.folderLineEdit, self.folder_Tab.destinationLineEdit, *common_line_edits),
            self.folder_Tab.cropButton)
        # Mapping logic
        wf.update_widget_state(
            self.all_filled(self.mappingTab.folderLineEdit, self.mappingTab.tableLineEdit,
                            self.mappingTab.destinationLineEdit, self.mappingTab.comboBox_1,
                            self.mappingTab.comboBox_2, *common_line_edits),
            self.mappingTab.cropButton)
        # Video logic
        wf.update_widget_state(
            self.all_filled(self.videoTab.videoLineEdit, self.videoTab.destinationLineEdit, *common_line_edits),
            self.videoTab.cropButton, self.videoTab.videocropButton)

    def setup_custom_crop_widget(self, tab_widget: QtWidgets.QTabWidget,
                                 function_type: FunctionType) -> Tuple[CustomCropWidget, QtGui.QIcon]:
        """
        Sets up a custom crop widget based on the specified function type.

        Args:
            self: The instance of the class.
            tab_widget (QtWidgets.QTabWidget): The tab widget to add the custom crop widget to.
            function_type (FunctionType): The type of function for the custom crop widget.

        Returns:
            Tuple[CustomCropWidget, QtGui.QIcon]: A tuple containing the custom crop widget and the icon.

        Example:
            ```python
            main_window = MainWindow()
            tab_widget = QtWidgets.QTabWidget()
            function_type = FunctionType.PHOTO
            custom_crop_widget, icon = main_window.setup_custom_crop_widget(tab_widget, function_type)
            ```
        """

        widget_list = (self.crop_worker, self.widthLineEdit, self.heightLineEdit, self.extWidget,
                       self.sensitivity_dialArea, self.face_dialArea, self.gamma_dialArea,
                       self.top_dialArea, self.bottom_dialArea, self.left_dialArea, self.right_dialArea)
        icon = QtGui.QIcon()

        match function_type:
            case FunctionType.PHOTO | FunctionType.FRAME:
                tab = CropPhotoWidget(*widget_list, parent=self)
                wf.create_tab(tab_widget, tab, icon, tab_name='photoTab', icon_name='picture')
                return tab, icon
            case FunctionType.FOLDER:
                tab = CropFolderWidget(*widget_list, parent=self)
                wf.create_tab(tab_widget, tab, icon, tab_name='folder_Tab', icon_name='folder')
                return tab, icon
            case FunctionType.MAPPING:
                tab = CropMapWidget(*widget_list, parent=self)
                wf.create_tab(tab_widget, tab, icon, tab_name='mappingTab', icon_name='excel')
                return tab, icon
            case FunctionType.VIDEO:
                tab = CropVideoWidget(*widget_list, parent=self)
                wf.create_tab(tab_widget, tab, icon, tab_name='videoTab', icon_name='clapperboard')
                return tab, icon

    def connect_combo_boxes(self, tab_widget: CustomCropWidget) -> None:
        """
        Connects the combo boxes in the tab widget to the disable_buttons method.

        Args:
            self: The instance of the class.
            tab_widget (CustomCropWidget): The tab widget containing the combo boxes.

        Returns:
            None
        """

        try:
            assert isinstance(tab_widget, CropMapWidget)
        except AssertionError:
            return
        tab_widget.comboBox_1.currentTextChanged.connect(lambda: self.disable_buttons())
        tab_widget.comboBox_2.currentTextChanged.connect(lambda: self.disable_buttons())
