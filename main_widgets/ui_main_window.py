from pathlib import Path
from typing import Union, Optional

from PyQt6 import QtCore, QtWidgets, QtGui

from core import face_tools as ft
from core import utils as ut
from core import window_functions as wf
from core.croppers import FolderCropper, PhotoCropper, MappingCropper, VideoCropper
from core.enums import FunctionType, Preset, GuiIcon
from file_types import Photo, Table, Video
from line_edits import NumberLineEdit, PathLineEdit, LineEditState
from .ui_control_widget import UiCropControlWidget
from .ui_crop_folder_widget import UiFolderTabWidget
from .ui_crop_map_widget import UiMappingTabWidget
from .ui_crop_photo_widget import UiPhotoTabWidget
from .ui_crop_vid_widget import UiVideoTabWidget
from .ui_crop_widget import UiCropWidget

type TabWidget = Union[UiPhotoTabWidget, UiFolderTabWidget, UiMappingTabWidget, UiVideoTabWidget]


class UiMainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        face_tool_list = ft.generate_face_detection_tools()
        self.folder_worker = FolderCropper(face_tool_list)
        self.photo_worker = PhotoCropper(face_tool_list)
        self.mapping_worker = MappingCropper(face_tool_list)
        self.video_worker = VideoCropper(face_tool_list)
        self.setObjectName(u"MainWindow")
        self.resize(1256, 652)
        icon = QtGui.QIcon()
        icon.addFile(GuiIcon.LOGO.value, QtCore.QSize(), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.setWindowIcon(icon)
        self.actionAbout_Face_Cropper = QtGui.QAction(self)
        self.actionAbout_Face_Cropper.setObjectName(u"actionAbout_Face_Cropper")
        self.actionUse_Mapping = QtGui.QAction(self)
        self.actionUse_Mapping.setObjectName(u"actionUse_Mapping")
        icon1 = QtGui.QIcon()
        icon1.addFile(GuiIcon.EXCEL.value, QtCore.QSize(), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.actionUse_Mapping.setIcon(icon1)
        self.actionCrop_File = QtGui.QAction(self)
        self.actionCrop_File.setObjectName(u"actionCrop_File")
        icon2 = QtGui.QIcon()
        icon2.addFile(GuiIcon.PICTURE.value, QtCore.QSize(), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.actionCrop_File.setIcon(icon2)
        self.actionCrop_Folder = QtGui.QAction(self)
        self.actionCrop_Folder.setObjectName(u"actionCrop_Folder")
        icon3 = QtGui.QIcon()
        icon3.addFile(GuiIcon.FOLDER.value, QtCore.QSize(), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.actionCrop_Folder.setIcon(icon3)
        self.actionSquare = QtGui.QAction(self)
        self.actionSquare.setObjectName(u"actionSquare")
        self.actionGolden_Ratio = QtGui.QAction(self)
        self.actionGolden_Ratio.setObjectName(u"actionGolden_Ratio")
        self.action2_3_Ratio = QtGui.QAction(self)
        self.action2_3_Ratio.setObjectName(u"action2_3_Ratio")
        self.action3_4_Ratio = QtGui.QAction(self)
        self.action3_4_Ratio.setObjectName(u"action3_4_Ratio")
        self.action4_5_Ratio = QtGui.QAction(self)
        self.action4_5_Ratio.setObjectName(u"action4_5_Ratio")
        self.actionCrop_Video = QtGui.QAction(self)
        self.actionCrop_Video.setObjectName(u"actionCrop_Video")
        icon4 = QtGui.QIcon()
        icon4.addFile(GuiIcon.CLAPPERBOARD.value, QtCore.QSize(), QtGui.QIcon.Mode.Normal,
                      QtGui.QIcon.State.Off)
        self.actionCrop_Video.setIcon(icon4)
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName(u"centralwidget")
        self.horizontalLayout_2 = wf.setup_hbox(u"horizontalLayout_2", self.centralwidget)
        self.verticalLayout_1 = QtWidgets.QVBoxLayout()
        self.verticalLayout_1.setObjectName(u"verticalLayout_1")
        self.function_tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.function_tabWidget.setObjectName(u"function_tabWidget")
        self.function_tabWidget.setMovable(True)
        self.photo_tab = QtWidgets.QWidget()
        self.photo_tab.setObjectName(u"photo_tab")
        self.verticalLayout_2 = wf.setup_vbox(u"verticalLayout_2", self.photo_tab)
        self.photo_tab_widget = UiPhotoTabWidget(self.photo_worker, u"photo_tab_widget", self.photo_tab, face_tool_list)

        self.verticalLayout_2.addWidget(self.photo_tab_widget)

        self.function_tabWidget.addTab(self.photo_tab, icon2, "")
        self.folder_tab = QtWidgets.QWidget()
        self.folder_tab.setObjectName(u"folder_tab")
        self.verticalLayout_3 = wf.setup_vbox(u"verticalLayout_3", self.folder_tab)
        self.folder_tab_widget = UiFolderTabWidget(self.folder_worker, u"folder_tab_widget", self.folder_tab, face_tool_list)

        self.verticalLayout_3.addWidget(self.folder_tab_widget)

        self.function_tabWidget.addTab(self.folder_tab, icon3, "")
        self.mapping_tab = QtWidgets.QWidget()
        self.mapping_tab.setObjectName(u"mapping_tab")
        self.verticalLayout_4 = wf.setup_vbox(u"verticalLayout_4", self.mapping_tab)
        self.mapping_tab_widget = UiMappingTabWidget(self.mapping_worker, u"mapping_tab_widget", self.mapping_tab, face_tool_list)

        self.verticalLayout_4.addWidget(self.mapping_tab_widget)

        self.function_tabWidget.addTab(self.mapping_tab, icon1, "")
        self.video_tab = QtWidgets.QWidget()
        self.video_tab.setObjectName(u"video_tab")
        self.verticalLayout_5 = wf.setup_vbox(u"verticalLayout_5", self.video_tab)
        self.video_tab_widget = UiVideoTabWidget(self.video_worker, u"video_tab_widget", self.video_tab, face_tool_list)

        self.verticalLayout_5.addWidget(self.video_tab_widget)

        self.function_tabWidget.addTab(self.video_tab, icon4, "")

        self.verticalLayout_1.addWidget(self.function_tabWidget)

        self.verticalLayout_1.setStretch(0, 13)

        self.horizontalLayout_2.addLayout(self.verticalLayout_1)

        self.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(self)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1256, 22))
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName(u"menuFile")
        self.menuTools = QtWidgets.QMenu(self.menubar)
        self.menuTools.setObjectName(u"menuTools")
        self.menuInfo = QtWidgets.QMenu(self.menubar)
        self.menuInfo.setObjectName(u"menuInfo")
        self.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName(u"statusbar")
        self.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuTools.menuAction())
        self.menubar.addAction(self.menuInfo.menuAction())
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

        # CONNECTIONS
        self.connect_combo_boxes(self.mapping_tab_widget)

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
        self.function_tabWidget.currentChanged.connect(lambda: self.video_tab_widget.player.pause())

        self.folder_worker.error.connect(lambda: self.folder_tab_widget.disable_buttons())
        self.photo_worker.error.connect(lambda: self.photo_tab_widget.disable_buttons())
        self.mapping_worker.error.connect(lambda: self.mapping_tab_widget.disable_buttons())
        self.video_worker.error.connect(lambda: self.video_tab_widget.disable_buttons())

        self.retranslateUi()
        self.actionCrop_File.triggered.connect(lambda: self.function_tabWidget.setFocus())
        self.actionCrop_Folder.triggered.connect(lambda: self.function_tabWidget.setFocus())
        self.actionCrop_Video.triggered.connect(lambda: self.function_tabWidget.setFocus())
        self.actionUse_Mapping.triggered.connect(lambda: self.function_tabWidget.setFocus())

        self.function_tabWidget.setCurrentIndex(0)

        QtCore.QMetaObject.connectSlotsByName(self)

    # setupUi

    def retranslateUi(self):
        self.setWindowTitle(QtCore.QCoreApplication.translate("self", u"Face Cropper", None))
        self.actionAbout_Face_Cropper.setText(QtCore.QCoreApplication.translate("self", u"About Face Cropper", None))
        self.actionUse_Mapping.setText(QtCore.QCoreApplication.translate("self", u"Use Mapping", None))
        self.actionCrop_File.setText(QtCore.QCoreApplication.translate("self", u"Crop File", None))
        self.actionCrop_Folder.setText(QtCore.QCoreApplication.translate("self", u"Crop Folder", None))
        self.actionSquare.setText(QtCore.QCoreApplication.translate("self", u"Square", None))
        self.actionGolden_Ratio.setText(QtCore.QCoreApplication.translate("self", u"Golden Ratio", None))
        self.action2_3_Ratio.setText(QtCore.QCoreApplication.translate("self", u"2:3 Ratio", None))
        self.action3_4_Ratio.setText(QtCore.QCoreApplication.translate("self", u"3:4 Ratio", None))
        self.action4_5_Ratio.setText(QtCore.QCoreApplication.translate("self", u"4:5 Ratio", None))
        self.actionCrop_Video.setText(QtCore.QCoreApplication.translate("self", u"Crop Video", None))
        self.function_tabWidget.setTabText(self.function_tabWidget.indexOf(self.photo_tab),
                                           QtCore.QCoreApplication.translate("self", u"Photo Crop", None))
        self.function_tabWidget.setTabText(self.function_tabWidget.indexOf(self.folder_tab),
                                           QtCore.QCoreApplication.translate("self", u"Folder Crop", None))
        self.function_tabWidget.setTabText(self.function_tabWidget.indexOf(self.mapping_tab),
                                           QtCore.QCoreApplication.translate("self", u"Mapping Crop", None))
        self.function_tabWidget.setTabText(self.function_tabWidget.indexOf(self.video_tab),
                                           QtCore.QCoreApplication.translate("self", u"Video Crop", None))
        self.menuFile.setTitle(QtCore.QCoreApplication.translate("self", u"Presets", None))
        self.menuTools.setTitle(QtCore.QCoreApplication.translate("self", u"Tools", None))
        self.menuInfo.setTitle(QtCore.QCoreApplication.translate("self", u"Info", None))

    # retranslateUi

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
            assert isinstance(self.mapping_tab_widget, UiMappingTabWidget)
            assert isinstance(self.folder_tab_widget, UiFolderTabWidget)
            assert isinstance(self.photo_tab_widget, UiPhotoTabWidget)
        except AssertionError:
            return

        if self.photo_tab_widget.selection_state == self.photo_tab_widget.SELECTED:
            self.handle_function_tab_state(self.photo_tab_widget, self.folder_tab_widget, self.mapping_tab_widget,
                                           self.video_tab_widget)
            self.photo_tab_widget.display_crop()
        elif self.folder_tab_widget.selection_state == self.folder_tab_widget.SELECTED:
            self.handle_function_tab_state(self.folder_tab_widget, self.photo_tab_widget, self.mapping_tab_widget,
                                           self.video_tab_widget)
            self.folder_tab_widget.load_data()
        elif self.mapping_tab_widget.selection_state == self.mapping_tab_widget.SELECTED:
            self.handle_function_tab_state(self.mapping_tab_widget, self.photo_tab_widget, self.folder_tab_widget,
                                           self.video_tab_widget)
            self.mapping_tab_widget.display_crop()

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
            assert isinstance(self.mapping_tab_widget, UiMappingTabWidget)
            assert isinstance(self.folder_tab_widget, UiFolderTabWidget)
        except AssertionError:
            return
        if any(mask):
            self.handle_path(file_path, FunctionType.MAPPING, self.mapping_tab_widget.inputLineEdit)
        else:
            self.handle_path(file_path, FunctionType.FOLDER, self.folder_tab_widget.inputLineEdit)

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
            assert isinstance(self.photo_tab_widget, UiPhotoTabWidget)
        except AssertionError:
            return
        self.handle_path(file_path, FunctionType.PHOTO, self.photo_tab_widget.inputLineEdit)

    def handle_video_file(self, file_path: Path) -> None:
        """
        Handles a video file by setting the function tab widget to the video tab, validating the file path, and configuring the video player.

        Args:
            self: The instance of the class.
            file_path (Path): The path to the video file.

        Returns:
            None
        """

        self.handle_function_tab_state(self.video_tab_widget, self.folder_tab_widget, self.photo_tab_widget,
                                       self.mapping_tab_widget)
        self.function_tabWidget.setCurrentIndex(FunctionType.VIDEO.value)
        try:
            assert isinstance(self.video_tab_widget, UiVideoTabWidget)
        except AssertionError:
            return
        self.video_tab_widget.inputLineEdit.setText(file_path.as_posix())
        self.video_tab_widget.mediacontrolWidget_1.playButton.setEnabled(True)
        self.video_tab_widget.mediacontrolWidget_1.playButton.setIcon(QtGui.QIcon(GuiIcon.MULTIMEDIA_PLAY))
        self.video_tab_widget.mediacontrolWidget_2.playButton.setEnabled(True)
        self.video_tab_widget.mediacontrolWidget_2.playButton.setIcon(QtGui.QIcon(GuiIcon.MULTIMEDIA_PLAY))
        self.video_tab_widget.player.setSource(QtCore.QUrl.fromLocalFile(self.video_tab_widget.inputLineEdit.text()))

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
            assert isinstance(self.mapping_tab_widget, UiMappingTabWidget)
        except AssertionError:
            return
        self.mapping_tab_widget.tableLineEdit.setText(file_path.as_posix())
        data = ut.open_table(file_path)
        self.mapping_tab_widget.validate_pandas_file(data)

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
                self.handle_function_tab_state(self.photo_tab_widget, self.folder_tab_widget, self.mapping_tab_widget,
                                               self.video_tab_widget)
            case FunctionType.FOLDER.value:
                self.handle_function_tab_state(self.folder_tab_widget, self.mapping_tab_widget, self.video_tab_widget,
                                               self.photo_tab_widget)
            case FunctionType.MAPPING.value:
                self.handle_function_tab_state(self.mapping_tab_widget, self.video_tab_widget, self.photo_tab_widget,
                                               self.folder_tab_widget)
            case FunctionType.VIDEO.value:
                self.handle_function_tab_state(self.video_tab_widget, self.photo_tab_widget, self.folder_tab_widget,
                                               self.mapping_tab_widget)
            case _:
                pass

    @staticmethod
    def handle_function_tab_state(selected_tab: UiCropWidget, *other_tabs: UiCropWidget):
        """
        Sets the selection state of the selected tab to SELECTED and the selection state of other tabs to NOT_SELECTED.

        Args:
            selected_tab (UiCropWidget): The selected tab.
            *other_tabs (UiCropWidget): The other tabs.

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

        def callback(control: UiCropControlWidget) -> None:
            if any(line.state is LineEditState.INVALID_INPUT for line in
                   (control.widthLineEdit, control.heightLineEdit)):
                control.widthLineEdit.setText(u'1000')
                control.heightLineEdit.setText(u'1000')

            match phi:
                case Preset.SQUARE:
                    if control.widthLineEdit.value() > control.heightLineEdit.value():
                        control.heightLineEdit.setText(control.widthLineEdit.text())
                    elif control.widthLineEdit.value() < control.heightLineEdit.value():
                        control.widthLineEdit.setText(control.heightLineEdit.text())
                case Preset.GOLDEN_RATIO | Preset.TWO_THIRDS | Preset.THREE_QUARTERS | Preset.FOUR_FIFTHS:
                    if control.widthLineEdit.value() >= control.heightLineEdit.value():
                        control.heightLineEdit.setText(str(int(control.widthLineEdit.value() * phi.value)))
                    else:
                        control.widthLineEdit.setText(str(int(control.heightLineEdit.value() / phi.value)))

        match self.function_tabWidget.currentIndex():
            case 0:
                callback(self.photo_tab_widget.controlWidget)
            case 1:
                callback(self.folder_tab_widget.controlWidget)
            case 2:
                callback(self.mapping_tab_widget.controlWidget)
            case 3:
                callback(self.video_tab_widget.controlWidget)
            case _:
                pass

    @staticmethod
    def all_filled(*line_edits: Union[PathLineEdit, NumberLineEdit, QtWidgets.QComboBox]) -> bool:
        x = all(edit.state == LineEditState.VALID_INPUT
                for edit in line_edits if isinstance(edit, (PathLineEdit, NumberLineEdit)))
        y = all(edit.currentText() for edit in line_edits if isinstance(edit, QtWidgets.QComboBox))
        return x and y

    def disable_buttons(self, tab_widget: TabWidget) -> None:
        """
        Disables buttons based on the filled state of line edits and combo boxes.

        Args:
            self: The instance of the class.
            tab_widget: The tab widget.

        Returns:
            None
        """

        common_line_edits = (tab_widget.controlWidget.widthLineEdit, tab_widget.controlWidget.heightLineEdit)

        match tab_widget:
            case tab_widget if isinstance(tab_widget, (UiPhotoTabWidget, UiFolderTabWidget)):
                wf.change_widget_state(
                    self.all_filled(
                        tab_widget.inputLineEdit,
                        tab_widget.destinationLineEdit,
                        *common_line_edits,
                    ),
                    tab_widget.cropButton,
                )
            case tab_widget if isinstance(tab_widget, UiMappingTabWidget):
                wf.change_widget_state(
                    self.all_filled(
                        tab_widget.inputLineEdit,
                        tab_widget.tableLineEdit,
                        tab_widget.destinationLineEdit,
                        tab_widget.comboBox_1,
                        tab_widget.comboBox_2,
                        *common_line_edits
                    ),
                    self.mapping_tab_widget.cropButton
                )
            case tab_widget if isinstance(tab_widget, UiVideoTabWidget):
                wf.change_widget_state(
                    self.all_filled(
                        tab_widget.inputLineEdit,
                        tab_widget.destinationLineEdit,
                        *common_line_edits
                    ),
                    tab_widget.mediacontrolWidget_1.cropButton,
                    tab_widget.mediacontrolWidget_2.cropButton,
                    tab_widget.mediacontrolWidget_1.videocropButton,
                    tab_widget.mediacontrolWidget_2.videocropButton
                )
            case _:
                pass

    def connect_combo_boxes(self, tab_widget: TabWidget) -> None:
        """
        Connects the combo boxes in the tab widget to the disable_buttons method.

        Args:
            self: The instance of the class.
            tab_widget (UiCropWidget): The tab widget containing the combo boxes.

        Returns:
            None
        """

        try:
            assert isinstance(tab_widget, UiMappingTabWidget)
        except AssertionError:
            return
        tab_widget.comboBox_1.currentTextChanged.connect(lambda: self.disable_buttons(tab_widget))
        tab_widget.comboBox_2.currentTextChanged.connect(lambda: self.disable_buttons(tab_widget))
