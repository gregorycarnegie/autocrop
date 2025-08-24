from PyQt6.QtCore import QByteArray, QRect, QSize, Qt
from PyQt6.QtGui import QAction, QIcon, QPainter, QPixmap
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import QMainWindow, QMenu, QMenuBar

from ui.enums import GuiIcon


class MenuManager:
    """Manages the main menu bar and its actions"""

    def __init__(self, main_window: QMainWindow):
        self.main_window = main_window
        self.actions = {}
        self.menus = {}

    def create_main_menu(self):
        """Create the main menu for the application"""
        self._create_actions()
        self._create_menu_bar()
        self._create_menus()
        self._add_actions_to_menus()

    def _create_actions(self):
        """Create all menu actions"""
        # Info actions
        self.actions['about'] = QAction(self.main_window)
        self.actions['about'].setObjectName("actionAbout_Face_Cropper")
        icon0 = QIcon.fromTheme("help-browser")
        self.actions['about'].setIcon(icon0)

        # Tools actions
        self.actions['use_mapping'] = QAction(self.main_window)
        self.actions['use_mapping'].setObjectName("actionUse_Mapping")
        icon1 = QIcon()
        icon1.addFile(GuiIcon.EXCEL, QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.actions['use_mapping'].setIcon(icon1)

        self.actions['crop_file'] = QAction(self.main_window)
        self.actions['crop_file'].setObjectName("actionCrop_File")
        icon2 = QIcon()
        icon2.addFile(GuiIcon.PICTURE, QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.actions['crop_file'].setIcon(icon2)

        self.actions['crop_folder'] = QAction(self.main_window)
        self.actions['crop_folder'].setObjectName("actionCrop_Folder")
        icon3 = QIcon()
        icon3.addFile(GuiIcon.FOLDER, QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.actions['crop_folder'].setIcon(icon3)

        self.actions['crop_video'] = QAction(self.main_window)
        self.actions['crop_video'].setObjectName("actionCrop_Video")
        icon4 = QIcon()
        icon4.addFile(GuiIcon.CLAPPERBOARD, QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.actions['crop_video'].setIcon(icon4)

        # Preset actions
        self.actions['square'] = QAction(self.main_window)
        self.actions['square'].setObjectName("actionSquare")
        self.actions['golden_ratio'] = QAction(self.main_window)
        self.actions['golden_ratio'].setObjectName("actionGolden_Ratio")
        self.actions['2_3_ratio'] = QAction(self.main_window)
        self.actions['2_3_ratio'].setObjectName("action2_3_Ratio")
        self.actions['3_4_ratio'] = QAction(self.main_window)
        self.actions['3_4_ratio'].setObjectName("action3_4_Ratio")
        self.actions['4_5_ratio'] = QAction(self.main_window)
        self.actions['4_5_ratio'].setObjectName("action4_5_Ratio")

        # Decorate preset actions with icons
        self._decorate_action(self.actions['square'], 1, 1, "#3498db")
        self._decorate_action(self.actions['golden_ratio'], 100, 162, "#f39c12")
        self._decorate_action(self.actions['2_3_ratio'], 2, 3, "#2ecc71")
        self._decorate_action(self.actions['3_4_ratio'], 3, 4, "#e74c3c")
        self._decorate_action(self.actions['4_5_ratio'], 4, 5, "#9b59b6")

    def _create_menu_bar(self):
        """Create the menu bar"""
        self.menubar = QMenuBar(self.main_window)
        self.menubar.setObjectName("menubar")
        self.menubar.setGeometry(QRect(0, 0, 1256, 22))
        self.main_window.setMenuBar(self.menubar)

    def _create_menus(self):
        """Create menu items"""
        self.menus['file'] = QMenu(self.menubar)
        self.menus['file'].setObjectName("menuFile")

        self.menus['tools'] = QMenu(self.menubar)
        self.menus['tools'].setObjectName("menuTools")

        self.menus['info'] = QMenu(self.menubar)
        self.menus['info'].setObjectName("menuInfo")

        # Add menus to menu bar
        self.menubar.addAction(self.menus['file'].menuAction())
        self.menubar.addAction(self.menus['tools'].menuAction())
        self.menubar.addAction(self.menus['info'].menuAction())

    def _add_actions_to_menus(self):
        """Add actions to their respective menus"""
        # Add preset actions to File menu
        self.menus['file'].addAction(self.actions['square'])
        self.menus['file'].addAction(self.actions['golden_ratio'])
        self.menus['file'].addAction(self.actions['2_3_ratio'])
        self.menus['file'].addAction(self.actions['3_4_ratio'])
        self.menus['file'].addAction(self.actions['4_5_ratio'])

        # Add tool actions to Tools menu
        self.menus['tools'].addAction(self.actions['crop_file'])
        self.menus['tools'].addAction(self.actions['crop_folder'])
        self.menus['tools'].addAction(self.actions['use_mapping'])
        self.menus['tools'].addAction(self.actions['crop_video'])

        # Add info actions to Info menu
        self.menus['info'].addAction(self.actions['about'])

    def _decorate_action(self, action: QAction, width: float, height: float, color: str):
        """Set the aspect ratio icons for menu actions"""
        max_dim = 64
        # Compute a uniform scale factor that fits the longer side to `max_dim`.
        scale = max_dim / max(width, height)
        w, h = int(width * scale), int(height * scale)

        # Centre the rectangle on the square canvas.
        x_offset, y_offset = (max_dim - w) // 2, (max_dim - h) // 2

        # Create SVG data with the rectangle
        svg_data = f"""
        <svg width="{max_dim}" height="{max_dim}" xmlns="http://www.w3.org/2000/svg">
            <rect x="{x_offset}" y="{y_offset}" width="{w}" height="{h}" fill="{color}"
                stroke="#333333" stroke-width="2" rx="2" ry="2"/>
        </svg>
        """

        svg_bytes = QByteArray(svg_data.encode())
        renderer = QSvgRenderer(svg_bytes)

        # Create pixmap at multiple sizes for better scaling
        icon = QIcon()
        for size in [16, 24, 32, 48, 64]:
            pixmap = QPixmap(size, size)
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)
            renderer.render(painter)
            painter.end()
            icon.addPixmap(pixmap)

        action.setIcon(icon)

    def get_action(self, action_name: str) -> QAction:
        """Get action by name"""
        return self.actions.get(action_name)

    def get_menu(self, menu_name: str) -> QMenu:
        """Get menu by name"""
        return self.menus.get(menu_name)

    def retranslate_ui(self):
        """Retranslate menu items"""
        from PyQt6.QtCore import QCoreApplication

        # Set action texts
        self.actions['about'].setText(QCoreApplication.translate("self", "About Autocrop", None))
        self.actions['use_mapping'].setText(QCoreApplication.translate("self", "Use Mapping", None))
        self.actions['crop_file'].setText(QCoreApplication.translate("self", "Crop File", None))
        self.actions['crop_folder'].setText(QCoreApplication.translate("self", "Crop Folder", None))
        self.actions['square'].setText(QCoreApplication.translate("self", "Square", None))
        self.actions['golden_ratio'].setText(QCoreApplication.translate("self", "Golden Ratio", None))
        self.actions['2_3_ratio'].setText(QCoreApplication.translate("self", "2:3 Ratio", None))
        self.actions['3_4_ratio'].setText(QCoreApplication.translate("self", "3:4 Ratio", None))
        self.actions['4_5_ratio'].setText(QCoreApplication.translate("self", "4:5 Ratio", None))
        self.actions['crop_video'].setText(QCoreApplication.translate("self", "Crop Video", None))

        # Set menu titles
        self.menus['file'].setTitle(QCoreApplication.translate("self", "Presets", None))
        self.menus['tools'].setTitle(QCoreApplication.translate("self", "Tools", None))
        self.menus['info'].setTitle(QCoreApplication.translate("self", "Info", None))
