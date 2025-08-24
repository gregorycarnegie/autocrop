from PyQt6.QtCore import QCoreApplication, pyqtSignal
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

from core.enums import FunctionType
from line_edits import PathLineEdit, PathType
from ui.enums import GuiIcon


class AddressBarWidget(QWidget):
    """Unified address bar widget with context-aware functionality"""

    # Signals for communication with main window
    context_button_clicked = pyqtSignal()
    secondary_button_clicked = pyqtSignal()
    destination_button_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("addressBarWidget")
        self.setMinimumHeight(48)
        self.setMaximumHeight(48)

        self._setup_widgets()
        self._setup_layout()
        self._connect_signals()

    def _setup_widgets(self):
        """Initialize all address bar widgets"""
        # Navigation buttons
        self.back_button = QPushButton()
        self.back_button.setIcon(QIcon.fromTheme("go-previous"))
        self.back_button.setObjectName("backButton")
        self.back_button.setToolTip("Back")
        self.back_button.setFixedSize(36, 36)

        self.forward_button = QPushButton()
        self.forward_button.setIcon(QIcon.fromTheme("go-next"))
        self.forward_button.setObjectName("forwardButton")
        self.forward_button.setToolTip("Forward")
        self.forward_button.setFixedSize(36, 36)

        self.refresh_button = QPushButton()
        self.refresh_button.setIcon(QIcon.fromTheme("view-refresh"))
        self.refresh_button.setObjectName("refreshButton")
        self.refresh_button.setToolTip("Refresh")
        self.refresh_button.setFixedSize(36, 36)

        # Unified address bar (dynamic path field)
        self.unified_address_bar = PathLineEdit(path_type=PathType.IMAGE)  # Default to image
        self.unified_address_bar.setObjectName("unifiedAddressBar")
        self.unified_address_bar.setPlaceholderText("Enter path...")

        # Context-aware open button with changing icon
        self.context_button = QPushButton()
        self.context_button.setObjectName("contextButton")
        self.context_button.setToolTip("Open File")
        self.context_button.setIcon(QIcon(GuiIcon.PICTURE))  # Default icon
        self.context_button.setFixedSize(36, 36)

        # Secondary input for mapping tab (initially hidden)
        self.secondary_input_container = QWidget()
        self.secondary_input_container.setObjectName("secondaryInputContainer")
        self.secondary_input_container.setVisible(False)  # Hidden by default

        secondary_layout = QHBoxLayout(self.secondary_input_container)
        secondary_layout.setContentsMargins(0, 0, 0, 0)
        secondary_layout.setSpacing(5)

        self.secondary_input = PathLineEdit(path_type=PathType.TABLE)
        self.secondary_input.setObjectName("secondaryInput")
        self.secondary_input.setPlaceholderText("Select table file...")

        self.secondary_button = QPushButton()
        self.secondary_button.setObjectName("secondaryButton")
        self.secondary_button.setIcon(QIcon(GuiIcon.EXCEL))
        self.secondary_button.setFixedSize(36, 36)
        self.secondary_button.setToolTip("Open Table File")

        secondary_layout.addWidget(self.secondary_input)
        secondary_layout.addWidget(self.secondary_button)

        # Destination section (always visible)
        self.destination_container = QWidget()
        self.destination_container.setObjectName("destinationContainer")

        destination_layout = QHBoxLayout(self.destination_container)
        destination_layout.setContentsMargins(0, 0, 0, 0)
        destination_layout.setSpacing(5)

        self.destination_label = QLabel("Save to:")
        self.destination_label.setObjectName("destinationLabel")

        self.destination_input = PathLineEdit(path_type=PathType.FOLDER)
        self.destination_input.setObjectName("destinationInput")
        self.destination_input.setPlaceholderText("Select destination folder...")

        self.destination_button = QPushButton()
        self.destination_button.setObjectName("destinationButton")
        self.destination_button.setIcon(QIcon(GuiIcon.FOLDER))
        self.destination_button.setFixedSize(36, 36)
        self.destination_button.setToolTip("Select Destination Folder")

        destination_layout.addWidget(self.destination_label)
        destination_layout.addWidget(self.destination_input)
        destination_layout.addWidget(self.destination_button)

        # Info button (on the right)
        self.info_button = QPushButton()
        self.info_button.setIcon(QIcon.fromTheme("help-browser"))
        self.info_button.setObjectName("infoButton")
        self.info_button.setToolTip("Info")
        self.info_button.setFixedSize(36, 36)

    def _setup_layout(self):
        """Setup the address bar layout"""
        address_bar_layout = QHBoxLayout(self)
        address_bar_layout.setContentsMargins(10, 5, 10, 5)
        address_bar_layout.setSpacing(10)

        # Add widgets to layout
        address_bar_layout.addWidget(self.back_button)
        address_bar_layout.addWidget(self.forward_button)
        address_bar_layout.addWidget(self.refresh_button)
        address_bar_layout.addWidget(self.unified_address_bar)
        address_bar_layout.addWidget(self.context_button)
        address_bar_layout.addWidget(self.secondary_input_container)
        address_bar_layout.addWidget(self.destination_container)
        address_bar_layout.addWidget(self.info_button)

        # Set stretch factors
        address_bar_layout.setStretch(3, 3)  # Unified address bar gets more space
        address_bar_layout.setStretch(5, 3)  # Secondary input gets more space
        address_bar_layout.setStretch(6, 4)  # Destination gets more space

    def _connect_signals(self):
        """Connect internal signals"""
        self.context_button.clicked.connect(self.context_button_clicked.emit)
        self.secondary_button.clicked.connect(self.secondary_button_clicked.emit)
        self.destination_button.clicked.connect(self.destination_button_clicked.emit)

    def update_context(self, function_type: FunctionType):
        """Update address bar context based on current function type"""
        match function_type:
            case FunctionType.PHOTO:
                # Update primary address bar
                self.unified_address_bar.set_path_type(PathType.IMAGE)
                self.unified_address_bar.setPlaceholderText("Enter image file path...")
                self.context_button.setIcon(QIcon(GuiIcon.PICTURE))
                self.context_button.setToolTip("Open Image")
                # Hide secondary input (not needed for photo tab)
                self.secondary_input_container.setVisible(False)

            case FunctionType.FOLDER:
                self.unified_address_bar.set_path_type(PathType.FOLDER)
                self.unified_address_bar.setPlaceholderText("Enter folder path...")
                self.context_button.setIcon(QIcon(GuiIcon.FOLDER))
                self.context_button.setToolTip("Select Folder")
                # Hide secondary input (not needed for folder tab)
                self.secondary_input_container.setVisible(False)

            case FunctionType.MAPPING:
                self.unified_address_bar.set_path_type(PathType.FOLDER)
                self.unified_address_bar.setPlaceholderText("Enter source folder path...")
                self.context_button.setIcon(QIcon(GuiIcon.FOLDER))
                self.context_button.setToolTip("Select Source Folder")
                # Show and configure secondary input for mapping tab
                self.secondary_input_container.setVisible(True)
                self.secondary_input.set_path_type(PathType.TABLE)
                self.secondary_input.setPlaceholderText("Enter table file path...")

            case FunctionType.VIDEO:
                self.unified_address_bar.set_path_type(PathType.VIDEO)
                self.unified_address_bar.setPlaceholderText("Enter video file path...")
                self.context_button.setIcon(QIcon(GuiIcon.CLAPPERBOARD))
                self.context_button.setToolTip("Open Video")
                # Hide secondary input (not needed for video tab)
                self.secondary_input_container.setVisible(False)

    def update_paths(self, primary_path: str = "", secondary_path: str = "", destination_path: str = ""):
        """Update the paths in the address bar inputs"""
        self.unified_address_bar.blockSignals(True)
        self.unified_address_bar.setText(primary_path)
        self.unified_address_bar.blockSignals(False)

        if self.secondary_input_container.isVisible():
            self.secondary_input.blockSignals(True)
            self.secondary_input.setText(secondary_path)
            self.secondary_input.blockSignals(False)

        self.destination_input.blockSignals(True)
        self.destination_input.setText(destination_path)
        self.destination_input.blockSignals(False)

        # Update clear button visibility
        self.unified_address_bar.update_clear_button(self.unified_address_bar.text())
        self.destination_input.update_clear_button(self.destination_input.text())
        if self.secondary_input_container.isVisible():
            self.secondary_input.update_clear_button(self.secondary_input.text())

    def initialize_clear_button_states(self):
        """Initialize clear button states for all path line edits"""
        self.unified_address_bar.update_clear_button(self.unified_address_bar.text())
        self.destination_input.update_clear_button(self.destination_input.text())
        self.secondary_input.update_clear_button(self.secondary_input.text())

        # Make sure visibility is properly set
        if not self.unified_address_bar.text():
            self.unified_address_bar.clearButton.setVisible(False)
        if not self.destination_input.text():
            self.destination_input.clearButton.setVisible(False)
        if not self.secondary_input.text():
            self.secondary_input.clearButton.setVisible(False)

    def retranslate_ui(self):
        """Retranslate UI elements"""
        self.unified_address_bar.setPlaceholderText(
            QCoreApplication.translate("self", "Enter file path or drag and drop files here", None)
        )
        self.back_button.setToolTip(QCoreApplication.translate("self", "Back", None))
        self.forward_button.setToolTip(QCoreApplication.translate("self", "Forward", None))
        self.refresh_button.setToolTip(QCoreApplication.translate("self", "Refresh Preview", None))
        self.info_button.setToolTip(QCoreApplication.translate("self", "Settings", None))
