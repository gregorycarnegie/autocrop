import platform

from PyQt6.QtCore import QFile, QTextStream
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication

from core.ffmpeg_utils import configure_ffmpeg
from core.resource_path import ResourcePath
from ui import UiMainWindow
from ui.enums import GuiIcon


def load_stylesheet(app: QApplication) -> None:
    """Load the browser-style CSS stylesheet"""
    # Path to our custom stylesheet
    style_path = ResourcePath("resources\\browser_style.css").as_resource()

    # Load the stylesheet
    file = QFile(style_path)
    if file.open(QFile.OpenModeFlag.ReadOnly | QFile.OpenModeFlag.Text):
        stream = QTextStream(file)
        app.setStyleSheet(stream.readAll())
        file.close()

def setup_fonts(app: QApplication) -> None:
    """Load and set custom fonts for the application"""
    # We could add custom font loading here if needed
    font = app.font()
    font.setFamily("Segoe UI")  # Windows default (fallbacks to system default on other OS)
    font.setPointSize(10)
    app.setFont(font)

def main():
    app = QApplication([])

    # Set application details
    app.setApplicationName("Autocrop")
    app.setApplicationDisplayName("Autocrop")
    app.setWindowIcon(QIcon(GuiIcon.ICON))

    # Configure FFmpeg to use bundled version
    if platform.system() == "Windows":
        configure_ffmpeg()

    # Setup custom styling
    setup_fonts(app)
    load_stylesheet(app)

    # Create the main window
    window = UiMainWindow()
    window.adjust_ui(app)
    window.show()

    app.aboutToQuit.connect(window.cleanup_workers)
    app.exec()

if __name__ == '__main__':
    main()
