from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QGuiApplication
from PyQt6.QtWidgets import QApplication, QSplashScreen

from main_widgets import UiMainWindow


def show_splash_screen() -> QSplashScreen:
    screen = QGuiApplication.primaryScreen()
    screen_height = screen.size().height() if screen is not None else 240
    splash_pix = QPixmap('resources\\logos\\logo.svg')  # Replace with the path to your image
    splash_pix = splash_pix.scaledToHeight(screen_height // 3, Qt.TransformationMode.SmoothTransformation)
    splash = QSplashScreen(splash_pix, Qt.WindowType.WindowStaysOnTopHint)
    splash.show()
    return splash

def main() -> None:
    app = QApplication([])
    app.setStyle('Fusion')
    splash = show_splash_screen()
    QTimer.singleShot(3000, splash.close)  # Show the splash screen for 3 seconds
    window = UiMainWindow()
    QTimer.singleShot(3000, window.show)  # Show the main window after 3 secondsS
    app.exec()

if __name__ == '__main__':
    main()
