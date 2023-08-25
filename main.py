from PyQt6.QtWidgets import QApplication
from qt_material import apply_stylesheet

from main_widgets import UiMainWindow, ClickableSplashScreen


def main():
    app = QApplication([])
    apply_stylesheet(app, theme='dark_teal.xml')
    splash = ClickableSplashScreen()
    splash.show_message()
    window = UiMainWindow()
    window.show()
    splash.finish(window)
    app.exec()

if __name__ == '__main__':
    main()
