from PyQt6.QtWidgets import QApplication
from qt_material import apply_stylesheet

from main_widgets import UiMainWindow, UiClickableSplashScreen

def main():
    app = QApplication([])
    apply_stylesheet(app, theme='dark_teal.xml')
    splash = UiClickableSplashScreen()
    splash.show_message()
    window = UiMainWindow()
    window.adjust_ui(app)
    window.show()
    splash.finish(window)
    app.exec()


if __name__ == '__main__':
    main()
