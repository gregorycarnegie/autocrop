from PyQt6.QtWidgets import QApplication

from main_widgets import UiMainWindow, ClickableSplashScreen


def main():
    app = QApplication([])
    app.setStyle('Fusion')
    splash = ClickableSplashScreen()
    splash.show_message()
    window = UiMainWindow()
    window.show()
    splash.finish(window)
    app.exec()

if __name__ == '__main__':
    main()
