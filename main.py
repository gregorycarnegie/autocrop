from PyQt6.QtWidgets import QApplication
from qt_material import apply_stylesheet

from ui import UiMainWindow

def main():
    app = QApplication([])
    apply_stylesheet(app, theme='dark_teal.xml')
    window = UiMainWindow()
    window.adjust_ui(app)
    window.show()
    app.aboutToQuit.connect(window.cleanup_workers)
    app.exec()


if __name__ == '__main__':
    main()
