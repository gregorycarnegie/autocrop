from core import UiMainWindow
from PyQt6.QtWidgets import QApplication


def main():
    app = QApplication([])
    app.setStyle('Fusion')
    main_window = UiMainWindow()
    main_window.show()
    app.exec()

if __name__ == "__main__":
    main()
