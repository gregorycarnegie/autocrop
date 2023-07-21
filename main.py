from PyQt6.QtWidgets import QApplication

from main_objects import UiMainWindow


def main():
    app = QApplication([])
    app.setStyle('Fusion')
    window = UiMainWindow()
    window.show()
    app.exec()

if __name__ == "__main__":
    main()
