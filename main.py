import sys
from PyQt6.QtWidgets import QApplication
from ux import MainWindow


def run():
    app = QApplication([])
    app.setStyle('Fusion')
    widget = MainWindow()
    widget.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    run()
