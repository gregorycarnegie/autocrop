import sys
from PyQt6.QtWidgets import QApplication
import ux


def run():
    app = QApplication([])
    app.setStyle('Fusion')
    widget = ux.MainWindow()
    widget.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    run()
