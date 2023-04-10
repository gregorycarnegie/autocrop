from mainwindow import UiMainWindow
from PyQt6 import QtWidgets
from PIL import Image
import numpy as np

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    app.setStyle('Fusion')
    main_window = UiMainWindow()
    main_window.show()
    app.exec()
    