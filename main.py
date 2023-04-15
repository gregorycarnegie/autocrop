from mainwindow import UiMainWindow
from PyQt6 import QtWidgets

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    app.setStyle('Fusion')
    main_window = UiMainWindow()
    main_window.show()
    app.exec()
