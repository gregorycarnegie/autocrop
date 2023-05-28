from mainwindow import Ui_MainWindow
from PyQt6 import QtWidgets

def main():
    app = QtWidgets.QApplication([])
    app.setStyle('Fusion')
    main_window = Ui_MainWindow()
    main_window.show()
    app.exec()

if __name__ == "__main__":
    main()
