from mainwindow import UiMainWindow
from PyQt6 import QtWidgets
from utils import profileit

# @profileit
def main():
    app = QtWidgets.QApplication([])
    app.setStyle('Fusion')
    main_window = UiMainWindow()
    main_window.show()
    app.exec()   

if __name__ == "__main__":
    main()
