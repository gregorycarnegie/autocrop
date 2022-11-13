import sys

from PyQt6.QtWidgets import QApplication
import pandas as pd
import numpy as np
import os
import ux
from settings import FileTypeList, SpreadSheet


def run():
    app = QApplication([])
    app.setStyle('Fusion')
    widget = ux.MainWindow()
    widget.show()
    sys.exit(app.exec())


def open_mapping(input_filename):
    """Given a filename, returns a numpy array"""
    extension = os.path.splitext(input_filename)[1].lower()
    if extension.lower() in SpreadSheet().list1:
        try:
            return pd.read_csv(input_filename)
        except FileNotFoundError:
            return None
    if extension.lower() in SpreadSheet().list2:
        try:
            return pd.read_excel(input_filename)
        except FileNotFoundError:
            return None
    return None


if __name__ == '__main__':
    run()
    # source_folder = 'C:\\Users\\Gregory\\Pictures\\Memes'
    #
    # data_frame = open_mapping('C:\\Users\\Gregory\\Pictures\\Memes\\Book1.xlsx')
