from typing import Optional
from PyQt6.QtWidgets import QLineEdit, QWidget
    

class NumberLineEdit(QLineEdit):
    def __init__(self, parent: Optional[QWidget]=None):
        super(NumberLineEdit, self).__init__(parent)
        self.textChanged.connect(self.validate_path)
        self.validColour = '#7fda91' # light green
        self.invalidColour = '#ff6c6c' # rose
        self.setStyleSheet(f'background-color: {self.invalidColour}; color: black;')

    def validate_path(self):
        if self.text().isdigit():
            self.setStyleSheet(f'background-color: {self.validColour}; color: black;')
        else:
            self.setStyleSheet(f'background-color: {self.invalidColour}; color: black;')
        return
    