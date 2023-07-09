from .custom_line_edit import CustomLineEdit
    

class NumberLineEdit(CustomLineEdit):
    def validate_path(self):
        self.colour = self.validColour if self.text().isdigit() else self.invalidColour
        self.setStyleSheet(f'background-color: {self.colour}; color: black;')
        self.change_state()
