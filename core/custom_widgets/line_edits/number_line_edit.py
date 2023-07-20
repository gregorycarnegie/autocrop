from .custom_line_edit import CustomLineEdit
    

class NumberLineEdit(CustomLineEdit):
    def validate_path(self):
        """Validate QLineEdit based on input and set color accordingly."""
        self.color_logic(self.text().isdigit())
        self.update_style()
