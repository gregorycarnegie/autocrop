from ui import utils as ut
from line_edits import NumberLineEdit, PathLineEdit
from PyQt6 import QtWidgets

class TabStateManager:
    @staticmethod
    def disable_buttons(line_edits, buttons):
        """Generic button disabling based on lineedit states"""
        is_valid = ut.all_filled(*line_edits)
        ut.change_widget_state(is_valid, *buttons)

    def connect_input_widgets(self, *input_widgets):
        """Connect input widgets to state changes"""
        for input_widget in input_widgets:
            if isinstance(input_widget, (NumberLineEdit, PathLineEdit)):
                input_widget.textChanged.connect(self.disable_buttons)
            elif isinstance(input_widget, QtWidgets.QCheckBox):
                self.connect_checkbox(input_widget)
    