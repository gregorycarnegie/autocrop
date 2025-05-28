from PyQt6.QtCore import QObject

from .svg_radio_button import SvgRadioButton


class SvgRadioButtonGroup(QObject):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._buttons = []

    def addButton(self, button: SvgRadioButton):
        if button not in self._buttons:
            self._buttons.append(button)
            button.toggled.connect(self._on_button_toggled)

    def _on_button_toggled(self, checked: bool):
        if checked:
            sender = self.sender()
            for button in self._buttons:
                if button is not sender and button.isChecked():
                    button.setChecked(False)
