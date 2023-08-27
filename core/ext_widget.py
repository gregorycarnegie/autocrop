import re
from typing import ClassVar, Optional

from PyQt6 import QtCore, QtWidgets

from .literals import FileExtension
from core import window_functions as wf


class ExtWidget(QtWidgets.QWidget):
    RADIO_STYLESHEET: ClassVar[str] = """QRadioButton::indicator:checked {
        image: url(resources/icons/file_string_checked.svg);
        }
        QRadioButton::indicator:unchecked {
            image: url(resources/icons/file_string_unchecked.svg);
        }
        QRadioButton::indicator {
            width: 128px;
            height: 128px;
        }"""
    spacerItem: ClassVar[QtWidgets.QSpacerItem] = QtWidgets.QSpacerItem(
        293, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName('Form')
        self.horizontalLayout = QtWidgets.QHBoxLayout(self)
        self.horizontalLayout.setObjectName('horizontalLayout')
        self.radioButton_1 = self.setup_radio_button('radioButton_1', 'no')
        self.radioButton_1.setChecked(True)
        self.horizontalLayout.addWidget(self.radioButton_1)
        self.horizontalLayout.addItem(self.spacerItem)
        self.radioButton_2 = self.setup_radio_button('radioButton_2', 'bmp')
        self.radioButton_3 = self.setup_radio_button('radioButton_3', 'jpg')
        self.radioButton_4 = self.setup_radio_button('radioButton_4', 'png')
        self.radioButton_5 = self.setup_radio_button('radioButton_5', 'tiff')
        self.radioButton_6 = self.setup_radio_button('radioButton_6', 'webp')
        wf.add_widgets(self.horizontalLayout, self.radioButton_2, self.radioButton_3,
                       self.radioButton_4, self.radioButton_5, self.radioButton_6)
        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate('Form', 'Form'))

    def setup_radio_button(self, name: str, file_type: FileExtension) -> QtWidgets.QRadioButton:
        radio_button = QtWidgets.QRadioButton(parent=self)
        style_sheet = re.sub('_string', f'_{file_type}', self.RADIO_STYLESHEET)
        radio_button.setStyleSheet(style_sheet)
        radio_button.setText('')
        radio_button.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        radio_button.setObjectName(name)
        return radio_button
