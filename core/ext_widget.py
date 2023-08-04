import re
from typing import Optional

from PyQt6 import QtCore, QtWidgets

from .literals import FileExtension

RADIO_STYLESHEET = """QRadioButton::indicator:checked{
        image: url(resources/icons/file_string_checked.svg);
        }
        QRadioButton::indicator:unchecked{
            image: url(resources/icons/file_string_unchecked.svg);
        }"""


class ExtWidget(QtWidgets.QWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName('Form')
        self.horizontalLayout = QtWidgets.QHBoxLayout(self)
        self.horizontalLayout.setObjectName('horizontalLayout')
        self.radioButton_1 = QtWidgets.QRadioButton(parent=self)
        self.radioButton_1.setStyleSheet(self.set_style_sheet('no'))
        self.radioButton_1.setText('')
        self.radioButton_1.setIconSize(QtCore.QSize(64, 64))
        self.radioButton_1.setChecked(True)
        self.radioButton_1.setObjectName('radioButton_1')
        self.horizontalLayout.addWidget(self.radioButton_1)
        spacerItem = QtWidgets.QSpacerItem(
            293, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.radioButton_2 = QtWidgets.QRadioButton(parent=self)
        self.radioButton_2.setStyleSheet(self.set_style_sheet('bmp'))
        self.radioButton_2.setText('')
        self.radioButton_2.setObjectName('radioButton_2')
        self.horizontalLayout.addWidget(self.radioButton_2)
        self.radioButton_3 = QtWidgets.QRadioButton(parent=self)
        self.radioButton_3.setStyleSheet(self.set_style_sheet('jpg'))
        self.radioButton_3.setText('')
        self.radioButton_3.setObjectName('radioButton_3')
        self.horizontalLayout.addWidget(self.radioButton_3)
        self.radioButton_4 = QtWidgets.QRadioButton(parent=self)
        self.radioButton_4.setStyleSheet(self.set_style_sheet('png'))
        self.radioButton_4.setText('')
        self.radioButton_4.setObjectName('radioButton_4')
        self.horizontalLayout.addWidget(self.radioButton_4)
        self.radioButton_5 = QtWidgets.QRadioButton(parent=self)
        self.radioButton_5.setStyleSheet(self.set_style_sheet('tiff'))
        self.radioButton_5.setText('')
        self.radioButton_5.setObjectName('radioButton_5')
        self.horizontalLayout.addWidget(self.radioButton_5)
        self.radioButton_6 = QtWidgets.QRadioButton(parent=self)
        self.radioButton_6.setStyleSheet(self.set_style_sheet('webp'))
        self.radioButton_6.setText('')
        self.radioButton_6.setObjectName('radioButton_6')
        self.horizontalLayout.addWidget(self.radioButton_6)

        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate('Form', 'Form'))

    @staticmethod
    def set_style_sheet(file_type: FileExtension):
        return re.sub('_string', f'_{file_type}', RADIO_STYLESHEET)
