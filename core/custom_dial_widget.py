from typing import Optional

from PyQt6 import QtCore, QtWidgets


class CustomDialWidget(QtWidgets.QWidget):
    def __init__(self, _label: str,
                 _min: int = 1,
                 _max: int = 100,
                 single_step: int = 1,
                 page_step: int = 10,
                 _value: int = 0,
                 parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.object_name = f'{_label}_dialArea'
        self.setObjectName(self.object_name)
        self._label = f'{_label.capitalize()} %:' if _label=='face' else f'{_label.capitalize()}:'
        self.verticalLayout = QtWidgets.QVBoxLayout(self)
        self.verticalLayout.setObjectName('verticalLayout')
        self.dial = QtWidgets.QDial(parent=self)
        self.dial.setMinimum(_min)
        self.dial.setMaximum(_max)
        self.dial.setSingleStep(single_step)
        self.dial.setPageStep(page_step)
        self.dial.setSliderPosition(_value)
        self.dial.setProperty('value', _value)
        self.dial.setNotchesVisible(True)
        self.dial.setObjectName('dial')
        self.verticalLayout.addWidget(self.dial)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName('horizontalLayout')
        spacerItem = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.label = QtWidgets.QLabel(parent=self)
        self.label.setObjectName('label')
        self.horizontalLayout.addWidget(self.label)
        self.lcdNumber = QtWidgets.QLCDNumber(parent=self)
        self.lcdNumber.setStyleSheet('background : lightgreen; color : gray;')
        self.lcdNumber.setProperty('intValue', _value)
        self.lcdNumber.setObjectName('lcdNumber')
        self.horizontalLayout.addWidget(self.lcdNumber)
        spacerItem1 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.verticalLayout.setStretch(0, 1)

        #connections
        self.dial.valueChanged['int'].connect(self.lcdNumber.display)

        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate(self.object_name, self.object_name))
        self.label.setText(_translate(self.object_name, self._label))
