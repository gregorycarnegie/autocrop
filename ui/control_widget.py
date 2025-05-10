from functools import cache
from typing import ClassVar

from PyQt6 import QtCore, QtWidgets

from core import ResourcePath
from line_edits import NumberLineEdit
from ui import utils as ut

from .svg_radio_button_group import SvgRadioButton, SvgRadioButtonGroup


@cache
def get_icon_path_tuple(icon_name: str) -> tuple[str, str]:
    """
    Return the *checked* and *unchecked* SVG paths for a “file_*” icon.

    Example
    -------
    >>> get_icon_path_tuple("folder")
    (Path('.../file_folder_checked.svg'),
     Path('.../file_folder_unchecked.svg'))
    """
    base = ResourcePath("resources/icons")
    checked   = (base / f"file_{icon_name}_checked.svg").as_resource()
    unchecked = (base / f"file_{icon_name}_unchecked.svg").as_resource()
    return checked, unchecked

RADIO_NONE = get_icon_path_tuple('no')
RADIO_BMP = get_icon_path_tuple('bmp')
RADIO_JPG = get_icon_path_tuple('jpg')
RADIO_PNG = get_icon_path_tuple('png')
RADIO_TIFF = get_icon_path_tuple('tiff')
RADIO_WEBP = get_icon_path_tuple('webp')

RadioButtonTuple = tuple[bool, bool, bool, bool, bool, bool]


class UiCropControlWidget(QtWidgets.QWidget):
    GAMMA_VAL: ClassVar[int] = 1000
    SENSITIVITY_VAL: ClassVar[int] = 50
    FPCT_VAL: ClassVar[int] = 62

    def __init__(self, parent: QtWidgets.QWidget):
        super().__init__(parent)
        self.setObjectName("CropControlWidget")
        self.horizontalLayout = ut.setup_hbox("horizontalLayout", self)
        self.verticalLayout_1 = ut.setup_vbox("verticalLayout_1")

        self._svg_radio_button_group = SvgRadioButtonGroup()

        self.radioButton_none = self.create_radio_button("radioButton_none", RADIO_NONE, self.verticalLayout_1)
        self.radioButton_none.setChecked(True)
        self.radioButton_bmp = self.create_radio_button("radioButton_bmp", RADIO_BMP, self.verticalLayout_1)
        self.radioButton_jpg = self.create_radio_button("radioButton_jpg", RADIO_JPG, self.verticalLayout_1)
        self.radioButton_png = self.create_radio_button("radioButton_png", RADIO_PNG, self.verticalLayout_1)
        self.radioButton_tiff = self.create_radio_button("radioButton_tiff", RADIO_TIFF, self.verticalLayout_1)
        self.radioButton_webp = self.create_radio_button("radioButton_webp", RADIO_WEBP, self.verticalLayout_1)

        self.verticalSpacer_1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum,
                                                      QtWidgets.QSizePolicy.Policy.Expanding)
        self.verticalLayout_1.addItem(self.verticalSpacer_1)

        self.horizontalLayout.addLayout(self.verticalLayout_1)

        self.horizontalSpacer_1 = QtWidgets.QSpacerItem(292, 20, QtWidgets.QSizePolicy.Policy.Expanding,
                                                        QtWidgets.QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_1)

        self.verticalLayout_2 = ut.setup_vbox("verticalLayout_2")
        self.gridLayout_1 = QtWidgets.QGridLayout()
        self.gridLayout_1.setObjectName("gridLayout_1")

        self.topDial = self.create_dial("topDial")
        self.gridLayout_1.addWidget(self.topDial, 0, 0, 1, 1)

        self.leftDial = self.create_dial("leftDial")
        self.gridLayout_1.addWidget(self.leftDial, 0, 2, 1, 1)

        self.rightDial = self.create_dial("rightDial")
        self.gridLayout_1.addWidget(self.rightDial, 0, 3, 1, 1)

        self.bottomDial = self.create_dial("bottomDial")
        self.gridLayout_1.addWidget(self.bottomDial, 0, 1, 1, 1)

        self.horizontalLayout_1 = ut.setup_hbox("horizontalLayout_1")
        self.topLabel = QtWidgets.QLabel(self)
        self.topLabel.setObjectName("topLabel")

        self.horizontalLayout_1.addWidget(self.topLabel)

        self.topLCDNumber = self.create_lcd_number("topLCDNumber")
        self.horizontalLayout_1.addWidget(self.topLCDNumber)

        self.gridLayout_1.addLayout(self.horizontalLayout_1, 1, 0, 1, 1)

        self.horizontalLayout_2 = ut.setup_hbox("horizontalLayout_2")
        self.bottomLabel = QtWidgets.QLabel(self)
        self.bottomLabel.setObjectName("bottomLabel")

        self.horizontalLayout_2.addWidget(self.bottomLabel)

        self.bottomLCDNumber = self.create_lcd_number("bottomLCDNumber")
        self.horizontalLayout_2.addWidget(self.bottomLCDNumber)

        self.gridLayout_1.addLayout(self.horizontalLayout_2, 1, 1, 1, 1)

        self.horizontalLayout_3 = ut.setup_hbox("horizontalLayout_3")
        self.leftLabel = QtWidgets.QLabel(self)
        self.leftLabel.setObjectName("leftLabel")

        self.horizontalLayout_3.addWidget(self.leftLabel)

        self.leftLCDNumber = self.create_lcd_number("leftLCDNumber")
        self.horizontalLayout_3.addWidget(self.leftLCDNumber)

        self.gridLayout_1.addLayout(self.horizontalLayout_3, 1, 2, 1, 1)

        self.horizontalLayout_4 = ut.setup_hbox("horizontalLayout_4")
        self.rightLabel = QtWidgets.QLabel(self)
        self.rightLabel.setObjectName("rightLabel")

        self.horizontalLayout_4.addWidget(self.rightLabel)

        self.rightLCDNumber = self.create_lcd_number("rightLCDNumber")
        self.horizontalLayout_4.addWidget(self.rightLCDNumber)

        self.gridLayout_1.addLayout(self.horizontalLayout_4, 1, 3, 1, 1)

        self.verticalLayout_2.addLayout(self.gridLayout_1)

        self.verticalSpacer_2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum,
                                                      QtWidgets.QSizePolicy.Policy.Expanding)
        self.verticalLayout_2.addItem(self.verticalSpacer_2)

        self.horizontalLayout_5 = ut.setup_hbox("horizontalLayout_5")

        self.horizontalSpacer_3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding,
                                                        QtWidgets.QSizePolicy.Policy.Minimum)

        self.horizontalLayout_5.addItem(self.horizontalSpacer_3)

        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")

        self.heightLabel = QtWidgets.QLabel(self)
        self.heightLabel.setObjectName("heightLabel")
        self.heightLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.gridLayout_2.addWidget(self.heightLabel, 0, 1, 1, 1)

        self.widthLabel = QtWidgets.QLabel(self)
        self.widthLabel.setObjectName("widthLabel")
        self.widthLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.gridLayout_2.addWidget(self.widthLabel, 0, 0, 1, 1)

        self.widthLineEdit = NumberLineEdit(name="widthLineEdit", parent=self)

        self.gridLayout_2.addWidget(self.widthLineEdit, 1, 0, 1, 1)

        self.heightLineEdit = NumberLineEdit(name="heightLineEdit", parent=self)

        self.gridLayout_2.addWidget(self.heightLineEdit, 1, 1, 1, 1)

        self.horizontalLayout_5.addLayout(self.gridLayout_2)

        self.horizontalSpacer_4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding,
                                                        QtWidgets.QSizePolicy.Policy.Minimum)

        self.horizontalLayout_5.addItem(self.horizontalSpacer_4)

        self.verticalLayout_2.addLayout(self.horizontalLayout_5)
        self.verticalLayout_2.setStretch(1, 1)

        self.horizontalLayout.addLayout(self.verticalLayout_2)

        self.horizontalSpacer_2 = QtWidgets.QSpacerItem(292, 20, QtWidgets.QSizePolicy.Policy.Expanding,
                                                        QtWidgets.QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_2)

        self.verticalLayout_3 = ut.setup_vbox("verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(-1, 6, -1, 6)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.gammaDial = self.create_dial("gammaDial", self.GAMMA_VAL, 2000)

        self.verticalLayout_4.addWidget(self.gammaDial)

        self.horizontalLayout_6 = ut.setup_hbox("horizontalLayout_6")
        self.gammaLabel = QtWidgets.QLabel(self)
        self.gammaLabel.setObjectName("gammaLabel")

        self.horizontalLayout_6.addWidget(self.gammaLabel)

        self.gammaLCDNumber = self.create_lcd_number("gammaLCDNumber", self.GAMMA_VAL)
        self.horizontalLayout_6.addWidget(self.gammaLCDNumber)

        self.verticalLayout_4.addLayout(self.horizontalLayout_6)

        self.verticalLayout_3.addLayout(self.verticalLayout_4)

        self.verticalLayout_5 = ut.setup_vbox("verticalLayout_5")

        self.sensitivityDial = self.create_dial("sensitivityDial", self.SENSITIVITY_VAL, min_value=1)
        self.verticalLayout_5.addWidget(self.sensitivityDial)

        self.horizontalLayout_7 = ut.setup_hbox("horizontalLayout_7")
        self.sensitivityLabel = QtWidgets.QLabel(self)
        self.sensitivityLabel.setObjectName("sensitivityLabel")

        self.horizontalLayout_7.addWidget(self.sensitivityLabel)

        self.sensitivityLCDNumber = self.create_lcd_number("sensitivityLCDNumber", self.SENSITIVITY_VAL)
        self.horizontalLayout_7.addWidget(self.sensitivityLCDNumber)

        self.verticalLayout_5.addLayout(self.horizontalLayout_7)

        self.verticalLayout_3.addLayout(self.verticalLayout_5)

        self.verticalLayout_6 = ut.setup_vbox("verticalLayout_6")

        self.fpctDial = self.create_dial("fpctDial", self.FPCT_VAL)
        self.verticalLayout_6.addWidget(self.fpctDial)

        self.horizontalLayout_8 = ut.setup_hbox("horizontalLayout_8")
        self.fpctLabel = QtWidgets.QLabel(self)
        self.fpctLabel.setObjectName("fpctLabel")

        self.horizontalLayout_8.addWidget(self.fpctLabel)

        self.fpctLCDNumber = self.create_lcd_number("fpctLCDNumber", self.FPCT_VAL)
        self.horizontalLayout_8.addWidget(self.fpctLCDNumber)

        self.verticalLayout_6.addLayout(self.horizontalLayout_8)

        self.verticalLayout_3.addLayout(self.verticalLayout_6)
        self.verticalSpacer_3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum,
                                                      QtWidgets.QSizePolicy.Policy.Expanding)
        self.verticalLayout_3.addItem(self.verticalSpacer_3)

        self.horizontalLayout.addLayout(self.verticalLayout_3)
        self.horizontalLayout.setStretch(1, 1)
        self.horizontalLayout.setStretch(3, 1)

        self.retranslateUi()
        self.topDial.valueChanged.connect(self.topLCDNumber.display)
        self.bottomDial.valueChanged.connect(self.bottomLCDNumber.display)
        self.leftDial.valueChanged.connect(self.leftLCDNumber.display)
        self.rightDial.valueChanged.connect(self.rightLCDNumber.display)
        self.gammaDial.valueChanged.connect(self.gammaLCDNumber.display)
        self.sensitivityDial.valueChanged.connect(self.sensitivityLCDNumber.display)
        self.fpctDial.valueChanged.connect(self.fpctLCDNumber.display)

        QtCore.QMetaObject.connectSlotsByName(self)

    def retranslateUi(self):
        self.setWindowTitle(QtCore.QCoreApplication.translate("self", "Form", None))
        self.radioButton_none.setText("")
        self.radioButton_bmp.setText("")
        self.radioButton_jpg.setText("")
        self.radioButton_png.setText("")
        self.radioButton_tiff.setText("")
        self.radioButton_webp.setText("")
        self.topLabel.setText(QtCore.QCoreApplication.translate("self", "Top:", None))
        self.bottomLabel.setText(QtCore.QCoreApplication.translate("self", "Bottom:", None))
        self.leftLabel.setText(QtCore.QCoreApplication.translate("self", "Left:", None))
        self.rightLabel.setText(QtCore.QCoreApplication.translate("self", "Right:", None))
        self.heightLabel.setText(QtCore.QCoreApplication.translate("self", "Height (px)", None))
        self.widthLabel.setText(QtCore.QCoreApplication.translate("self", "Width (px)", None))
        self.gammaLabel.setText(QtCore.QCoreApplication.translate("self", "Gamma:", None))
        self.sensitivityLabel.setText(QtCore.QCoreApplication.translate("self", "Sensitivity:", None))
        self.fpctLabel.setText(QtCore.QCoreApplication.translate("self", "Face%:", None))

    @property
    def radio_tuple(self) -> RadioButtonTuple:
        return (self.radioButton_none.isChecked(), self.radioButton_bmp.isChecked(),
                self.radioButton_jpg.isChecked(), self.radioButton_png.isChecked(),
                self.radioButton_tiff.isChecked(), self.radioButton_webp.isChecked())

    @staticmethod
    def format_style_sheet(style_sheet, true_path: str, false_path: str) -> str:
        return style_sheet.format(
            true_icon=true_path,
            false_icon=false_path,
        )

    def create_radio_button(self,
                            name: str,
                            icon_resource: tuple[str, str],
                            layout: QtWidgets.QHBoxLayout | QtWidgets.QVBoxLayout) -> SvgRadioButton:
        # Create the custom radio button
        radio_button = SvgRadioButton(
            parent=self,
            checked_icon=icon_resource[0],
            unchecked_icon=icon_resource[1]
        )
        radio_button.setObjectName(name)

        # Set expanding size policy
        radio_button.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding
        )

        # Add to a button group
        self._svg_radio_button_group.addButton(radio_button)

        # Add to layout
        layout.addWidget(radio_button)

        return radio_button

    def create_dial(self, name: str,
                    value: int = 0,
                    max_value: int = 100,
                    min_value: int | None = None) -> QtWidgets.QDial:
        dial = QtWidgets.QDial(self)
        dial.setObjectName(name)
        if min_value:
            dial.setMinimum(min_value)
        dial.setMaximum(max_value)
        dial.setValue(value)
        dial.setNotchesVisible(True)
        return dial

    def create_lcd_number(self, name: str, value: int = 0) -> QtWidgets.QLCDNumber:
        lcd_number = QtWidgets.QLCDNumber(self)
        lcd_number.setObjectName(name)
        lcd_number.setStyleSheet("background : lightgreen; color : gray;")
        lcd_number.setProperty("intValue", value)
        return lcd_number
