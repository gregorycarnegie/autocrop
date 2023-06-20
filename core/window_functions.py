import re
from .custom_widgets import UiDialog
from .cropper import Cropper
from os import startfile
from PyQt6 import QtCore, QtWidgets
from typing import Optional, Union, Tuple

def setup_radio_button(parent: QtWidgets.QWidget, 
                       layout: Union[QtWidgets.QHBoxLayout, QtWidgets.QVBoxLayout],
                       filetype: str, 
                       series: int, 
                       checked: Optional[bool] = False,
                       spacer: Optional[bool] = False) -> QtWidgets.QRadioButton:
    stylesheet = """QRadioButton::indicator:checked{
        image: url(resources/icons/file_string_checked.svg);
        }
        QRadioButton::indicator:unchecked{
            image: url(resources/icons/file_string_unchecked.svg);
        }"""
    radioButton = QtWidgets.QRadioButton(parent=parent)
    radioButton.setStyleSheet(re.sub('_string', filetype, stylesheet))
    radioButton.setText("")
    radioButton.setIconSize(QtCore.QSize(64, 64))
    if checked:
        radioButton.setChecked(True)
    radioButton.setObjectName(f"radioButton_{series}")
    layout.addWidget(radioButton)
    if spacer:
        spacerItem = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum
        )
        layout.addItem(spacerItem)
    return radioButton

def setup_frame(parent: QtWidgets.QWidget, 
                name: str, 
                set_size: Optional[bool] = False) -> QtWidgets.QFrame:
    frame = QtWidgets.QFrame(parent=parent)
    if set_size:
        frame.setMinimumSize(QtCore.QSize(0, 40))
        frame.setMaximumSize(QtCore.QSize(16_777_215, 40))
    frame.setStyleSheet("background: #1f2c33")
    frame.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    frame.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
    frame.setObjectName(name)
    return frame

def setup_progress_bar(parent: QtWidgets.QWidget,
                       name: str,
                       layout: Union[QtWidgets.QHBoxLayout, QtWidgets.QVBoxLayout]) -> QtWidgets.QProgressBar:
    progress_bar = QtWidgets.QProgressBar(parent=parent)
    progress_bar.setMinimumSize(QtCore.QSize(0, 12))
    progress_bar.setMaximumSize(QtCore.QSize(16_777_215, 12))
    progress_bar.setProperty("value", 0)
    progress_bar.setTextVisible(False)
    progress_bar.setObjectName(name)
    layout.addWidget(progress_bar)
    return progress_bar

def setup_lcd(parent: QtWidgets.QWidget,
              name: str,
              layout: Union[QtWidgets.QHBoxLayout, QtWidgets.QVBoxLayout],
              int_val: Optional[int] = None) -> QtWidgets.QLCDNumber:
    lcd_number = QtWidgets.QLCDNumber(parent=parent)
    lcd_number.setStyleSheet("background : lightgreen; color : gray;")
    if int_val is not None:
        lcd_number.setProperty("intValue", int_val)
    lcd_number.setObjectName(name)
    layout.addWidget(lcd_number)
    return lcd_number

def setup_combo(parent: QtWidgets.QWidget,
                name: str,
                layout: Union[QtWidgets.QHBoxLayout, QtWidgets.QVBoxLayout]) -> QtWidgets.QComboBox:
    combo_box = QtWidgets.QComboBox(parent=parent)
    combo_box.setMinimumSize(QtCore.QSize(0, 22))
    combo_box.setMaximumSize(QtCore.QSize(16_777_215, 22))
    combo_box.setObjectName(name)
    layout.addWidget(combo_box)
    return combo_box

def setup_dial(parent: QtWidgets.QWidget, 
               min_: Optional[int] = None, 
               max_: Optional[int] = None,
               snglstp: Optional[int] = None, 
               pgstp: Optional[int] = None, 
               dval: Optional[int] = None,
               position: Optional[int] = None, 
               invapp: Optional[bool] = None, 
               invctrl: Optional[bool] = None,
               wrap: Optional[bool] = None, 
               notchvis: Optional[bool] = None, 
               name: Optional[str] = None,
               layout: Optional[Union[QtWidgets.QHBoxLayout, QtWidgets.QVBoxLayout]] = None) -> QtWidgets.QDial:

    dial = QtWidgets.QDial(parent=parent)

    parameters = {
        'minimum': min_,
        'maximum': max_,
        'singleStep': snglstp,
        'pageStep': pgstp,
        'value': dval,
        'sliderPosition': position,
        'invertedAppearance': invapp,
        'invertedControls': invctrl,
        'wrapping': wrap,
        'notchesVisible': notchvis,
        'objectName': name
    }

    methods = {
        'minimum': dial.setMinimum,
        'maximum': dial.setMaximum,
        'singleStep': dial.setSingleStep,
        'pageStep': dial.setPageStep,
        'value': dial.setValue,
        'sliderPosition': dial.setSliderPosition,
        'invertedAppearance': dial.setInvertedAppearance,
        'invertedControls': dial.setInvertedControls,
        'wrapping': dial.setWrapping,
        'notchesVisible': dial.setNotchesVisible,
        'objectName': dial.setObjectName,
    }

    for attr, value in parameters.items():
        if value is not None:
            methods[attr](value)

    if layout is not None:
        layout.addWidget(dial)

    return dial


def setup_dial_area(parent: QtWidgets.QWidget,
                    name: str,
                    label_name:str,
                    layout_name: str,
                    main_layout: Union[QtWidgets.QHBoxLayout, QtWidgets.QVBoxLayout]) -> Tuple[QtWidgets.QDial, QtWidgets.QLCDNumber, QtWidgets.QLabel]:
    vertical_layout = QtWidgets.QVBoxLayout()
    vertical_layout.setObjectName(layout_name)

    dial = setup_dial(parent, max_=100, notchvis=True, name=f'{name}Dial', layout=vertical_layout)
    horizontal_layout = QtWidgets.QHBoxLayout()
    horizontal_layout.setObjectName(layout_name.replace("vertical", "horizontal"))

    spacer_item = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
    horizontal_layout.addItem(spacer_item)
    
    label = QtWidgets.QLabel(parent=parent)
    label.setObjectName(label_name)
    horizontal_layout.addWidget(label)
    
    lcd_number = setup_lcd(parent, f'{name}LCDNumber', horizontal_layout)
    
    horizontal_layout.addItem(spacer_item)
    
    vertical_layout.addLayout(horizontal_layout)
    main_layout.addLayout(vertical_layout)
    return dial, lcd_number, label

def uncheck_boxes(*checkboxes: QtWidgets.QCheckBox) -> None:
    for checkbox in checkboxes:
        checkbox.setCheckState(QtCore.Qt.CheckState.Unchecked)

def load_about_form() -> None:
    about_ui = UiDialog()
    about_ui.exec()

def show_message_box(destination: QtWidgets.QLineEdit) -> None:
    def message_button(answer):
        if answer.text() == '&Yes':
            startfile(destination.text())

    def helper_function(msg_box: QtWidgets.QMessageBox):
        msg_box.setWindowTitle('Open Destination Folder')
        msg_box.setText('Open destination folder?')
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Question)
        msg_box.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No)
        msg_box.buttonClicked.connect(message_button)
        x = msg_box.exec()

    msg = QtWidgets.QMessageBox()
    helper_function(msg)

def disable_widget(*args: QtWidgets.QWidget) -> None:
    for arg in args:
        arg.setDisabled(True)

def enable_widget(*args: QtWidgets.QWidget) -> None:
    for arg in args:
        arg.setEnabled(True)

def change_widget_state(boolean: bool, *args: QtWidgets.QWidget) -> None:
    for arg in args:
        if boolean:
            arg.setEnabled(boolean)
        else:
            arg.setDisabled(not boolean)

def terminate(cropper: Cropper) -> None:
    cropper.end_task = True