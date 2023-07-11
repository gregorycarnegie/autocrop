import re
from os import startfile
from typing import Optional, Union, Tuple

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import QCheckBox, QComboBox, QDial, QFrame, QHBoxLayout, QLabel, QLCDNumber, \
    QLineEdit, QMessageBox, QProgressBar, QRadioButton, QSizePolicy, QSpacerItem, QVBoxLayout, QWidget

from .cropper import Cropper
from .custom_widgets import UiDialog

RADIO_STYLESHEET = """QRadioButton::indicator:checked{
        image: url(resources/icons/file_string_checked.svg);
        }
        QRadioButton::indicator:unchecked{
            image: url(resources/icons/file_string_unchecked.svg);
        }"""

CHECKBOX_STYLESHEET = """QCheckBox:unchecked{color: red}
        QCheckBox:checked{color: white}
        QCheckBox::indicator{
                width: 20px;
                height: 20px;
        }
        QCheckBox::indicator:checked{
                image: url(resources/icons/checkbox_checked.svg);
        }
        QCheckBox::indicator:unchecked{
                image: url(resources/icons/checkbox_unchecked.svg);
        }
        QCheckBox::indicator:checked:hover{
                image: url(resources/icons/checkbox_checked_hover.svg);
        }
        QCheckBox::indicator:unchecked:hover{
                image: url(resources/icons/checkbox_unchecked_hover.svg);
        }"""

def setup_radio_button(parent: QWidget, 
                       layout: Union[QHBoxLayout, QVBoxLayout],
                       filetype: str, 
                       series: int, 
                       checked: Optional[bool] = False,
                       spacer: Optional[bool] = False) -> QRadioButton:
    radioButton = QRadioButton(parent=parent)
    radioButton.setStyleSheet(re.sub('_string', filetype, RADIO_STYLESHEET))
    radioButton.setText("")
    radioButton.setIconSize(QSize(64, 64))
    if checked:
        radioButton.setChecked(True)
    radioButton.setObjectName(f'radioButton_{series}')
    layout.addWidget(radioButton)
    if spacer:
        spacerItem = QSpacerItem(
            40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )
        layout.addItem(spacerItem)
    return radioButton

def setup_frame(parent: QWidget, 
                name: str, 
                set_size: Optional[bool] = False) -> QFrame:
    frame = QFrame(parent=parent)
    if set_size:
        frame.setMinimumSize(QSize(0, 40))
        frame.setMaximumSize(QSize(16_777_215, 40))
    frame.setStyleSheet('background: #1f2c33')
    frame.setFrameShape(QFrame.Shape.NoFrame)
    frame.setFrameShadow(QFrame.Shadow.Raised)
    frame.setObjectName(name)
    return frame

def finalize_widget(widget: QWidget,
                    name: str,
                    layout: Union[QHBoxLayout, QVBoxLayout]) -> QWidget:
    widget.setObjectName(name)
    layout.addWidget(widget)
    return widget

def setup_progress_bar(parent: QWidget,
                       name: str,
                       layout: Union[QHBoxLayout, QVBoxLayout]) -> QProgressBar:
    progress_bar = QProgressBar(parent=parent)
    progress_bar.setMinimumSize(QSize(0, 12))
    progress_bar.setMaximumSize(QSize(16_777_215, 12))
    progress_bar.setProperty('value', 0)
    progress_bar.setTextVisible(False)
    return finalize_widget(progress_bar, name, layout)

def setup_lcd(parent: QWidget,
              name: str,
              layout: Union[QHBoxLayout, QVBoxLayout],
              int_val: Optional[int] = None) -> QLCDNumber:
    lcd_number = QLCDNumber(parent=parent)
    lcd_number.setStyleSheet('background : lightgreen; color : gray;')
    if int_val is not None:
        lcd_number.setProperty('intValue', int_val)
    return finalize_widget(lcd_number, name, layout)

def setup_combo(parent: QWidget,
                name: str,
                layout: Union[QHBoxLayout, QVBoxLayout]) -> QComboBox:
    combo_box = QComboBox(parent=parent)
    combo_box.setMinimumSize(QSize(0, 22))
    combo_box.setMaximumSize(QSize(16_777_215, 22))
    return finalize_widget(combo_box, name, layout)

def setup_dial(parent: QWidget, 
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
               layout: Optional[Union[QHBoxLayout, QVBoxLayout]] = None) -> QDial:

    dial = QDial(parent=parent)
    
    if min_ is not None: dial.setMinimum(min_)
    if max_ is not None: dial.setMaximum(max_)
    if snglstp is not None: dial.setSingleStep(snglstp)
    if pgstp is not None: dial.setPageStep(pgstp)
    if dval is not None: dial.setValue(dval)
    if position is not None: dial.setSliderPosition(position)
    if invapp is not None: dial.setInvertedAppearance(invapp)
    if invctrl is not None: dial.setInvertedControls(invctrl)
    if wrap is not None: dial.setWrapping(wrap)
    if notchvis is not None: dial.setNotchesVisible(notchvis)
    if name is not None: dial.setObjectName(name)

    if layout is not None:
        layout.addWidget(dial)

    return dial

def setup_dial_area(parent: QWidget,
                    name: str,
                    label_name:str,
                    layout_name: str,
                    main_layout: Union[QHBoxLayout, QVBoxLayout]) -> Tuple[QDial, QLCDNumber, QLabel]:
    vertical_layout = QVBoxLayout()
    vertical_layout.setObjectName(layout_name)

    dial = setup_dial(parent, max_=100, notchvis=True, name=f'{name}Dial', layout=vertical_layout)
    horizontal_layout = QHBoxLayout()
    horizontal_layout.setObjectName(layout_name.replace('vertical', 'horizontal'))

    spacer_item = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
    horizontal_layout.addItem(spacer_item)
    
    label = QLabel(parent=parent)
    label.setObjectName(label_name)
    horizontal_layout.addWidget(label)
    
    lcd_number = setup_lcd(parent, f'{name}LCDNumber', horizontal_layout)
    
    horizontal_layout.addItem(spacer_item)
    
    vertical_layout.addLayout(horizontal_layout)
    main_layout.addLayout(vertical_layout)
    return dial, lcd_number, label

def uncheck_boxes(*checkboxes: QCheckBox) -> None:
    for checkbox in checkboxes:
        checkbox.setCheckState(Qt.CheckState.Unchecked)

def load_about_form() -> None:
    about_ui = UiDialog()
    about_ui.exec()

def show_message_box(destination: QLineEdit) -> None:
    def message_button(answer: QMessageBox):
        if answer.text() == '&Yes':
            startfile(destination.text())

    def helper_function(msg_box: QMessageBox):
        msg_box.setWindowTitle('Open Destination Folder')
        msg_box.setText('Open destination folder?')
        msg_box.setIcon(QMessageBox.Icon.Question)
        msg_box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        msg_box.buttonClicked.connect(message_button)
        x = msg_box.exec()

    msg = QMessageBox()
    helper_function(msg)

def disable_widget(*args: QWidget) -> None:
    for arg in args:
        arg.setDisabled(True)

def enable_widget(*args: QWidget) -> None:
    for arg in args:
        arg.setEnabled(True)

def change_widget_state(boolean: bool, *args: QWidget) -> None:
    for arg in args:
        if boolean:
            arg.setEnabled(boolean)
        else:
            arg.setDisabled(not boolean)

def terminate(cropper: Cropper, series: int) -> None:
    if series == 1:
        cropper.end_f_task = True
    elif series == 2:
        cropper.end_m_task = True
    elif series == 3:
        cropper.end_v_task = True
