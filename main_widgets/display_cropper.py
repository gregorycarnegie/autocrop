import collections.abc as c
from functools import lru_cache
from typing import Optional, TypedDict

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QImage

from core.croppers.cropper import Cropper
from core.enums import FunctionType
from core.job import Job
from core.operation_types import FaceToolPair
from .ui_crop_folder_widget import UiFolderTabWidget
from .ui_crop_map_widget import UiMappingTabWidget
from .ui_crop_photo_widget import UiPhotoTabWidget
from .ui_crop_vid_widget import UiVideoTabWidget
from .ui_crop_widget import UiCropWidget
from .display_crop_utils import WidgetState, perform_crop_helper


class WidgetData(TypedDict):
    widget: UiCropWidget
    crop_method: c.Callable[[], None]

class DisplayCropper(Cropper):
    image_changed = pyqtSignal(FunctionType, QImage)

    def __init__(self, face_detection_tools: FaceToolPair, *,
                 p_widget: UiPhotoTabWidget,
                 f_widget: UiFolderTabWidget,
                 m_widget: UiMappingTabWidget,
                 v_widget: UiVideoTabWidget):
        super().__init__()
        self.face_detection_tools = face_detection_tools

        # Map function types to their corresponding widgets
        self.widgets: dict[FunctionType, WidgetData] = {
            FunctionType.PHOTO: {'widget': p_widget,'crop_method': lambda: self.crop(FunctionType.PHOTO)},
            FunctionType.FOLDER: {'widget': f_widget,'crop_method': lambda: self.crop(FunctionType.FOLDER)},
            FunctionType.MAPPING: {'widget': m_widget,'crop_method': lambda: self.crop(FunctionType.MAPPING)},
            FunctionType.VIDEO: {'widget': v_widget,'crop_method': lambda: self.crop(FunctionType.VIDEO)}
        }

        # Connect signals for each widget
        for data in self.widgets.values():
            widget = data['widget']
            crop_method = data['crop_method']
            self.connect_signals(widget, crop_method)

        # Connect the image_changed signal to the set_image method
        self.image_changed.connect(self.set_image)

    @staticmethod
    def connect_signals(widget: UiCropWidget, crop_method):
        # List of the signals to connect
        signals = [
            widget.inputLineEdit.textChanged,
            widget.controlWidget.widthLineEdit.textChanged,
            widget.controlWidget.heightLineEdit.textChanged,
            widget.tiltCheckBox.checkStateChanged,
            widget.mfaceCheckBox.checkStateChanged,
            widget.exposureCheckBox.checkStateChanged,
            widget.controlWidget.sensitivityDial.valueChanged,
            widget.controlWidget.fpctDial.valueChanged,
            widget.controlWidget.gammaDial.valueChanged,
            widget.controlWidget.topDial.valueChanged,
            widget.controlWidget.bottomDial.valueChanged,
            widget.controlWidget.leftDial.valueChanged,
            widget.controlWidget.rightDial.valueChanged,
        ]
        # Connect all signals to the crop method
        for signal in signals:
            signal.connect(crop_method)

    def create_job(self, function_type: FunctionType) -> Optional[Job]:
        data: dict = self.widgets.get(function_type)
        if not data:
            return None
        widget: UiCropWidget = data['widget']
        control = widget.controlWidget
        return Job(
            control.widthLineEdit.value(),
            control.heightLineEdit.value(),
            widget.exposureCheckBox.isChecked(),
            widget.mfaceCheckBox.isChecked(),
            widget.tiltCheckBox.isChecked(),
            control.sensitivityDial.value(),
            control.fpctDial.value(),
            control.gammaDial.value(),
            control.topDial.value(),
            control.bottomDial.value(),
            control.leftDial.value(),
            control.rightDial.value(),
            control.radio_tuple,
            photo_path=widget.inputLineEdit.text() if function_type == FunctionType.PHOTO else None,
            folder_path=widget.inputLineEdit.text() if function_type != FunctionType.PHOTO else None,
        )

    def _get_widget_state(self, function_type: FunctionType) -> WidgetState:
        data = self.widgets[function_type]
        widget = data['widget']
        control = widget.controlWidget

        return (
            widget.inputLineEdit.text(),
            control.widthLineEdit.text(),
            control.heightLineEdit.text(),
            widget.exposureCheckBox.isChecked(),
            widget.mfaceCheckBox.isChecked(),
            widget.tiltCheckBox.isChecked(),
            control.sensitivityDial.value(),
            control.fpctDial.value(),
            control.gammaDial.value(),
            control.topDial.value(),
            control.bottomDial.value(),
            control.leftDial.value(),
            control.rightDial.value(),
            control.radio_tuple,
        )

    def crop(self, function_type: FunctionType) -> None:
        widget_state = self._get_widget_state(function_type)
        img_path_str = self.widgets[function_type]['widget'].inputLineEdit.text()

        image = self._perform_crop_cached(
            function_type,
            widget_state,
            img_path_str,
            self.face_detection_tools
        )

        if image is not None:
            self.image_changed.emit(function_type, image)

    @staticmethod
    @lru_cache(maxsize=128)
    def _perform_crop_cached(
        function_type: FunctionType,
        widget_state: WidgetState,
        img_path_str: str,
        face_detection_tools: FaceToolPair
    ) -> Optional[QImage]:
        # Call the helper function
        return perform_crop_helper(function_type, widget_state, img_path_str, face_detection_tools)

    def set_image(self, function_type: FunctionType, image: QImage) -> None:
        widget: UiCropWidget = self.widgets[function_type]['widget']
        widget.imageWidget.setImage(image)
