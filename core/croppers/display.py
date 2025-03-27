from collections.abc import Callable
from functools import lru_cache
from typing import Optional, TypeVar

from PyQt6.QtCore import pyqtSignal, QObject
from PyQt6.QtGui import QImage

from core.croppers.base import Cropper
from core.enums import FunctionType
from core.operation_types import FaceToolPair
from .display_crop_utils import perform_crop_helper, WidgetState

T = TypeVar('T')


class EventEmitter(QObject):
    image_updated = pyqtSignal(FunctionType, object)


class DisplayCropper(Cropper):
    def __init__(self, face_detection_tools: FaceToolPair):
        super().__init__()
        self.face_detection_tools = face_detection_tools
        self.events = EventEmitter()
        self._widget_states = {}
        self._input_paths = {}
        
    def register_widget_state(self, function_type: FunctionType, get_state_callback: Callable[[], tuple], get_path_callback: Callable[[], str]):
        """Register efficient callbacks to get state information directly without dependencies"""
        self._widget_states[function_type] = get_state_callback
        self._input_paths[function_type] = get_path_callback
    
    def crop(self, function_type: FunctionType) -> None:
        """Perform the crop operation with minimal overhead"""
        if function_type not in self._widget_states:
            return

        widget_state = self._widget_states[function_type]()
        img_path_str = self._input_paths[function_type]()

        if image := self._perform_crop_cached(
            function_type, widget_state, img_path_str, self.face_detection_tools
        ):
            self.events.image_updated.emit(function_type, image)

    @staticmethod
    @lru_cache(maxsize=32)  # Smaller cache size for memory efficiency
    def _perform_crop_cached(
        function_type: FunctionType,
        widget_state: WidgetState,  # Using tuple instead of complex class for faster cache lookup
        img_path_str: str,
        face_detection_tools: FaceToolPair
    ) -> Optional[QImage]:
        return perform_crop_helper(function_type, widget_state, img_path_str, face_detection_tools)
