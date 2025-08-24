from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import cv2
import cv2.typing as cvt
import rawpy
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QImage
from rawpy._rawpy import (
    LibRawError,
    LibRawFatalError,
    LibRawNonFatalError,
    NotSupportedError,
)

from core import processing as prc
from core.croppers.base import Cropper
from core.croppers.display_crop_utils import WidgetState
from core.enums import FunctionType
from core.face_tools import FaceToolPair
from core.job import Job
from file_types import FileCategory, file_manager


class EventEmitter(QObject):
    image_updated = pyqtSignal(FunctionType, object)


@dataclass(slots=True, frozen=True)
class Preview:
    """
    A data class representing a preview image.
    """
    image: cvt.MatLike
    color_space: QImage.Format


class DisplayCropper(Cropper):
    def __init__(self, face_detection_tools: FaceToolPair):
        super().__init__()
        self.face_detection_tools = face_detection_tools
        self.events = EventEmitter()
        self._widget_states = {}
        self._input_paths = {}

        # Simple cache for the currently loaded raw image - keyed by function type
        self.preview_data: dict[FunctionType, Preview | None] = {}
        self.current_paths: dict[FunctionType, str | None] = {}

    def register_widget_state(
            self,
            function_type: FunctionType,
            get_state_callback: Callable[[], WidgetState],
            get_path_callback: Callable[[], str]
    ):
        """Register efficient callbacks to get state information directly without dependencies"""
        self._widget_states[function_type] = get_state_callback
        self._input_paths[function_type] = get_path_callback

        # Initialize cache entries for this function type
        if function_type not in self.preview_data:
            self.preview_data[function_type] = None
        if function_type not in self.current_paths:
            self.current_paths[function_type] = None

    def crop(self, function_type: FunctionType) -> None:
        """Perform the crop operation with simple image caching"""
        if function_type not in self._widget_states:
            return None

        # Skip video tab since it has its own display methods
        if function_type == FunctionType.VIDEO:
            return None

        # Get current widget state and path
        widget_state = self._widget_states[function_type]()
        img_path_str = self._input_paths[function_type]()

        # Validate path
        if not img_path_str:
            # Clear any previous no-face messages
            if hasattr(self, '_no_face_messages') and function_type in self._no_face_messages:
                del self._no_face_messages[function_type]
            self.events.image_updated.emit(function_type, None)
            return None

        # For folder and mapping tabs, we need to check if the folder has changed
        if function_type in (FunctionType.FOLDER, FunctionType.MAPPING):
            # Clear cache if the folder path has changed
            current_path = self.current_paths.get(function_type)
            if current_path != img_path_str:
                self.clear_cache(function_type)

        # Only load image if the path changed or no image loaded yet for this function type
        file_category: FileCategory | None = None
        current_path = self.current_paths.get(function_type)

        # For folder/mapping, always reload since we're looking for the first image
        if function_type in (FunctionType.FOLDER, FunctionType.MAPPING) or \
        current_path != img_path_str or \
        self.preview_data.get(function_type) is None:

            # Load appropriate image data based on the function type
            raw_image, file_category = self._load_appropriate_image(function_type, img_path_str)

            if raw_image is None or file_category is None:
                # Clear the cache for this function type if loading fails
                self.clear_cache(function_type)
                # Emit error message
                self._emit_no_face_detected(function_type, "Unable to load image")
                return None

            self.preview_data[function_type] = Preview(
                raw_image,
                QImage.Format.Format_BGR888 if file_category == FileCategory.RAW else QImage.Format.Format_RGB888
            )
            self.current_paths[function_type] = img_path_str

        # Now process the cached image with current settings
        with suppress(cv2.error, ValueError, TypeError):
            # Create a job with all settings
            job = self._create_job_from_widget_state(widget_state, img_path_str, function_type)

            if image := self._process_cached_image(function_type, job):
                self.events.image_updated.emit(function_type, image)
            # If image is None, _process_cached_image already emitted the no-face signal

        return None

    def _load_appropriate_image(
            self,
            function_type: FunctionType,
            path_str: str
    ) -> tuple[cvt.MatLike | None, FileCategory] | tuple[None, None]:
        """Load the appropriate image based on the function type"""
        match function_type:
            case FunctionType.PHOTO:
                # For Photo tab, load a single image file directly
                return self._load_raw_image(Path(path_str))

            case FunctionType.FOLDER | FunctionType.MAPPING:
                # For Folder and Mapping tabs, find the first image in the directory
                folder_path = Path(path_str)
                if folder_path.is_dir():
                    # Get all files and sort them for consistency
                    files = sorted(folder_path.iterdir())

                    # Find the first valid image in the directory
                    for file_path in files:
                        if file_path.is_file() and self._is_supported_image(file_path):
                            return self._load_raw_image(file_path)
                return None, None

            case _:
                # We don't handle VIDEO tab here - it uses its own display methods
                return None, None

    @staticmethod
    def _is_supported_image(file_path: Path) -> bool:
        """Check if the file is a supported image type"""
        return (file_manager.is_valid_type(file_path, FileCategory.PHOTO) or
                file_manager.is_valid_type(file_path, FileCategory.TIFF) or
                file_manager.is_valid_type(file_path, FileCategory.RAW))

    @staticmethod
    def _load_raw_image(file_path: Path) -> tuple[cvt.MatLike | None, FileCategory] | tuple[None, None]:
        """Load the raw image data without any processing"""
        with suppress(cv2.error):
            # Determine the file type
            if file_manager.is_valid_type(file_path, FileCategory.PHOTO) or \
               file_manager.is_valid_type(file_path, FileCategory.TIFF):
                # Standard image file - use OpenCV directly
                return cv2.imread(file_path.as_posix()), FileCategory.PHOTO

            elif file_manager.is_valid_type(file_path, FileCategory.RAW):
                # RAW image file - use rawpy
                with suppress(
                    NotSupportedError,
                    LibRawFatalError,
                    LibRawError,
                    LibRawNonFatalError,
                    MemoryError,
                    ValueError,
                    TypeError,
                ):
                    with rawpy.imread(file_path.as_posix()) as raw:
                        # Process the RAW file to get a standard image
                        return raw.postprocess(use_camera_wb=True), FileCategory.RAW

            return None, None

        return None, None

    @staticmethod
    def _create_job_from_widget_state(
        widget_state: WidgetState,
        img_path_str: str,
        function_type: FunctionType
    ) -> Job:
        """Create a Job with all parameters from widget state"""
        job_params = {
            'width': int(widget_state.width) if widget_state.width.isdigit() else 0,
            'height': int(widget_state.height) if widget_state.height.isdigit() else 0,
            'fix_exposure_job': widget_state.fix_exposure,
            'multi_face_job': widget_state.multi_face,
            'auto_tilt_job': widget_state.auto_tilt,
            'sensitivity': widget_state.sensitivity,
            'face_percent': widget_state.face_percent,
            'gamma': widget_state.gamma,
            'top': widget_state.top,
            'bottom': widget_state.bottom,
            'left': widget_state.left,
            'right': widget_state.right,
            'radio_buttons': widget_state.radio_buttons,
        }

        # Add the appropriate path parameter based on the function type
        match function_type:
            case FunctionType.PHOTO:
                job_params['photo_path'] = Path(img_path_str)
            case FunctionType.FOLDER | FunctionType.MAPPING:
                job_params['folder_path'] = Path(img_path_str)

        return Job(**job_params)

    def _process_cached_image(self,
                            function_type: FunctionType,
                            job: Job
    ) -> QImage | None:
        """Process the cached raw image with the job parameters"""
        # Get the cached image for this function type
        cached_image = self.preview_data.get(function_type)

        # Add None check for cached_image itself
        if cached_image is None:
            return None

        # Check if the image within cached_image is None
        if cached_image.image is None:
            return None

        # Create a copy to avoid modifying the original
        image_copy = cached_image.image.copy()

        if job.multi_face_job:
            # For multi-face mode, show annotations
            processed_image = prc.annotate_faces(image_copy, job, self.face_detection_tools)
            if processed_image is None:
                # No faces detected in multi-face mode
                self._emit_no_face_detected(function_type, "No faces detected")
                return None
        else:
            # For single-face mode, crop and process
            bounding_box = prc.detect_face_box(image_copy, job, self.face_detection_tools)
            if not bounding_box:
                # No face detected in single-face mode
                self._emit_no_face_detected(function_type, "No face detected")
                return None

            # Get rotation matrix if auto-tilt is enabled
            rotation_matrix = prc.get_rotation_matrix(image_copy, self.face_detection_tools, job)

            # Create and apply the processing pipeline with rotation matrix
            pipeline = prc.build_crop_instruction_pipeline(
                job, bounding_box, display=True, rotation_matrix=rotation_matrix
            )
            processed_image = prc.run_processing_pipeline(image_copy, pipeline)

        return self._convert_to_qimage(processed_image, cached_image.color_space)

    def _emit_no_face_detected(self, function_type: FunctionType, message: str):
        """Emit a signal indicating no face was detected"""
        # Create a special signal for no face detected
        # We'll use the existing image_updated signal but with None as the image
        # The receiving widget can check for None and display appropriate feedback
        self.events.image_updated.emit(function_type, None)

        # Store the error message for the widget to access
        self._no_face_messages = getattr(self, '_no_face_messages', {})
        self._no_face_messages[function_type] = message

    def get_no_face_message(self, function_type: FunctionType) -> str:
        """Get the no face message for a specific function type"""
        messages = getattr(self, '_no_face_messages', {})
        return messages.get(function_type, "No face detected")

    @staticmethod
    def _convert_to_qimage(cv_image: cvt.MatLike, color_space: QImage.Format) -> QImage:
        """Convert OpenCV image to QImage"""
        height, width, channels = cv_image.shape
        bytes_per_line = channels * width
        return QImage(bytes(cv_image.data), width, height, bytes_per_line, color_space)

    def clear_cache(self, function_type: FunctionType | None = None):
        """Clear the image cache for a specific function type or all types"""
        if function_type is not None:
            # Clear cache for the specific function type
            if function_type in self.preview_data:
                self.preview_data[function_type] = None
            if function_type in self.current_paths:
                self.current_paths[function_type] = None
        else:
            # Clear all caches
            for ft in FunctionType:
                if ft in self.preview_data:
                    self.preview_data[ft] = None
                if ft in self.current_paths:
                    self.current_paths[ft] = None
