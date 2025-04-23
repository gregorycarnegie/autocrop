from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Union

import cv2
import rawpy
from rawpy._rawpy import NotSupportedError, LibRawError, LibRawFatalError, LibRawNonFatalError
from PyQt6.QtCore import pyqtSignal, QObject
from PyQt6.QtGui import QImage

from core.croppers.base import Cropper
from core.enums import FunctionType
from core.face_tools import FaceToolPair
from core import processing as prc
from core.job import Job
from file_types import file_manager, FileCategory


class EventEmitter(QObject):
    image_updated = pyqtSignal(FunctionType, object)


@dataclass(slots=True, frozen=True, repr=True)
class Preview:
    """
    A data class representing a preview image.
    """
    image: Optional[cv2.Mat]
    color_space: QImage.Format


class DisplayCropper(Cropper):
    def __init__(self, face_detection_tools: FaceToolPair):
        super().__init__()
        self.face_detection_tools = face_detection_tools
        self.events = EventEmitter()
        self._widget_states = {}
        self._input_paths = {}
        
        # Simple cache for the currently loaded raw image - keyed by function type
        self.preview_data: Dict[FunctionType, Optional[Preview]] = {}
        self.current_paths: Dict[FunctionType, Optional[str]] = {}

    def register_widget_state(self, function_type: FunctionType, get_state_callback: Callable[[], tuple], get_path_callback: Callable[[], str]):
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
            return None
        
        # Only load image if the path changed or no image loaded yet for this function type
        file_category: Optional[FileCategory] = None
        current_path = self.current_paths.get(function_type)
        if current_path != img_path_str or self.preview_data.get(function_type) is None:
            # Load appropriate image data based on the function type
            if (image := self._load_appropriate_image(function_type, img_path_str)) is None:
                return None
                
            raw_image, file_category = image
            
            if raw_image is None or file_category is None:
                # print(f"Failed to load image for {function_type.name}: {img_path_str}")
                return None
                
            self.preview_data[function_type] = Preview(
                raw_image,
                QImage.Format.Format_RGB888 if file_category == FileCategory.PHOTO else QImage.Format.Format_BGR888
            )
            self.current_paths[function_type] = img_path_str
        
        # Now process the cached image with current settings
        with suppress(cv2.error):
            # Create a job with all settings
            job = self._create_job_from_widget_state(widget_state, img_path_str, function_type)

            if image := self._process_cached_image(function_type, job, file_category):
                self.events.image_updated.emit(function_type, image)

        return None

    def _load_appropriate_image(self, function_type: FunctionType, path_str: str) -> Union[tuple[cv2.Mat, FileCategory] | tuple[None, None]]:
        """Load the appropriate image based on the function type"""
        match function_type:
            case FunctionType.PHOTO:
                # For Photo tab, load a single image file directly
                return self._load_raw_image(Path(path_str))

            case FunctionType.FOLDER | FunctionType.MAPPING:
                # For Folder and Mapping tabs, find the first image in the directory
                folder_path = Path(path_str)
                if folder_path.is_dir():
                    # Find the first valid image in the directory
                    for file_path in folder_path.iterdir():
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
    def _load_raw_image(file_path: Path) -> Union[tuple[cv2.Mat, FileCategory] | tuple[None, None]]:
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
    def _create_job_from_widget_state(widget_state, img_path_str: str, function_type: FunctionType) -> Job:
        """Create a Job with all parameters from widget state"""
        input_path, width_text, height_text, fix_exposure, multi_face, auto_tilt, sensitivity, face_percent, \
            gamma, top, bottom, left, right, radio_buttons = widget_state
            
        job_params = {
            'width': int(width_text) if width_text.isdigit() else 0,
            'height': int(height_text) if height_text.isdigit() else 0,
            'fix_exposure_job': fix_exposure,
            'multi_face_job': multi_face,
            'auto_tilt_job': auto_tilt,
            'sensitivity': sensitivity,
            'face_percent': face_percent,
            'gamma': gamma,
            'top': top,
            'bottom': bottom,
            'left': left,
            'right': right,
            'radio_buttons': radio_buttons,
        }
        
        # Add the appropriate path parameter based on the function type
        match function_type:
            case FunctionType.PHOTO:
                job_params['photo_path'] = Path(img_path_str)
            case FunctionType.FOLDER | FunctionType.MAPPING:
                job_params['folder_path'] = Path(img_path_str)
        
        return Job(**job_params)
    
    def _process_cached_image(self, function_type: FunctionType, job: Job, file_category: FileCategory) -> Optional[QImage]:
        """Process the cached raw image with the job parameters"""
        # Get the cached image for this function type
        cached_image = self.preview_data.get(function_type)

        if cached_image.image is None:
            return None

        # Create a copy to avoid modifying the original
        image_copy = cached_image.image.copy()

        if job.multi_face_job:
            # For multi-face mode, show annotations
            processed_image = prc.annotate_faces(image_copy, job, self.face_detection_tools)
            if processed_image is None:
                return None
        else:
            # For single-face mode, crop and process
            bounding_box = prc.detect_face_box(image_copy, job, self.face_detection_tools)
            if not bounding_box:
                return None

            # Create and apply the processing pipeline
            pipeline = prc.build_processing_pipeline(job, self.face_detection_tools, bounding_box, True)
            processed_image = prc.run_processing_pipeline(image_copy, pipeline)

        return self._convert_to_qimage(processed_image, file_category, cached_image.color_space)
    
    @staticmethod
    def _convert_to_qimage(cv_image: cv2.Mat, file_category: FileCategory, color_space: QImage.Format) -> QImage:
        """Convert OpenCV image to QImage"""
        height, width, channels = cv_image.shape
        bytes_per_line = channels * width

        return QImage(cv_image.data, width, height, bytes_per_line, color_space)
    
    def clear_cache(self, function_type: Optional[FunctionType] = None):
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
