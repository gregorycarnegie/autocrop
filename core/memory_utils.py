"""
Memory optimization utilities for the autocrop application.
Provides functions to monitor and manage memory usage during image processing.
"""

import gc
from collections.abc import Callable
from typing import TypeVar

import cv2
import numpy as np

T = TypeVar('T')

def memory_efficient_resize(image: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """
    Resize image with memory optimization by using appropriate interpolation
    and ensuring memory-efficient operations.
    """
    height, width = image.shape[:2]
    target_width, target_height = target_size

    # Choose interpolation based on scaling direction
    if target_width < width and target_height < height:
        # Downscaling - use INTER_AREA for better quality and speed
        interpolation = cv2.INTER_AREA
    else:
        # Upscaling - use INTER_CUBIC for better quality
        interpolation = cv2.INTER_CUBIC

    # Ensure input is contiguous for better performance
    if not image.flags['C_CONTIGUOUS']:
        image = np.ascontiguousarray(image)

    return cv2.resize(image, target_size, interpolation=interpolation)

def with_memory_cleanup[T](func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator that performs memory cleanup after function execution.
    """
    def wrapper(*args, **kwargs) -> T:
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            gc.collect()
    return wrapper

def optimize_array_memory(array: np.ndarray) -> np.ndarray:
    """
    Optimize numpy array memory layout for better performance.
    """
    if not array.flags['C_CONTIGUOUS']:
        array = np.ascontiguousarray(array)

    # Convert to most efficient dtype if possible
    if array.dtype == np.float64 and (np.all(np.isfinite(array)) and array.max() <= np.finfo(np.float32).max):
        array = array.astype(np.float32)

    return array
