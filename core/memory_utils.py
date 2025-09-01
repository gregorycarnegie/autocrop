"""
Memory optimization utilities for the autocrop application.
Provides functions to monitor and manage memory usage during image processing.
"""

import gc
import sys
import psutil
from typing import Any, Callable, TypeVar
import cv2
import numpy as np

T = TypeVar('T')

class MemoryManager:
    """Manages memory usage and provides optimization utilities."""
    
    def __init__(self):
        self.process = psutil.Process()
        self._initial_memory = self.process.memory_info().rss
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def get_memory_growth(self) -> float:
        """Get memory growth since initialization in MB."""
        current = self.process.memory_info().rss
        return (current - self._initial_memory) / 1024 / 1024
    
    def cleanup_memory(self) -> None:
        """Force garbage collection and clear OpenCV caches."""
        gc.collect()
        # Clear OpenCV memory pools
        cv2.setUseOptimized(True)
        cv2.setNumThreads(1)
    
    def is_memory_critical(self, threshold_mb: float = 1000) -> bool:
        """Check if memory usage is approaching critical levels."""
        available_mb = psutil.virtual_memory().available / 1024 / 1024
        return available_mb < threshold_mb

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

def with_memory_cleanup(func: Callable[..., T]) -> Callable[..., T]:
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
    if array.dtype == np.float64:
        # Check if values fit in float32 range
        if np.all(np.isfinite(array)) and array.max() <= np.finfo(np.float32).max:
            array = array.astype(np.float32)
    
    return array

class MemoryLimitedPool:
    """
    A pool that limits the number of concurrent operations based on available memory.
    """
    
    def __init__(self, max_workers: int = None, memory_limit_mb: float = 2000):
        self.max_workers = max_workers or min(4, psutil.cpu_count())
        self.memory_limit_mb = memory_limit_mb
        self._current_workers = 0
    
    def can_start_worker(self) -> bool:
        """Check if we can start another worker without exceeding memory limits."""
        if self._current_workers >= self.max_workers:
            return False
        
        available_mb = psutil.virtual_memory().available / 1024 / 1024
        return available_mb > self.memory_limit_mb
    
    def acquire_worker(self) -> bool:
        """Try to acquire a worker slot."""
        if self.can_start_worker():
            self._current_workers += 1
            return True
        return False
    
    def release_worker(self) -> None:
        """Release a worker slot."""
        if self._current_workers > 0:
            self._current_workers -= 1
            gc.collect()  # Clean up after each worker