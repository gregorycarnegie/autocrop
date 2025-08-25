"""
Centralized colour space handling utilities for the autocrop application.

This module provides common functions for colour space conversions and manipulations,
eliminating code duplication across the application.
"""
import autocrop_rs.image_processing as r_img  # type: ignore
import cv2
import cv2.typing as cvt
import numpy as np

from .config import config


def ensure_rgb(image: cvt.MatLike) -> cvt.MatLike:
    """
    Ensures the image is in RGB format by converting from BGR if necessary.

    Args:
        image: Input image, assumed to be in BGR format (OpenCV default)

    Returns:
        Image in RGB format
    """
    # Only convert if the image has 3 channels (colour image)
    if len(image.shape) >= 3 and image.shape[2] >= 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def ensure_bgr(image: cvt.MatLike) -> cvt.MatLike:
    """
    Ensures the image is in BGR format (OpenCV standard) by converting from RGB if necessary.

    Args:
        image: Input image, assumed to be in RGB format

    Returns:
        Image in BGR format for OpenCV operations
    """
    # Only convert if the image has 3 channels (colour image)
    if len(image.shape) >= 3 and image.shape[2] >= 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def to_grayscale(image: cvt.MatLike) -> cvt.MatLike:
    """
    Converts an image to grayscale using specified coefficients.

    Args:
        image: Input image in BGR format (OpenCV standard)

    Returns:
        Grayscale image
    """
    # If the image is already grayscale, return as is
    if image.shape[2] == 1:
        return image

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def adjust_gamma(image: cvt.MatLike, gam: float) -> cvt.MatLike:
    """
    Adjusts image gamma using a precomputed lookup table.
    """
    return cv2.LUT(image, r_img.gamma(gam * config.gamma_threshold))


def normalize_image(image: cvt.MatLike) -> cvt.MatLike:
    """
    Normalizes an image to use the full dynamic range.

    Args:
        image: Input image

    Returns:
        Normalized image
    """
    # Get min and max values
    min_val = np.min(image)

    # Avoid division by zero
    if (delta_val := np.max(image) - min_val) == 0:
        return image

    # Normalize to [0, 255]
    return cv2.convertScaleAbs(image, alpha=float(255/delta_val), beta=float(-min_val*255/delta_val))
