"""Type stubs for rawpy.enhance module"""

import numpy as np
from rawpy._rawpy import RawPy

def find_bad_pixels(
        paths: list[str],
        find_hot: bool = True,
        find_dead: bool = True,
        confirm_ratio: float = 0.9
) -> np.ndarray:
    """Find and return coordinates of hot/dead pixels in the given RAW images.

    The probability that a detected bad pixel is really a bad pixel gets higher
    the more input images are given. The images should be taken around the same time,
    that is, each image must contain the same bad pixels. Also, there should be movement
    between the images to avoid the false detection of bad pixels in non-moving high-contrast areas.

    Parameters:
        paths: paths to RAW images shot with the same camera
        find_hot: whether to find hot pixels
        find_dead: whether to find dead pixels
        confirm_ratio: ratio of how many out of all given images must contain a bad pixel to confirm it

    Returns:
        ndarray of shape (n,2) with y,x coordinates relative to visible RAW size
    """
    ...

def repair_bad_pixels(raw: RawPy, coords: np.ndarray, method: str = 'median') -> None:
    """Repair bad pixels in the given RAW image.

    Note that the function works in-place on the RAW image data.
    It has to be called before postprocessing the image.

    Parameters:
        raw: the RAW image to repair
        coords: coordinates of bad pixels to repair, array of shape (n,2)
               with y,x coordinates relative to visible RAW size
        method: currently only 'median' is supported
    """
    ...

def save_dcraw_bad_pixels(path: str, bad_pixels: np.ndarray) -> None:
    """Save the given bad pixel coordinates in dcraw file format.

    Note that timestamps cannot be set and will be written as zero.

    Parameters:
        path: path of the badpixels file that will be written
        bad_pixels: array of shape (n,2) with y,x coordinates relative to visible image area
    """
    ...
