"""Type stubs for rawpy.

This module contains type stubs for the rawpy package.
rawpy is a Python wrapper for LibRaw, a library for reading and processing RAW camera images.
"""

from __future__ import annotations

import enum
from typing import Any, BinaryIO, NamedTuple

import numpy as np

class RawType(enum.IntEnum):
    """RAW image type."""
    Flat = 0
    Stack = 1

class ThumbFormat(enum.IntEnum):
    """Thumbnail/preview image type."""
    JPEG = 1
    BITMAP = 2

class DemosaicAlgorithm(enum.IntEnum):
    """Identifiers for demosaic algorithms."""
    LINEAR = 0
    VNG = 1
    PPG = 2
    AHD = 3
    DCB = 4
    MODIFIED_AHD = 5
    AFD = 6
    VCD = 7
    VCD_MODIFIED_AHD = 8
    LMMSE = 9
    AMAZE = 10
    DHT = 11
    AAHD = 12

    @property
    def isSupported(self) -> bool | None:
        """Return True if the demosaic algorithm is supported, False if it is not, and None if the support status is unknown."""
        ...

    def checkSupported(self) -> None:
        """Like isSupported but raises an exception for the False case."""
        ...

class ColorSpace(enum.IntEnum):
    """Output color space options."""
    raw = 0
    sRGB = 1
    Adobe = 2
    Wide = 3
    ProPhoto = 4
    XYZ = 5

class HighlightMode(enum.IntEnum):
    """Highlight mode for handling overexposed pixels."""
    Clip = 0
    Ignore = 1
    Blend = 2
    ReconstructDefault = 5

    @classmethod
    def Reconstruct(cls, level: int) -> HighlightMode:
        """Create a reconstruct highlight mode with specified level.

        Parameters:
            level (int): 3 to 9, low numbers favor whites, high numbers favor colors
        """
        ...

class FBDDNoiseReductionMode(enum.IntEnum):
    """FBDD noise reduction modes."""
    Off = 0
    Light = 1
    Full = 2

class ImageSizes(NamedTuple):
    """Container for various size information of RAW images."""
    raw_width: int
    raw_height: int
    width: int
    height: int
    top_margin: int
    left_margin: int
    iwidth: int
    iheight: int
    raw_pitch: int
    pixel_aspect: float
    flip: int

class Thumbnail(NamedTuple):
    """Container for thumbnail data."""
    format: ThumbFormat
    data: bytes | np.ndarray

class Params:
    """A class that handles postprocessing parameters."""
    def __init__(
        self,
        demosaic_algorithm: DemosaicAlgorithm | None = None,
        half_size: bool = False,
        four_color_rgb: bool = False,
        dcb_iterations: int = 0,
        dcb_enhance: bool = False,
        fbdd_noise_reduction: FBDDNoiseReductionMode = FBDDNoiseReductionMode.Off,
        noise_thr: float | None = None,
        median_filter_passes: int = 0,
        use_camera_wb: bool = False,
        use_auto_wb: bool = False,
        user_wb: list[float] | None = None,
        output_color: ColorSpace = ColorSpace.sRGB,
        output_bps: int = 8,
        user_flip: int | None = None,
        user_black: int | None = None,
        user_sat: int | None = None,
        no_auto_bright: bool = False,
        auto_bright_thr: float | None = None,
        adjust_maximum_thr: float = 0.75,
        bright: float = 1.0,
        highlight_mode: HighlightMode | int = HighlightMode.Clip,
        exp_shift: float | None = None,
        exp_preserve_highlights: float = 0.0,
        no_auto_scale: bool = False,
        gamma: tuple[float, float] | None = None,
        chromatic_aberration: tuple[float, float] | None = None,
        bad_pixels_path: str | None = None
    ) -> None:
        """Initialize postprocessing parameters.

        If use_camera_wb and use_auto_wb are False and user_wb is None, then
        daylight white balance correction is used.
        If both use_camera_wb and use_auto_wb are True, then use_auto_wb has priority.

        Parameters:
            demosaic_algorithm: Algorithm used for demosaicing, default is AHD
            half_size: Outputs image in half size by reducing each 2x2 block to one pixel
            four_color_rgb: Whether to use separate interpolations for two green channels
            dcb_iterations: Number of DCB correction passes, requires DCB demosaicing algorithm
            dcb_enhance: DCB interpolation with enhanced interpolated colors
            fbdd_noise_reduction: Controls FBDD noise reduction before demosaicing
            noise_thr: Threshold for wavelet denoising (default disabled)
            median_filter_passes: Number of median filter passes after demosaicing to reduce color artifacts
            use_camera_wb: Whether to use the as-shot white balance values
            use_auto_wb: Whether to try automatically calculating the white balance
            user_wb: List of length 4 with white balance multipliers for each color
            output_color: Output color space
            output_bps: Bits per sample (8 or 16)
            user_flip: 0=none, 3=180, 5=90CCW, 6=90CW, default is to use image orientation from RAW
            user_black: Custom black level
            user_sat: Saturation adjustment (custom white level)
            no_auto_bright: Whether to disable automatic increase of brightness
            auto_bright_thr: Ratio of clipped pixels when automatic brighness increase is used
            adjust_maximum_thr: Threshold for adjusting maximum allowed pixel value
            bright: Brightness scaling
            highlight_mode: Method for handling highlights
            exp_shift: Exposure shift in linear scale
            exp_preserve_highlights: Preserve highlights when lightening the image with exp_shift
            no_auto_scale: Whether to disable pixel value scaling
            gamma: Pair (power,slope), default is (2.222, 4.5) for rec. BT.709
            chromatic_aberration: Pair (red_scale, blue_scale), default is (1,1)
            bad_pixels_path: Path to dcraw bad pixels file
        """
        ...

class RawPy:
    """Load RAW images, work on their data, and create a postprocessed (demosaiced) image."""

    def __enter__(self) -> RawPy:
        """Context manager entry."""
        ...

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        ...

    def open_file(self, path: str) -> None:
        """Opens the given RAW image file. Should be followed by a call to unpack().

        This is a low-level method, consider using rawpy.imread() instead.

        Parameters:
            path: The path to the RAW image
        """
        ...

    def open_buffer(self, fileobj: BinaryIO) -> None:
        """Opens the given RAW image file-like object. Should be followed by a call to unpack().

        This is a low-level method, consider using rawpy.imread() instead.

        Parameters:
            fileobj: The file-like object
        """
        ...

    def unpack(self) -> None:
        """Unpacks/decodes the opened RAW image.

        This is a low-level method, consider using rawpy.imread() instead.
        """
        ...

    def unpack_thumb(self) -> None:
        """Unpacks/decodes the thumbnail/preview image, whichever is bigger.

        This is a low-level method, consider using extract_thumb() instead.
        """
        ...

    def dcraw_process(self, params: Params | None = None, **kw: Any) -> None:
        """Postprocess the currently loaded RAW image.

        This is a low-level method, consider using postprocess() instead.

        Parameters:
            params: The parameters to use for postprocessing
            **kw: Alternative way to provide postprocessing parameters
        """
        ...

    def dcraw_make_mem_image(self) -> np.ndarray:
        """Return the postprocessed image (see dcraw_process()) as numpy array.

        This is a low-level method, consider using postprocess() instead.

        Returns:
            ndarray of shape (h,w,c)
        """
        ...

    def dcraw_make_mem_thumb(self) -> Thumbnail:
        """Return the thumbnail/preview image (see unpack_thumb()) as Thumbnail object.

        This is a low-level method, consider using extract_thumb() instead.

        Returns:
            Thumbnail object
        """
        ...

    def close(self) -> None:
        """Release all resources and close the RAW image."""
        ...

    def raw_value(self, row: int, column: int) -> int:
        """Return RAW value at given position relative to the full RAW image.

        Only usable for flat RAW images (see raw_type property).
        """
        ...

    def raw_value_visible(self, row: int, column: int) -> int:
        """Return RAW value at given position relative to visible area of image.

        Only usable for flat RAW images (see raw_type property).
        """
        ...

    def raw_color(self, row: int, column: int) -> int:
        """Return color index for the given coordinates relative to the full RAW size.

        Only usable for flat RAW images (see raw_type property).
        """
        ...

    def extract_thumb(self) -> Thumbnail:
        """Extracts and returns the thumbnail/preview image (whichever is bigger).

        For JPEG thumbnails, data is a bytes object and can be written as-is to file.
        For bitmap thumbnails, data is an ndarray of shape (h,w,c).
        If no image exists or the format is unsupported, an exception is raised.

        Returns:
            Thumbnail object
        """
        ...

    def postprocess(self, params: Params | None = None, **kw: Any) -> np.ndarray:
        """Postprocess the currently loaded RAW image and return the new resulting image.

        Parameters:
            params: The parameters to use for postprocessing
            **kw: Alternative way to provide postprocessing parameters

        Returns:
            ndarray of shape (h,w,c)
        """
        ...

    @property
    def raw_type(self) -> RawType:
        """Return the RAW type."""
        ...

    @property
    def color_desc(self) -> bytes:
        """String description of colors numbered from 0 to 3 (RGBG,RGBE,GMCY, or GBTG)."""
        ...

    @property
    def raw_pattern(self) -> np.ndarray | None:
        """The smallest possible Bayer pattern of this image."""
        ...

    @property
    def raw_colors(self) -> np.ndarray:
        """An array of color indices for each pixel in the RAW image.

        Equivalent to calling raw_color(y,x) for each pixel.
        Only usable for flat RAW images (see raw_type property).

        Returns:
            ndarray of shape (h,w)
        """
        ...

    @property
    def raw_colors_visible(self) -> np.ndarray:
        """Like raw_colors but without margin.

        Returns:
            ndarray of shape (hv,wv)
        """
        ...

    @property
    def black_level_per_channel(self) -> list[int]:
        """Per-channel black level correction.

        Returns:
            list of length 4
        """
        ...

    @property
    def camera_white_level_per_channel(self) -> list[int] | None:
        """Per-channel saturation levels read from raw file metadata, if it exists.

        Returns:
            list of length 4, or None if metadata missing
        """
        ...

    @property
    def white_level(self) -> int:
        """Level at which the raw pixel value is considered to be saturated."""
        ...

    @property
    def num_colors(self) -> int:
        """Number of colors.

        Note that e.g. for RGBG this can be 3 or 4, depending on the camera model,
        as some use two different greens.
        """
        ...

    @property
    def color_matrix(self) -> np.ndarray:
        """Color matrix, read from file for some cameras, calculated for others.

        Returns:
            ndarray of shape (3,4)
        """
        ...

    @property
    def rgb_xyz_matrix(self) -> np.ndarray:
        """Camera RGB - XYZ conversion matrix.

        This matrix is constant (different for different models).
        Last row is zero for RGB cameras and non-zero for different color models (CMYG and so on).

        Returns:
            ndarray of shape (4,3)
        """
        ...

    @property
    def tone_curve(self) -> np.ndarray:
        """Camera tone curve, read from file for Nikon, Sony and some other cameras.

        Returns:
            ndarray of length 65536
        """
        ...

    @property
    def raw_image(self) -> np.ndarray:
        """View of RAW image. Includes margin.

        For Bayer images, a 2D ndarray is returned.
        For Foveon and other RGB-type images, a 3D ndarray is returned.
        Note that there may be 4 color channels, where the 4th channel can be blank (zeros).

        Modifying the returned array directly influences the result of calling postprocess()

        Warning:
            The returned numpy array can only be accessed while this RawPy instance is not closed yet,
            that is, within a with block or before calling close().
            If you need to work on the array after closing the RawPy instance,
            make sure to create a copy of it with raw_image = raw.raw_image.copy().

        Returns:
            ndarray of shape (h,w[,c])
        """
        ...

    @property
    def raw_image_visible(self) -> np.ndarray:
        """Like raw_image but without margin.

        Returns:
            ndarray of shape (hv,wv[,c])
        """
        ...

    @property
    def sizes(self) -> ImageSizes:
        """Return a rawpy.ImageSizes instance with size information of the RAW image and postprocessed image."""
        ...

    @property
    def camera_whitebalance(self) -> list[float]:
        """White balance coefficients (as shot). Either read from file or calculated.

        Returns:
            list of length 4
        """
        ...

    @property
    def daylight_whitebalance(self) -> list[float]:
        """White balance coefficients for daylight (daylight balance).

        Either read from file, or calculated on the basis of file data,
        or taken from hardcoded constants.

        Returns:
            list of length 4
        """
        ...

class LibRawError(Exception):
    """Base class for all LibRaw errors."""
    ...

class LibRawFatalError(LibRawError):
    """Exception for a fatal error in LibRaw."""
    ...

class LibRawNonFatalError(LibRawError):
    """Exception for a non-fatal error in LibRaw."""
    ...

class LibRawNoThumbnailError(LibRawNonFatalError):
    """Exception for when no thumbnail is available."""
    ...

class LibRawUnsupportedThumbnailError(LibRawNonFatalError):
    """Exception for when a thumbnail has an unsupported format."""
    ...

class NotSupportedError(Exception):
    """Exception for when a feature is not supported by the LibRaw version being used."""
    ...

def imread(pathOrFile: str | BinaryIO) -> RawPy:
    """Convenience function that creates a RawPy instance and opens the given file.

    Returns a RawPy instance for further processing.

    Parameters:
        pathOrFile: path or file object of RAW image that will be read

    Returns:
        RawPy instance
    """
    ...

# Module level variables
libraw_version: tuple[int, int, int]
"""Version of the underlying LibRaw library as a tuple (major, minor, patch)."""

flags: dict[str, bool]
"""Dictionary of LibRaw compile flags indicating which optional features are available."""

# Module enhancement.py
class enhance:
    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def save_dcraw_bad_pixels(path: str, bad_pixels: np.ndarray) -> None:
        """Save the given bad pixel coordinates in dcraw file format.

        Note that timestamps cannot be set and will be written as zero.

        Parameters:
            path: path of the badpixels file that will be written
            bad_pixels: array of shape (n,2) with y,x coordinates relative to visible image area
        """
        ...
