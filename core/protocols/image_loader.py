from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Literal, overload

import cv2
import cv2.typing as cvt
import rawpy


class ImageLoader:
    """
    Utility class to load images using different loaders.

    Usage:
        img = ImageLoader.loader('standard')('path/to/image.jpg')
        with ImageLoader.loader('raw')('path/to/image.raw') as raw:
    """

    @staticmethod
    @overload
    def loader(loader_type: Literal['standard']) -> Callable[[str], cvt.MatLike]: ...

    @staticmethod
    @overload
    def loader(loader_type: Literal['raw']) -> Callable[[str], AbstractContextManager]: ...

    @staticmethod
    def loader(
        loader_type: str = 'standard'
    ) -> Callable[[str], cvt.MatLike] | Callable[[str], AbstractContextManager]:
        """
        Return an image loader function based on the loader_type.

        Args:
            loader_type: 'standard' for cv2.imread or 'raw' for rawpy.imread.
        Returns:
            A callable that takes a file path and returns an image or context manager.
        """
        loaders = {
            'standard': cv2.imread,
            'raw': rawpy.imread,
        }
        # Default to standard loader if unknown type provided
        return loaders.get(loader_type, cv2.imread)
