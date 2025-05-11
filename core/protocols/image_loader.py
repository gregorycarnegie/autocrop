from collections.abc import Callable

import cv2
import cv2.typing as cvt
import rawpy


class ImageLoader:
    @staticmethod
    def loader(loader_type: str) -> Callable[..., cvt.MatLike]:
        loaders = {
            'standard': cv2.imread,
            'raw': rawpy.imread,
            # Add other types as needed
        }
        return loaders.get(loader_type, cv2.imread)
