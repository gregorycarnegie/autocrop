from collections.abc import Callable

import cv2
import rawpy


class ImageLoader:
    @staticmethod
    def loader(loader_type: str) -> Callable[..., cv2.Mat]:
        loaders = {
            'standard': cv2.imread,
            'raw': rawpy.imread,
            # Add other types as needed
        }
        return loaders.get(loader_type, cv2.imread)
