from enum import Enum


class Preset(Enum):
    GOLDEN_RATIO = 0.5 * (1 + 5 ** 0.5)
    SQUARE = 1
    TWO_THIRDS = 1.5
    THREE_QUARTERS = 4 / 3
    FOUR_FIFTHS = 1.25
