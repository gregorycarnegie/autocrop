from enum import Enum


class Preset(Enum):
    """
    Enumeration class for preset values.

    Preset values include:
    - GOLDEN_RATIO: The golden ratio value.
    - SQUARE: The square value.
    - TWO_THIRDS: Two-thirds value.
    - THREE_QUARTERS: Three-quarters value.
    - FOUR_FIFTHS: Four-fifths value.
    """

    GOLDEN_RATIO = 0.5 * (1 + 5 ** 0.5)
    SQUARE = 1
    TWO_THIRDS = 1.5
    THREE_QUARTERS = 4 / 3
    FOUR_FIFTHS = 1.25
