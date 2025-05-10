from enum import Enum, auto, unique


@unique
class ButtonType(Enum):
    PROCESS_BUTTON = auto()
    NAVIGATION_BUTTON = auto()
