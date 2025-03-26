from enum import auto, Enum, unique


@unique
class ButtonType(Enum):
    PROCESS_BUTTON = auto()
    NAVIGATION_BUTTON = auto()
