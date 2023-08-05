from enum import auto, Enum, unique


@unique
class FunctionTabSelectionState(Enum):
    SELECTED = auto()
    NOT_SELECTED = auto()
