from enum import auto, Enum, unique


@unique
class FunctionTabSelectionState(Enum):
    """
    Enumeration class for function tab selection states.

    Function tab selection states include:
    - SELECTED: Represents a selected state.
    - NOT_SELECTED: Represents a not selected state.
    """

    SELECTED = auto()
    NOT_SELECTED = auto()
