from enum import auto, Enum, unique


@unique
class LineEditState(Enum):
    """
    Enumeration class for line edit states.

    Line edit states include:
    - VALID_INPUT: Represents a valid input state.
    - INVALID_INPUT: Represents an invalid input state.
    """

    VALID_INPUT = auto()
    INVALID_INPUT = auto()
