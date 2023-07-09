from enum import auto, Enum, unique

@unique
class LineEditState(Enum):
    VALID_INPUT = auto()
    INVALID_INPUT = auto()
