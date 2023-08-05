from enum import auto, Enum, unique


@unique
class FunctionType(Enum):
    PHOTO = auto()
    FOLDER = auto()
    MAPPING = auto()
    VIDEO = auto()
