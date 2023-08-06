from enum import auto, Enum, unique


@unique
class PathType(Enum):
    IMAGE = auto()
    TABLE = auto()
    VIDEO = auto()
    FOLDER = auto()
