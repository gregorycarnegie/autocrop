from enum import auto, Enum, unique


@unique
class Terminator(Enum):
    END_FOLDER_TASK = auto()
    END_MAPPING_TASK = auto()
    END_VIDEO_TASK = auto()
