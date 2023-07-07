from enum import auto, Enum, unique

@unique
class MediaPlaybackState(Enum):
    MUTED = auto()
    UNMUTED = auto()
