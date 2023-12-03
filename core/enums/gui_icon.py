from enum import Enum
from pathlib import Path
import sys

def resource_path(relative_path: str) -> str:
    base_path = getattr(sys, '_MEIPASS2', Path().resolve())

    return (Path(base_path) / relative_path).as_posix()


class GuiIcon(Enum):
    CLAPPERBOARD = resource_path('resources\\icons\\clapperboard.svg')
    CANCEL = resource_path('resources\\icons\\cancel.svg')
    CROP = resource_path('resources\\icons\\crop.svg')
    EXCEL = resource_path('resources\\icons\\excel.svg')
    FOLDER = resource_path('resources\\icons\\folder.svg')
    LOGO = resource_path('resources\\logos\\logo.svg')
    MULTIMEDIA_MUTE = resource_path('resources\\icons\\multimedia_mute.svg')
    MULTIMEDIA_PAUSE = resource_path('resources\\icons\\multimedia_pause.svg')
    MULTIMEDIA_PLAY = resource_path('resources\\icons\\multimedia_play.svg')
    MULTIMEDIA_UNMUTE = resource_path('resources\\icons\\multimedia_unmute.svg')
    MULTIMEDIA_STOP = resource_path('resources\\icons\\multimedia_stop.svg')
    MULTIMEDIA_LEFT = resource_path('resources\\icons\\multimedia_left.svg')
    MULTIMEDIA_RIGHT = resource_path('resources\\icons\\multimedia_right.svg')
    MULTIMEDIA_REWIND = resource_path('resources\\icons\\multimedia_rewind.svg')
    MULTIMEDIA_FASTFWD = resource_path('resources\\icons\\multimedia_fastfwd.svg')
    MULTIMEDIA_BEGINING = resource_path('resources\\icons\\multimedia_begining.svg')
    MULTIMEDIA_END = resource_path('resources\\icons\\multimedia_end.svg')
    MULTIMEDIA_LEFTMARKER = resource_path('resources\\icons\\multimedia_leftmarker.svg')
    MULTIMEDIA_RIGHTMARKER = resource_path('resources\\icons\\multimedia_rightmarker.svg')
    MULTIMEDIA_CROPVIDEO = resource_path('resources\\icons\\crop_video.svg')
    MULTIMEDIA_LABEL_A = resource_path('resources\\icons\\marker_label_a.svg')
    MULTIMEDIA_LABEL_B = resource_path('resources\\icons\\marker_label_b.svg')
    PICTURE = resource_path('resources\\icons\\picture.svg')
