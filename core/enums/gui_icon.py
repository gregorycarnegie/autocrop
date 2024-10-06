from enum import StrEnum

from core.resource_path import ResourcePath


class GuiIcon(StrEnum):
    CLAPPERBOARD = ResourcePath('resources\\icons\\clapperboard.svg').meipass_path
    CLEAR = ResourcePath('resources\\icons\\clear.svg').meipass_path
    CANCEL = ResourcePath('resources\\icons\\cancel.svg').meipass_path
    CROP = ResourcePath('resources\\icons\\crop.svg').meipass_path
    EXCEL = ResourcePath('resources\\icons\\excel.svg').meipass_path
    FOLDER = ResourcePath('resources\\icons\\folder.svg').meipass_path
    ICON = ResourcePath('resources\\logos\\logo.ico').meipass_path
    LOGO = ResourcePath('resources\\logos\\logo.svg').meipass_path
    MULTIMEDIA_MUTE = ResourcePath('resources\\icons\\multimedia_mute.svg').meipass_path
    MULTIMEDIA_PAUSE = ResourcePath('resources\\icons\\multimedia_pause.svg').meipass_path
    MULTIMEDIA_PLAY = ResourcePath('resources\\icons\\multimedia_play.svg').meipass_path
    MULTIMEDIA_UNMUTE = ResourcePath('resources\\icons\\multimedia_unmute.svg').meipass_path
    MULTIMEDIA_STOP = ResourcePath('resources\\icons\\multimedia_stop.svg').meipass_path
    MULTIMEDIA_LEFT = ResourcePath('resources\\icons\\multimedia_left.svg').meipass_path
    MULTIMEDIA_RIGHT = ResourcePath('resources\\icons\\multimedia_right.svg').meipass_path
    MULTIMEDIA_REWIND = ResourcePath('resources\\icons\\multimedia_rewind.svg').meipass_path
    MULTIMEDIA_FASTFWD = ResourcePath('resources\\icons\\multimedia_fastfwd.svg').meipass_path
    MULTIMEDIA_BEGINING = ResourcePath('resources\\icons\\multimedia_begining.svg').meipass_path
    MULTIMEDIA_END = ResourcePath('resources\\icons\\multimedia_end.svg').meipass_path
    MULTIMEDIA_LEFTMARKER = ResourcePath('resources\\icons\\multimedia_leftmarker.svg').meipass_path
    MULTIMEDIA_RIGHTMARKER = ResourcePath('resources\\icons\\multimedia_rightmarker.svg').meipass_path
    MULTIMEDIA_CROPVIDEO = ResourcePath('resources\\icons\\crop_video.svg').meipass_path
    MULTIMEDIA_LABEL_A = ResourcePath('resources\\icons\\marker_label_a.svg').meipass_path
    MULTIMEDIA_LABEL_B = ResourcePath('resources\\icons\\marker_label_b.svg').meipass_path
    PICTURE = ResourcePath('resources\\icons\\picture.svg').meipass_path
