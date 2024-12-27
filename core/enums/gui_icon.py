from enum import StrEnum

from core.resource_path import ResourcePath

def get_icon_path(icon_name: str, folder:str = 'icons', extention: str = '.svg', mm: bool = False) -> str:
    icon_path = f'resources/{folder}/'
    icon_path += f'multimedia_{icon_name}{extention}' if mm else f'{icon_name}{extention}'
    return ResourcePath(icon_path).meipass_path

class GuiIcon(StrEnum):
    CLAPPERBOARD = get_icon_path('clapperboard')
    CLEAR = get_icon_path('clear')
    CANCEL = get_icon_path('cancel')
    CROP = get_icon_path('crop')
    EXCEL = get_icon_path('excel')
    FOLDER = get_icon_path('folder')
    ICON = get_icon_path('logo', folder='logos', extention='.ico')
    LOGO = get_icon_path('logo', folder='logos')
    MULTIMEDIA_MUTE = get_icon_path('mute', mm=True)
    MULTIMEDIA_PAUSE = get_icon_path('pause')
    MULTIMEDIA_PLAY = get_icon_path('play', mm=True)
    MULTIMEDIA_UNMUTE = get_icon_path('unmute', mm=True)
    MULTIMEDIA_STOP = get_icon_path('stop', mm=True)
    MULTIMEDIA_LEFT = get_icon_path('left', mm=True)
    MULTIMEDIA_RIGHT = get_icon_path('right', mm=True)
    MULTIMEDIA_REWIND = get_icon_path('rewind', mm=True)
    MULTIMEDIA_FASTFWD = get_icon_path('fastfwd', mm=True)
    MULTIMEDIA_BEGINING = get_icon_path('begining', mm=True)
    MULTIMEDIA_END = get_icon_path('end', mm=True)
    MULTIMEDIA_LEFTMARKER = get_icon_path('leftmarker', mm=True)
    MULTIMEDIA_RIGHTMARKER = get_icon_path('rightmarker', mm=True)
    MULTIMEDIA_CROPVIDEO = get_icon_path('crop_video')
    MULTIMEDIA_LABEL_A = get_icon_path('marker_label_a')
    MULTIMEDIA_LABEL_B = get_icon_path('marker_label_b')
    PICTURE = get_icon_path('picture')
