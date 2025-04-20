from enum import StrEnum
from typing import Final

from core.resource_path import ResourcePath

RESOURCES_ROOT: Final = ResourcePath("resources")


def get_icon_path(
    name: str,
    *,
    folder: str = "icons",
    ext: str = ".svg",
    mm: bool = False,
) -> str:
    """
    Build the path to an icon *inside* the project and return the on‑disk
    location (handles PyInstaller transparently).

    Parameters
    ----------
    name : str      Base filename, without extension.
    folder : str    Sub‑folder under ``resources``.  Default ``icons``.
    ext : str       File‑name extension, including the dot.
    mm : bool       Prepend ``multimedia_`` to `name`.
    """
    filename = f"multimedia_{name}{ext}" if mm else f"{name}{ext}"
    rel_path = RESOURCES_ROOT / folder / filename
    return rel_path.as_resource().as_posix()


class GuiIcon(StrEnum):
    CLAPPERBOARD = get_icon_path('clapperboard')
    CLEAR = get_icon_path('clear')
    CANCEL = get_icon_path('cancel')
    CROP = get_icon_path('crop')
    EXCEL = get_icon_path('excel')
    FOLDER = get_icon_path('folder')
    ICON = get_icon_path('logo', folder='logos', ext='.ico')
    LOGO = get_icon_path('logo', folder='logos')
    MULTIMEDIA_MUTE = get_icon_path('mute', mm=True)
    MULTIMEDIA_PAUSE = get_icon_path('pause', mm=True)
    MULTIMEDIA_PLAY = get_icon_path('play', mm=True)
    MULTIMEDIA_UNMUTE = get_icon_path('unmute', mm=True)
    MULTIMEDIA_STOP = get_icon_path('stop', mm=True)
    MULTIMEDIA_LEFT = get_icon_path('left', mm=True)
    MULTIMEDIA_RIGHT = get_icon_path('right', mm=True)
    MULTIMEDIA_REWIND = get_icon_path('rewind', mm=True)
    MULTIMEDIA_FASTFWD = get_icon_path('fastfwd', mm=True)
    MULTIMEDIA_BEGINNING = get_icon_path('beginning', mm=True)
    MULTIMEDIA_END = get_icon_path('end', mm=True)
    MULTIMEDIA_LEFTMARKER = get_icon_path('leftmarker', mm=True)
    MULTIMEDIA_RIGHTMARKER = get_icon_path('rightmarker', mm=True)
    MULTIMEDIA_CROPVIDEO = get_icon_path('crop_video')
    MULTIMEDIA_LABEL_A = get_icon_path('marker_label_a')
    MULTIMEDIA_LABEL_B = get_icon_path('marker_label_b')
    PICTURE = get_icon_path('picture')
