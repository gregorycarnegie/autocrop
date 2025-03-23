import sys
from pathlib import Path
from typing import Union, cast


class ResourcePath(Path):
    """
    A class that extends the Path class to support resource paths, especially useful
    for PyInstaller packaged applications.
    """

    def __new__(cls, *args: Union[str, Path], **kwargs) -> 'ResourcePath':
        """
        Create a new instance of ResourcePath.
        """
        instance = super().__new__(cls, *args, **kwargs)
        return cast(ResourcePath, instance)

    @property
    def meipass_path(self) -> str:
        """
        Return the correct resource path for PyInstaller packaged applications.

        PyInstaller uses a temporary folder named `_MEIPASS2` to store temporary files.
        This method checks if the `_MEIPASS2` attribute is present in the sys module
        and constructs the path accordingly.

        Returns:
            str: The resource path.
        """
        base_path = Path(getattr(sys, '_MEIPASS2', str(Path().resolve())))
        return (base_path / self).as_posix()
