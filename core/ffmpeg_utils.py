import os
import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication, QMessageBox

from core.config import logger


def configure_ffmpeg():
    """Configure ffmpeg-python to use the bundled FFmpeg"""
    import ffmpeg

    try:
        ffmpeg_path = get_ffmpeg_path()
        ffprobe_path = get_ffprobe_path()

        # Test if the executables actually exist and are accessible
        if os.path.isfile(ffmpeg_path) and os.access(ffmpeg_path, os.X_OK):
            ffmpeg._run.DEFAULT_EXE = ffmpeg_path
            logger.debug(f"Successfully configured FFmpeg: {ffmpeg_path}")
        else:
            raise FileNotFoundError(f"FFmpeg not found or not executable at: {ffmpeg_path}")

        if os.path.isfile(ffprobe_path) and os.access(ffprobe_path, os.X_OK):
            ffmpeg._probe.DEFAULT_EXE = ffprobe_path
            logger.debug(f"Successfully configured FFprobe: {ffprobe_path}")
        else:
            raise FileNotFoundError(f"FFprobe not found or not executable at: {ffprobe_path}")

        # Test FFmpeg version instead of probing the executable itself
        try:
            # Use a simple command to test if FFmpeg is working
            stream = ffmpeg.input('nullsrc=s=320x240:d=1', f='lavfi')
            stream = ffmpeg.output(stream, '-', format='null')
            ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
            logger.debug("FFmpeg test successful!")
        except Exception as e:
            logger.exception(f"FFmpeg test warning: {e}")
            # This is just a test, don't fail the whole configuration

    except Exception as e:
        # Show error to user
        error_msg = f"FFmpeg configuration failed: {str(e)}\n\nVideo features will not be available."
        logger.exception(error_msg)

        # Show this in a GUI dialog if we have a QApplication
        if QApplication.instance():
            QMessageBox.warning(None, "FFmpeg Error", error_msg)

def get_ffmpeg_path() -> str:
    """Get the path to FFmpeg executable"""
    import sys
    from pathlib import Path

    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        base_path = Path(sys._MEIPASS2)
        ffmpeg_path = base_path / 'ffmpeg' / 'ffmpeg.exe'
    else:
        # Running as script - use system FFmpeg
        ffmpeg_path = Path('C:/ffmpeg/bin/ffmpeg.exe')
        if not ffmpeg_path.exists():
            # Fallback to system PATH
            import shutil
            ffmpeg_path = shutil.which('ffmpeg')

    return str(ffmpeg_path) if ffmpeg_path else 'ffmpeg'

def get_ffprobe_path() -> str:
    """Get the path to FFprobe executable"""

    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        base_path = Path(sys._MEIPASS2)
        ffprobe_path = base_path / 'ffmpeg' / 'ffprobe.exe'
    else:
        # Running as script - use system FFmpeg
        ffprobe_path = Path('C:/ffmpeg/bin/ffprobe.exe')
        if not ffprobe_path.exists():
            # Fallback to system PATH
            import shutil
            ffprobe_path = shutil.which('ffprobe')

    return str(ffprobe_path) if ffprobe_path else 'ffprobe'
