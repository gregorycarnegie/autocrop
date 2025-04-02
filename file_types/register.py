from pathlib import Path

from .registry import FileRegistry, FileTypeInfo

def initialize_file_types():
    """
    Initialize all file types in the registry.
    This function should be called at application startup.
    """
    # Photo file types
    FileRegistry.register_type("photo", FileTypeInfo(
        extensions={
            '.bmp', '.dib', '.jfif', '.jpeg', '.jpg', '.jpe', '.jp2', '.png', '.webp', 
            '.pbm', '.pgm', '.ppm', '.pxm', '.pnm', '.pfm', '.sr', '.ras', '.exr', '.hdr', '.pic'
        },
        default_dir=Path.home() / 'Pictures',
        description="Standard image formats supported by OpenCV",
        mime_types={'image/jpeg', 'image/png', 'image/bmp', 'image/webp'}
    ))
    
    # TIFF files get special handling
    FileRegistry.register_type("tiff", FileTypeInfo(
        extensions={'.tiff', '.tif'},
        default_dir=Path.home() / 'Pictures',
        description="TIFF image formats",
        mime_types={'image/tiff'}
    ))
    
    # RAW photo file types
    FileRegistry.register_type("raw", FileTypeInfo(
        extensions={
            '.dng', '.arw', '.cr2', '.crw', '.erf', '.kdc', '.nef', 
            '.nrw', '.orf', '.pef', '.raf', '.raw', '.sr2', '.srw', '.x3f'
        },
        default_dir=Path.home() / 'Pictures',
        description="Camera RAW formats",
        can_save=False  # RAW files can only be read, not written
    ))
    
    # Video file types
    FileRegistry.register_type("video", FileTypeInfo(
        extensions={'.avi', '.m4v', '.mkv', '.mov', '.mp4'},
        default_dir=Path.home() / 'Videos',
        description="Video formats"
    ))
    
    # Table file types
    FileRegistry.register_type("table", FileTypeInfo(
        extensions={'.csv', '.xlsx', '.xlsm', '.xltx', '.xltm'},
        default_dir=Path.home() / 'Documents',
        description="Spreadsheet and tabular data formats"
    ))

# Initialize file types when module is imported
initialize_file_types()