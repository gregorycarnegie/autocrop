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
        mime_types={'image/jpeg', 'image/png', 'image/bmp', 'image/webp'},
        headers={
            '.jpg': [(b'\xFF\xD8\xFF', 0)],
            '.jpeg': [(b'\xFF\xD8\xFF', 0)],
            '.jfif': [(b'\xFF\xD8\xFF', 0)],
            '.jpe': [(b'\xFF\xD8\xFF', 0)],
            '.png': [(b'\x89PNG\r\n\x1A\n', 0)],
            '.bmp': [(b'BM', 0)],
            '.dib': [(b'BM', 0)],
            '.webp': [(b'RIFF', 0), (b'WEBP', 8)],
            '.jp2': [(b'\x00\x00\x00\x0C\x6A\x50\x20\x20\x0D\x0A\x87\x0A', 0)],
            '.pbm': [(b'P1', 0), (b'P4', 0)],
            '.pgm': [(b'P2', 0), (b'P5', 0)],
            '.ppm': [(b'P3', 0), (b'P6', 0)],
            '.exr': [(b'\x76\x2F\x31\x01', 0)],
            '.hdr': [(b'#?RADIANCE', 0)],
            '.pnm': [(b'P1', 0), (b'P2', 0), (b'P3', 0), (b'P4', 0), (b'P5', 0), (b'P6', 0)],
            '.pxm': [(b'P1', 0), (b'P2', 0), (b'P3', 0), (b'P4', 0), (b'P5', 0), (b'P6', 0)],
            '.pfm': [(b'PF\n', 0), (b'Pf\n', 0)],
            '.sr': [(b'\x59\xA6\x6A\x95', 0)],
            '.ras': [(b'\x59\xA6\x6A\x95', 0)],
            '.pic': [(b'PICT', 0), (b'\x34\x12', 0), (b'\x8A\x67\x69\x50', 0)]
            # Add others as needed
        }
    ))
    
    # TIFF files get special handling
    FileRegistry.register_type("tiff", FileTypeInfo(
        extensions={'.tiff', '.tif'},
        default_dir=Path.home() / 'Pictures',
        description="TIFF image formats",
        mime_types={'image/tiff'},
        headers={
            '.tiff': [(b'\x49\x49\x2A\x00', 0), (b'\x4D\x4D\x00\x2A', 0)],
            '.tif': [(b'\x49\x49\x2A\x00', 0), (b'\x4D\x4D\x00\x2A', 0)],
        }
    ))
    
    # RAW photo file types
    FileRegistry.register_type("raw", FileTypeInfo(
        extensions={
            '.dng', '.arw', '.cr2', '.crw', '.erf', '.kdc', '.nef', 
            '.nrw', '.orf', '.pef', '.raf', '.raw', '.sr2', '.srw', '.x3f'
        },
        default_dir=Path.home() / 'Pictures',
        description="Camera RAW formats",
        headers={
            '.dng': [(b'\x49\x49\x2A\x00', 0)],
            '.cr2': [(b'\x49\x49\x2A\x00\x10\x00\x00\x00\x43\x52', 0)],
            '.arw': [(b'\x49\x49\x2A\x00', 0)],
            '.nef': [(b'\x49\x49\x2A\x00', 0)],
            '.crw': [(b'\x49\x49\x1A\x00\x00\x00\x48\x45\x41\x50', 0)],
            '.erf': [(b'\x49\x49\x2A\x00', 0)],
            '.kdc': [(b'\x45\x4B\x44\x43', 0)],
            '.orf': [(b'\x49\x49\x52\x4F\x08\x00\x00\x00\x00', 0), (b'\x49\x49\x2A\x00', 0)],
            '.pef': [(b'\x49\x49\x2A\x00', 0)],
            '.raf': [(b'FUJIFILMCCD-RAW', 0)],
            '.raw': [],  # Generic extension, needs specialized detection
            '.sr2': [(b'\x49\x49\x2A\x00', 0)],
            '.srw': [(b'\x49\x49\x2A\x00', 0)],
            '.x3f': [(b'FOVb', 0)]
            # Add more RAW formats as needed
        },
        can_save=False  # RAW files can only be read, not written
    ))
    
    # Video file types
    FileRegistry.register_type("video", FileTypeInfo(
        extensions={'.avi', '.m4v', '.mkv', '.mov', '.mp4'},
        default_dir=Path.home() / 'Videos',
        description="Video formats",
        headers={
            '.mp4': [(b'ftyp', 4)],
            '.m4v': [(b'ftyp', 4)],
            '.mov': [(b'ftyp', 4), (b'moov', 4), (b'mdat', 4)],
            '.avi': [(b'RIFF', 0), (b'AVI ', 8)],
            '.mkv': [(b'\x1A\x45\xDF\xA3', 0)],
        }
    ))
    
    # Table file types
    FileRegistry.register_type("table", FileTypeInfo(
        extensions={'.csv', '.xlsx', '.xlsm', '.xltx', '.xltm'},
        default_dir=Path.home() / 'Documents',
        description="Spreadsheet and tabular data formats",
        headers={
            '.csv': [], # Plain text, no reliable signature
            '.xlsx': [(b'PK\x03\x04', 0)], # XLSX is a ZIP file
            '.xlsm': [(b'PK\x03\x04', 0)],
            '.xltx': [(b'PK\x03\x04', 0)],
            '.xltm': [(b'PK\x03\x04', 0)],
        }
    ))

# Initialize file types when module is imported
initialize_file_types()