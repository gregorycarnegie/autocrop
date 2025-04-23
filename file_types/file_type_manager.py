"""
Simplified file type handling system using Python's standard libraries.
"""

import contextlib
import mimetypes
from enum import auto, Flag
from pathlib import Path
from typing import Optional

# Initialize mimetypes database
mimetypes.init()


class FileCategory(Flag):
    """Enumeration of file categories for application."""
    PHOTO = auto()
    RAW = auto()
    TIFF = auto()
    VIDEO = auto()
    TABLE = auto()
    UNKNOWN = auto()


class FileTypeManager:
    """
    A modernized file type manager that handles file type detection,
    validation, and common operations.
    """
    # Extensions by category - more maintainable as a single source of truth
    _EXTENSIONS: dict[FileCategory, set[str]] = {
        FileCategory.PHOTO: {
            '.bmp', '.dib', '.jfif', '.jpeg',
            '.jpg', '.jpe', '.jp2', '.png', 
            '.webp', '.pbm', '.pgm', '.ppm',
            '.pxm', '.pnm', '.pfm', '.sr', 
            '.ras', '.exr', '.hdr', '.pic'
        },
        FileCategory.TIFF: {'.tiff', '.tif'},
        FileCategory.RAW: {
            '.dng', '.arw', '.cr2', '.crw',
            '.erf', '.kdc', '.nef', '.nrw',
            '.orf', '.pef', '.raf', '.raw',
            '.sr2', '.srw', '.x3f', '.exr'
        },
        FileCategory.VIDEO: {'.avi', '.m4v', '.mkv', '.mov', '.mp4'},
        FileCategory.TABLE: {'.csv', '.xlsx', '.xlsm', '.xltx', '.xltm', '.parquet'},
    }

    # Default directories for each category
    _DEFAULT_DIRS: dict[FileCategory, Path] = {
        FileCategory.PHOTO: Path.home() / 'Pictures',
        FileCategory.TIFF: Path.home() / 'Pictures',
        FileCategory.RAW: Path.home() / 'Pictures',
        FileCategory.VIDEO: Path.home() / 'Videos',
        FileCategory.TABLE: Path.home() / 'Documents',
    }
    
    # Which categories can be saved in which formats
    _SAVE_FORMATS: dict[FileCategory, tuple[str, ...]] = {
        FileCategory.PHOTO: ('.bmp', '.jpg', '.png', '.tiff', '.webp'),
        FileCategory.TIFF: ('.tiff',),
        FileCategory.RAW: ('.jpg', '.png', '.tiff'),  # RAW files are converted on save
        FileCategory.VIDEO: ('.jpg', '.png', '.tiff', '.webp'),  # Video frames
        FileCategory.TABLE: (),  # Tables aren't saved in this application
    }
    
    # Common mime types for faster lookup
    _MIME_TYPES: dict[str, FileCategory] = {
        'image/bmp'                       : FileCategory.PHOTO,     # .bmp .dib
        'image/jpeg'                      : FileCategory.PHOTO,     # .jpg .jpeg .jfif .jpe
        'image/jp2'                       : FileCategory.PHOTO,     # .jp2  (JPEG-2000)      • alias: image/jpeg2000
        'image/png'                       : FileCategory.PHOTO,     # .png
        'image/webp'                      : FileCategory.PHOTO,     # .webp
        'image/x-portable-bitmap'         : FileCategory.PHOTO,     # .pbm
        'image/x-portable-graymap'        : FileCategory.PHOTO,     # .pgm
        'image/x-portable-pixmap'         : FileCategory.PHOTO,     # .ppm
        'image/x-pam'                     : FileCategory.PHOTO,     # .pnm .pxm (P7 “any-map”)
        'image/x-portable-floatmap'       : FileCategory.PHOTO,     # .pfm
        'image/x-sun-raster'              : FileCategory.PHOTO,     # .sr .ras
        'image/exr'                       : FileCategory.PHOTO,     # .exr  (OpenEXR)        • alias: image/x-exr
        'image/vnd.radiance'              : FileCategory.PHOTO,     # .hdr .pic (Radiance)   • alias: image/x-radiance
        'image/tiff'                      : FileCategory.TIFF,      # .tiff .tif
        'image/x-adobe-dng'               : FileCategory.RAW,       # .dng
        'image/x-sony-arw'                : FileCategory.RAW,       # .arw
        'image/x-canon-cr2'               : FileCategory.RAW,       # .cr2
        'image/x-canon-crw'               : FileCategory.RAW,       # .crw
        'image/x-epson-erf'               : FileCategory.RAW,       # .erf
        'image/x-kodak-kdc'               : FileCategory.RAW,       # .kdc
        'image/x-nikon-nef'               : FileCategory.RAW,       # .nef
        'image/x-nikon-nrw'               : FileCategory.RAW,       # .nrw
        'image/x-olympus-orf'             : FileCategory.RAW,       # .orf
        'image/x-pentax-pef'              : FileCategory.RAW,       # .pef
        'image/x-fujifilm-raf'            : FileCategory.RAW,       # .raf
        'image/x-panasonic-raw'           : FileCategory.RAW,       # .raw
        'image/x-sony-sr2'                : FileCategory.RAW,       # .sr2
        'image/x-samsung-srw'             : FileCategory.RAW,       # .srw
        'image/x-sigma-x3f'               : FileCategory.RAW,       # .x3f
        'video/x-msvideo'                 : FileCategory.VIDEO,     # .avi
        'video/x-m4v'                     : FileCategory.VIDEO,     # .m4v
        'video/x-matroska'                : FileCategory.VIDEO,     # .mkv
        'video/quicktime'                 : FileCategory.VIDEO,     # .mov
        'video/mp4'                       : FileCategory.VIDEO,     # .mp4
        'text/csv'                                              : FileCategory.TABLE,  # .csv
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                                                                : FileCategory.TABLE, # .xlsx
        'application/vnd.ms-excel.sheet.macroEnabled.12'        : FileCategory.TABLE, # .xlsm
        'application/vnd.openxmlformats-officedocument.spreadsheetml.template'
                                                                : FileCategory.TABLE, # .xltx
        'application/vnd.ms-excel.template.macroEnabled.12'     : FileCategory.TABLE, # .xltm
        'application/vnd.apache.parquet'                        : FileCategory.TABLE, # .parquet  (official)
        'application/x-parquet'                                 : FileCategory.TABLE, # legacy alias
    }
    
    @classmethod
    def get_extensions(cls, category: FileCategory) -> set[str]:
        """Get all file extensions for a given category."""
        return cls._EXTENSIONS.get(category, set())
    
    @classmethod
    def get_all_extensions(cls) -> set[str]:
        """Get all recognized file extensions."""
        all_extensions = set()
        for extensions in cls._EXTENSIONS.values():
            all_extensions.update(extensions)
        return all_extensions
    
    @classmethod
    def get_default_directory(cls, category: FileCategory) -> Path:
        """Get the default directory for a given file category."""
        return cls._DEFAULT_DIRS.get(category, Path.home())
    
    @classmethod
    def get_save_formats(cls, category: FileCategory) -> tuple[str, ...]:
        """Get the valid save formats for a given file category."""
        return cls._SAVE_FORMATS.get(category, ())
    
    @classmethod
    def detect_category(cls, path: Path) -> FileCategory:
        """
        Detect the file category based on extension and content.
        Uses a combination of extension checks and mimetype detection.
        """
        if not path.exists() or not path.is_file():
            return FileCategory.UNKNOWN

        # Check by extension first (fast path)
        suffix = path.suffix.lower()
        for category, extensions in cls._EXTENSIONS.items():
            if suffix in extensions:
                return category

        # Try mimetype detection as fallback
        with contextlib.suppress(Exception):
            mime_type, _ = mimetypes.guess_type(path)
            if mime_type in cls._MIME_TYPES:
                return cls._MIME_TYPES[mime_type]
        return FileCategory.UNKNOWN
    
    @classmethod
    def is_valid_type(cls, path: Path, category: FileCategory) -> bool:
        """Check if a file is of the specified category."""
        if not path.is_file():
            return False
            
        # For performance, first check by extension
        if path.suffix.lower() in cls.get_extensions(category):
            return True
            
        # Only do deeper validation if extension check fails
        detected = cls.detect_category(path)
        return detected == category
    
    @classmethod
    def get_filter_string(cls, category: Optional[FileCategory] = None) -> str:
        """
        Generate a filter string for file dialogs.
        
        Args:
            category: Optional category to filter for
            
        Returns:
            A Qt-compatible filter string for file dialogs
        """
        # Generate a single category filter
        if category is not None:
            extensions = cls.get_extensions(category)
            if not extensions:
                return "All Files (*)"

            match category:
                case FileCategory.PHOTO | FileCategory.RAW | FileCategory.TIFF:
                    category_txt = "Image"
                case FileCategory.VIDEO:
                    category_txt = "Video"
                case FileCategory.TABLE:
                    category_txt = "Table"
                case _:
                    return "All Files (*)"

            filters = [f'*{ext}' for ext in extensions]
            return f"{category_txt} Files ({' '.join(filters)});;All Files (*)"
            # return f"All Files (*);;{filter_text}"

        # Generate all categories filter
        all_filters = ["All Files (*)"]
        for cat in [c for c in FileCategory if c != FileCategory.UNKNOWN]:
            if extensions := cls.get_extensions(cat):
                filters = [f'*{ext}' for ext in extensions]
                all_filters.append(f"{cat.value.title()} Files ({' '.join(filters)})")

        return ";;".join(all_filters)
    
    @classmethod
    def should_use_tiff_save(cls, path: Path) -> bool:
        """Determine if TIFF saving should be used based on file extension."""
        return path.suffix.lower() in cls.get_extensions(FileCategory.TIFF)


# Create a singleton instance for easier access
file_manager = FileTypeManager()
