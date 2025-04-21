"""
Utilities for file signature detection and validation.
"""
from pathlib import Path
from typing import Optional

from .file_type_manager import FileCategory, FileTypeManager


class SignatureChecker:
    """
    A utility class for detecting file types based on binary signatures/magic numbers.
    Only used when more accurate type detection is needed beyond extensions.
    """
    # Common file signatures as (signature bytes, offset)
    _SIGNATURES: dict[FileCategory, dict[str, list[tuple[bytes, int]]]] = {
        FileCategory.PHOTO: {
            # JPEG‑2000 JP2 signature box (12 bytes)
            '.jp2': [(b'\x00\x00\x00\x0C\x6A\x50\x20\x20\x0D\x0A\x87\x0A', 0)],

            # JPEG
            '.jpg': [(b'\xFF\xD8\xFF', 0)],
            '.jpeg': [(b'\xFF\xD8\xFF', 0)],
            '.jfif': [(b'\xFF\xD8\xFF', 0)],
            '.jpe': [(b'\xFF\xD8\xFF', 0)],
            
            # Netpbm family (ASCII vs. raw variants)
            '.pbm': [(b'P4', 0), (b'P1', 0)],   # bitmap  (raw / ASCII)
            '.pgm': [(b'P5', 0), (b'P2', 0)],   # greymap
            '.ppm': [(b'P6', 0), (b'P3', 0)],   # pixmap
            '.pnm': [(b'P7', 0)],               # PAM/PNM “portable any‑map”
            '.pxm': [(b'P7', 0)],               # alias of PNM super‑set
            
            # Portable FloatMap (32‑bit float HDR)
            '.pfm': [(b'PF', 0), (b'Pf', 0)],   # colour / greyscale
            
            # Sun Raster / SR files
            '.sr':  [(b'\x59\xA6\x6A\x95', 0)],
            '.ras': [(b'\x59\xA6\x6A\x95', 0)],

            # Radiance HDR / PIC: ASCII “#?RADIANCE” (occasionally “#?RGBE”)
            '.hdr': [(b'#?RADIANCE', 0), (b'#?RGBE', 0)],
            '.pic': [(b'#?RADIANCE', 0), (b'#?RGBE', 0)],

            # Other image formats
            '.bmp': [(b'BM', 0)],
            '.dib': [(b'BM', 0)],
            '.webp': [(b'RIFF', 0), (b'WEBP', 8)],
            '.png': [(b'\x89PNG\r\n\x1A\n', 0)],
        },
        FileCategory.TIFF: {
            '.tiff': [(b'\x49\x49\x2A\x00', 0), (b'\x4D\x4D\x00\x2A', 0)],
            '.tif': [(b'\x49\x49\x2A\x00', 0), (b'\x4D\x4D\x00\x2A', 0)],
        },
        FileCategory.RAW: {
            # Generic TIFF‑EP header (little‑ or big‑endian) for most vendor RAWs
            '.arw': [(b'\x49\x49\x2A\x00', 0)],
            '.nef': [(b'\x49\x49\x2A\x00', 0), (b'\x4D\x4D\x00\x2A', 0)],            
            '.erf': [(b'\x49\x49\x2A\x00', 0), (b'\x4D\x4D\x00\x2A', 0)],   # Epson
            '.kdc': [(b'\x49\x49\x2A\x00', 0), (b'\x4D\x4D\x00\x2A', 0)],   # Kodak
            '.nrw': [(b'\x49\x49\x2A\x00', 0)],                             # Nikon compact
            '.pef': [(b'\x49\x49\x2A\x00', 0)],                             # Pentax
            '.raw': [(b'\x49\x49\x2A\x00', 0), (b'\x4D\x4D\x00\x2A', 0)],   # Panasonic/Leica, etc.
            '.sr2': [(b'\x49\x49\x2A\x00', 0)],                             # Sony
            '.srw': [(b'\x49\x49\x2A\x00', 0)],                             # Samsung
            '.dng': [(b'\x49\x49\x2A\x00', 0), (b'\x49\x49\x00\x2A', 0)],
            '.exr': [(b'\x76\x2F\x31\x01', 0)],
            '.cr2': [(b'\x49\x49\x2A\x00\x10\x00\x00\x00\x43\x52', 0)],
            '.crw': [(b'\x49\x49\x1A\x00\x00\x00\x48\x45\x41\x50\x43\x43\x44\x52\x02\x00', 0)],
            
            # Fujifilm RAF: ASCII “FUJIFILMCCD-RAW” at byte 0
            '.raf': [(b'FUJIFILMCCD-RAW', 0)],
            
            # Sigma X3F: ASCII “FOVb” at byte 0
            '.x3f': [(b'FOVb', 0)],
            
            # Olympus puts a custom four‑byte magic in place of the TIFF bytes
            '.orf': [(b'IIRO', 0), (b'II', 0)],        # IIRO/II = little‑endian, MMOR big‑endian
        },
        FileCategory.VIDEO: {
            '.mp4': [(b'ftyp', 4)],
            '.m4v': [(b'ftyp', 4)],
            '.mov': [(b'ftyp', 4)],
            '.avi': [(b'RIFF', 0), (b'AVI ', 8)],
            '.mkv': [(b'\x1A\x45\xDF\xA3', 0)],
        },
        FileCategory.TABLE: {
            # All OOXML spreadsheets are ZIP archives
            '.xlsx': [(b'\x50\x4B\x03\x04', 0)],
            '.xlsm': [(b'\x50\x4B\x03\x04', 0)],
            '.xltx': [(b'\x50\x4B\x03\x04', 0)],
            '.xltm': [(b'\x50\x4B\x03\x04', 0)],
        },
    }
    
    # Extensions without reliable signatures (text-based formats)
    _TEXT_EXTENSIONS: set[str] = {'.csv'}
    
    @classmethod
    def check_signature(cls, file_path: Path, expected_category: Optional[FileCategory] = None) -> bool:
        """
        Check if a file has a valid signature corresponding to its extension or expected category.
        
        Args:
            file_path: Path to the file to check
            expected_category: Optional expected file category to validate against
            
        Returns:
            bool: True if the file signature matches the expected type
        """
        if not file_path.exists() or not file_path.is_file():
            return False

        # Get file extension
        extension = file_path.suffix.lower()

        # Special case for text-based formats
        if extension in cls._TEXT_EXTENSIONS:
            return cls._validate_text_file(file_path)

        # Determine category to check
        category = expected_category
        if category is None:
            category = FileTypeManager.detect_category(file_path)

        if category == FileCategory.UNKNOWN:
            return False

        if signatures := cls._get_signatures(category, extension):
            # Check a file against known signatures
            return cls._check_file_signatures(file_path, signatures)
        else:
            # No signatures defined, can't validate beyond extension
            return True
    
    @classmethod
    def _get_signatures(cls, category: FileCategory, extension: str) -> list[tuple[bytes, int]]:
        """Get signatures for a given category and extension."""
        if category not in cls._SIGNATURES:
            return []
            
        category_signatures = cls._SIGNATURES[category]
        return category_signatures.get(extension, [])
    
    @staticmethod
    def _validate_text_file(file_path: Path) -> bool:
        """
        Simple validation for text-based files like CSV.
        Check that the file is readable as text.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Try to read a small amount to check if it's text
                f.read(1024)
            return True
        except UnicodeDecodeError:
            # Not a valid text file
            return False
    
    @staticmethod
    def _check_file_signatures(file_path: Path, signatures: list[tuple[bytes, int]]) -> bool:
        """Check if a file matches any of the provided signatures."""
        try:
            # Read enough bytes for signature checking
            max_offset = max(offset + len(sig) for sig, offset in signatures)
            with open(file_path, 'rb') as f:
                header = f.read(max_offset)

            return any(
                len(header) >= offset + len(signature)
                and header[offset : offset + len(signature)] == signature
                for signature, offset in signatures
            )
        except OSError:
            return False
