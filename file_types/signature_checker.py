"""
Enhanced utilities for file signature detection and validation.
"""
import mimetypes
import os
import zipfile
from contextlib import suppress
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import cv2

from .file_category import FileCategory
from .file_type_manager import FileTypeManager

mimetypes.init()

class SignatureChecker:
    """
    A utility class for detecting file types based on binary signatures/magic numbers.
    Only used when more accurate type detection is needed beyond extensions.
    """
    # Common file signatures as (signature bytes, offset)
    _SIGNATURES: dict[FileCategory, dict[str, list[tuple[bytes, int]]]] = {
        FileCategory.PHOTO: {
            # JPEG‑2000 JP2 signature box (12 bytes)
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
            
            # Sun Raster / SR files
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
            
            # Olympus puts custom four‑byte magic in place of the TIFF bytes
            '.orf': [(b'IIRO', 0), (b'II', 0)],        # IIRO/II = little‑endian, MMOR big‑endian
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
            '.parquet': [(b'PAR1', 0), (b'PARE', 0)],
        },
    }
    
    # Extensions without reliable signatures (text-based formats)
    _TEXT_EXTENSIONS: set[str] = {'.csv'}
    
    # Common patterns to look for in file content
    _CONTENT_PATTERNS: Dict[FileCategory, Dict[str, List[bytes]]] = {
        FileCategory.PHOTO: {
            '.jpg': [b'JFIF', b'Exif'],
            '.png': [b'IHDR', b'IDAT', b'IEND'],
            '.bmp': [b'DIB'],
            '.webp': [b'VP8 '],
        },
        FileCategory.VIDEO: {
            '.mp4': [b'moov', b'mdat', b'mvhd'],
            '.avi': [b'LIST', b'movi', b'idx1'],
            '.mkv': [b'Segment', b'Tracks', b'Cluster'],
        },
    }
    
    @classmethod
    def verify_file_type(cls, file_path: Path, expected_category: FileCategory) -> bool:
        """
        Comprehensive file type verification using multiple methods.
        
        Args:
            file_path: Path to the file to verify
            expected_category: Expected file category
            
        Returns:
            bool: True if the file content matches the expected category
        """
        if not file_path.exists() or not file_path.is_file():
            return False

        # Perform three verification methods
        checks = [
            # Method 1: Signature check
            cls.check_signature(file_path, expected_category),

            # Method 2: Content pattern search
            cls._check_content_patterns(file_path, expected_category),

            # Method 3: Category-specific verification
            cls._verify_file_content(file_path, expected_category)
        ]

        # File passes if at least 2 out of 3 checks succeed (or for text files, just 1)
        extension = file_path.suffix.lower()
        return any(checks) if extension in cls._TEXT_EXTENSIONS else sum(checks) >= 2

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

        # Determine the category to check
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
    def _check_content_patterns(cls, file_path: Path, expected_category: FileCategory) -> bool:
        """
        Search for content patterns that indicate the file type.
        
        Args:
            file_path: Path to the file
            expected_category: Expected file category
            
        Returns:
            bool: True if content patterns for the category are found
        """
        # Skip for categories without defined patterns
        if expected_category not in cls._CONTENT_PATTERNS:
            return False
            
        extension = file_path.suffix.lower()
        
        # Skip if no patterns for this extension
        if extension not in cls._CONTENT_PATTERNS[expected_category]:
            # For extensions without specific patterns, default to True
            # since we'll rely on other verification methods
            return True
            
        # Get patterns to search for
        patterns = cls._CONTENT_PATTERNS[expected_category][extension]
        
        try:
            # Read a chunk of the file (skip header, look into body)
            with open(file_path, 'rb') as f:
                # Skip the first 512 bytes (headers) to look for content patterns
                f.seek(512, os.SEEK_SET)
                # Read up to 8KB of content
                content = f.read(8192)
                
            # Check if any pattern is present in the content
            return any(pattern in content for pattern in patterns)
        except Exception:
            return False

    @classmethod
    def _verify_file_content(cls, file_path: Path, expected_category: FileCategory) -> bool:
        """
        Verify file content using OpenCV and other techniques.
        
        Args:
            file_path: Path to the file to check
            expected_category: Expected file category
            
        Returns:
            bool: True if content analysis confirms the expected category
        """
        try:
            # Match on file category
            if expected_category == FileCategory.PHOTO:
                return cls._verify_image_content(file_path)
            elif expected_category == FileCategory.TIFF:
                return cls._verify_tiff_content(file_path)
            elif expected_category == FileCategory.RAW:
                return cls._verify_raw_content(file_path)
            elif expected_category == FileCategory.VIDEO:
                return cls._verify_video_content(file_path)
            elif expected_category == FileCategory.TABLE:
                return cls._verify_table_content(file_path)
            else:
                return False
        except Exception:
            # If verification fails, return False
            return False
    
    @classmethod
    def _verify_image_content(cls, file_path: Path) -> bool:
        """
        Verify image content by attempting to load it with OpenCV.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            bool: True if the file can be loaded as an image
        """
        try:
            # Try OpenCV
            img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
            return img is not None and img.size > 0
        except cv2.error:
            return False
    
    @classmethod
    def _verify_tiff_content(cls, file_path: Path) -> bool:
        """
        Verify TIFF content by checking its structure.
        
        Args:
            file_path: Path to the TIFF file
            
        Returns:
            bool: True if the file has a valid TIFF structure
        """
        with suppress(cv2.error):
            # Check if OpenCV can load it as a TIFF
            img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
            if img is not None and img.size > 0:
                return True
        # Fallback to signature checking
        return cls._check_file_signatures(file_path, cls._SIGNATURES[FileCategory.TIFF]['.tiff'])
    
    @classmethod
    def _verify_raw_content(cls, file_path: Path) -> bool:
        """
        Verify RAW image content.
        
        Args:
            file_path: Path to the RAW file
            
        Returns:
            bool: True if the file has a valid RAW structure
        """
        # RAW verification is challenging without specialized libraries
        # Fallback to signature checking
        extension = file_path.suffix.lower()
        if extension in cls._SIGNATURES[FileCategory.RAW]:
            return cls._check_file_signatures(file_path, cls._SIGNATURES[FileCategory.RAW][extension])
        return False
    
    @classmethod
    def _verify_video_content(cls, file_path: Path) -> bool:
        """
        Verify video content by checking its structure with OpenCV.
        
        Args:
            file_path: Path to the video file
            
        Returns:
            bool: True if the file has a valid video structure
        """
        try:
            # Try to open with OpenCV's VideoCapture
            cap = cv2.VideoCapture(str(file_path))
            if not cap.isOpened():
                return False
                
            # Check if we can read frames
            ret, frame = cap.read()
            cap.release()
            
            return ret and frame is not None
        except cv2.error:
            return False
    
    @classmethod
    def _verify_table_content(cls, file_path: Path) -> bool:
        """
        Verify table file content based on extension.
        
        Args:
            file_path: Path to the table file
            
        Returns:
            bool: True if the file has a valid table structure
        """
        extension = file_path.suffix.lower()
        
        # CSV files
        if extension == '.csv':
            return cls._validate_text_file(file_path)
            
        # Excel files (check for ZIP structure and specific content)
        if extension in ('.xlsx', '.xlsm', '.xltx', '.xltm'):
            try:
                # Check for ZIP signature
                if not cls._check_file_signatures(file_path, [
                    (b'\x50\x4B\x03\x04', 0)  # ZIP signature
                ]):
                    return False
                    
                with zipfile.ZipFile(file_path) as zf:
                    # Excel files should contain these entries
                    required_entries = [
                        'xl/workbook.xml',
                        '[Content_Types].xml'
                    ]
                    
                    file_list = zf.namelist()
                    return any(entry in file_list for entry in required_entries)
            except Exception:
                return False
                
        # Parquet files
        if extension == '.parquet':
            # Check for PAR1 signature
            return cls._check_file_signatures(file_path, [
                (b'PAR1', 0)  # Parquet signature
            ])
            
        return False

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
        Enhanced validation for text-based files like CSV.
        Check that the file is readable as text and has the expected structure.
        """
        try:
            # Check if the file is text-based by attempting to read a portion
            with open(file_path, 'r', encoding='utf-8') as f:
                # Read the beginning of the file
                sample = f.read(4096)
                
                # Check for basic CSV structure (commas or tabs)
                lines = sample.splitlines()
                if len(lines) > 1:
                    # Check if the file has consistent delimiters
                    return ',' in lines[0] or '\t' in lines[0]
                    
            return True
        except UnicodeDecodeError:
            # Try with different encoding if the initial attempt fails
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    sample = f.read(4096)
                    lines = sample.splitlines()
                    if len(lines) > 1:
                        return ',' in lines[0] or '\t' in lines[0]
                return True
            except Exception:
                return False
        except Exception:
            return False
    
    @staticmethod
    def _check_file_signatures(file_path: Path, signatures: List[Tuple[bytes, int]]) -> bool:
        """
        Enhanced check if a file matches any of the provided signatures.
        Reads only the necessary bytes for each signature.
        """
        try:
            # Find the maximum offset and length needed
            max_offset_length = max(
                offset + len(sig) for sig, offset in signatures
            )
            
            # Open the file in binary mode
            with open(file_path, 'rb') as f:
                # Read enough bytes for all signatures
                header = f.read(max_offset_length)
                
                # Check each signature
                for signature, offset in signatures:
                    # Skip signatures that would read beyond the file length
                    if len(header) < offset + len(signature):
                        continue
                        
                    # Check if signature matches
                    if header[offset:offset + len(signature)] == signature:
                        return True
                        
            # No signatures matched
            return False
            
        except (IOError, OSError, IndexError):
            return False
